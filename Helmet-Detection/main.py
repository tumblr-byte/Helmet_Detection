import string
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os
import numpy as np
import datetime
from ultralytics import YOLO
import onnxruntime as ort
from deep_sort_realtime.deepsort_tracker import DeepSort
import warnings
warnings.filterwarnings('ignore')

# Paths - set your own
CRNN_MODEL_PATH = ""  
YOLO_MODEL_PATH = ""  
VIDEO_PATH = ""       

# Output setup
current_time = datetime.datetime.now().strftime("%H-%M-%S")
output_folder = "track_output"
track_folder = os.path.join(output_folder, current_time)
os.makedirs(track_folder, exist_ok=True)

# Load models
crnn_session = ort.InferenceSession(CRNN_MODEL_PATH)
yolo_model = YOLO(YOLO_MODEL_PATH, task='detect')

# Class names for YOLO
class_names = ["bike", "person with helmet", "number plate", "person without helmet"]

# DeepSORT tracker for bike tracking
tracker = DeepSort(max_age=50, n_init=3, max_iou_distance=0.7)

# Character set for OCR
CHARS = string.ascii_lowercase + string.ascii_uppercase + string.digits
char_to_int = {char: idx + 1 for idx, char in enumerate(CHARS)}
int_to_char = {idx: char for char, idx in char_to_int.items()}

# Image transforms for CRNN
test_transforms = A.Compose([
    A.Resize(100 , 200),
    A.Normalize(mean = (0.485 , 0.456 , 0.406) , std = (0.229 , 0.224 , 0.225)),
    ToTensorV2(),

])


def ctc_decode(logits, int_to_char):
    """Decode CTC output logits to readable text"""
    max_probs = torch.softmax(logits, dim=2)
    max_indices = torch.argmax(max_probs, dim=2)
    decoded_strings = []
    
    for seq in max_indices:
        prev = -1
        decoded = []
        for idx in seq:
            idx = idx.item()
            if idx != prev and idx != 0:
                decoded.append(int_to_char[idx])
            prev = idx
        decoded_strings.append("".join(decoded))
    
    return decoded_strings


def run_ocr(crnn_session, image_array, test_transforms, int_to_char):
    """Run OCR on number plate crop"""
    image_rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
    img_tensor = test_transforms(image = image_rgb)["image"]
    img_tensor = img_tensor.unsqueeze(0).cpu().numpy()
    
    ort_inputs = {crnn_session.get_inputs()[0].name: img_tensor}
    ort_outputs = crnn_session.run(None, ort_inputs)
    output_tensor = torch.tensor(ort_outputs[0])
    pred = ctc_decode(output_tensor, int_to_char)[0]
    
    return pred


def predict_class(model, frame):
    """YOLO detection"""
    results = model.predict(frame, verbose=False, conf=0.7)
    detections = []
    
    for result in results:
        for box in result.boxes:
            detection = {
                "confidence": float(box.conf.cpu().numpy()[0]),
                "class_id": int(box.cls.cpu().numpy()[0]),
                "class_name": class_names[int(box.cls.cpu().numpy()[0])],
                "bbox": {
                    "x1": int(box.xyxy.cpu().numpy()[0][0]),
                    "y1": int(box.xyxy.cpu().numpy()[0][1]),
                    "x2": int(box.xyxy.cpu().numpy()[0][2]),
                    "y2": int(box.xyxy.cpu().numpy()[0][3])
                }
            }
            detections.append(detection)
    
    return detections


def save_crop(output_folder, track_id, class_name, bbox, frame):
    """Save cropped image"""
    x1, y1, x2, y2 = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
    
    if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0:
        return None
    
    cropped_image = frame[y1:y2, x1:x2]
    
    if cropped_image.size == 0:
        return None
    
    filename = os.path.join(output_folder, f"bike_{track_id}_{class_name}.jpg")
    cv2.imwrite(filename, cropped_image)
    
    return filename


def process_video(yolo_model, crnn_session, test_transforms, int_to_char, output_folder, video_path):
    """Main video processing loop"""
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_video_path = "output_video.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    violation_counter = {}
    VIOLATION_THRESHOLD = 5 
    saved_items = {}
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Get detections from YOLO
        detections = predict_class(yolo_model, frame)
        
        bike_detections = []
        other_detections = []
        
        # Separate bikes from other objects
        for detection in detections:
            bbox = detection["bbox"]
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            width_box = x2 - x1
            height_box = y2 - y1
            class_name = detection["class_name"]
            
            if class_name == "bike":
                bike_detections.append(([x1, y1, width_box, height_box], detection["confidence"], 'bike'))
            else:
                other_detections.append(detection)
        
      
        tracks = tracker.update_tracks(bike_detections, frame=frame)
        violation_detected = any(d["class_name"] == "person without helmet" for d in other_detections)
        
        # Process tracked bikes
        for track in tracks:
            if track.is_confirmed():
                track_id = track.track_id
                ltrb = track.to_ltrb()
                bike_bbox = {
                    "x1": int(ltrb[0]),
                    "y1": int(ltrb[1]),
                    "x2": int(ltrb[2]),
                    "y2": int(ltrb[3])
                }
                x1, y1, x2, y2 = map(int, ltrb)
                
                # Update violation counter
                if violation_detected:
                    violation_counter[track_id] = violation_counter.get(track_id, 0) + 1
                else:
                    violation_counter[track_id] = 0
                
                is_confirmed_violation = violation_counter[track_id] >= VIOLATION_THRESHOLD
                
                # Initialize saved items tracking
                if track_id not in saved_items:
                    saved_items[track_id] = {'bike': False, 'person': False, 'plate': False}
                
               
                if is_confirmed_violation:
                    color = (0, 0, 255)  
                    label = f"bike_id:{track_id} - VIOLATION"
                else:
                    color = (0, 255, 0)  
                    label = f"bike_id:{track_id}"
                
                thickness = max(2, int(min(width, height) / 400))
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, width/1600, color, thickness)
                
               
                if is_confirmed_violation and not saved_items[track_id]['bike']:
                    save_crop(track_folder, track_id, "bike", bike_bbox, frame)
                    saved_items[track_id]['bike'] = True
        
        # Process other detections (person, helmet, plate)
        for detection in other_detections:
            bbox = detection["bbox"]
            x1, y1, x2, y2 = bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]
            class_name = detection["class_name"]
            confidence = detection["confidence"]
            
          
            thickness = max(2, int(min(width, height) / 400))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), thickness)
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, width/1600, (255, 0, 0), thickness)
            
            # Find which bike this detection belongs to
            for track in tracks:
                if track.is_confirmed():
                    track_id = track.track_id
                    ltrb = track.to_ltrb()
                    bx1, by1, bx2, by2 = map(int, ltrb)
                    
                   
                    if not (x2 < bx1 or x1 > bx2 or y2 < by1 or y1 > by2):
                        is_violation = violation_counter.get(track_id, 0) >= VIOLATION_THRESHOLD
                        
                        if is_violation:
                            # Save person without helmet
                            if class_name == "person without helmet" and not saved_items[track_id]['person']:
                                save_crop(track_folder, track_id, "person_no_helmet", bbox, frame)
                                saved_items[track_id]['person'] = True
                            
                            # Save number plate
                            if class_name == "number plate" and not saved_items[track_id]['plate']:
                                save_crop(track_folder, track_id, "number_plate", bbox, frame)
                                saved_items[track_id]['plate'] = True
                        
                        break  
            
            # Run OCR on number plates
            if class_name == "number plate" and confidence >= 0.5:
                number_plate_crop = frame[y1:y2, x1:x2]
                try:
                    pred = run_ocr(crnn_session, number_plate_crop, test_transforms, int_to_char)
                    cv2.putText(frame, pred, (x1, y2 + 30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 123, 124), 3)
                except:
                    pass
        
        out.write(frame)
    
    cap.release()
    out.release()
    print(f"Done! Output saved to {output_folder}")


# Start processing
process_video(yolo_model=yolo_model, crnn_session=crnn_session, test_transforms=test_transforms, int_to_char=int_to_char, output_folder=track_folder, video_path=VIDEO_PATH)
