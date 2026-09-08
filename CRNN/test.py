"""
Testing Script for CRNN Number Plate Recognition
Visualizes predictions on test image
"""
import numpy as np
import torch
import torch.nn as nn
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import string
import cv2
import matplotlib.pyplot as plt

CHARS = string.ascii_lowercase + string.ascii_uppercase + string.digits
char_to_int = {char:idx + 1 for idx , char in enumerate(CHARS)}
int_to_char = {idx:char for char , idx in char_to_int.items()}
num_classes = len(CHARS) + 1

valid_transforms = A.Compose([
    A.Resize(100 , 200),
    A.Normalize(mean = (0.485 , 0.456 , 0.406) , std = (0.229 , 0.224 , 0.225)),
    ToTensorV2(),

])

# Greedy CTC decoding: take argmax per time step, then collapse consecutive
# repeated characters and drop blank token (index 0)
def ctc_decode(logits, int_to_char):
    max_probs = torch.argmax(logits, dim=2)
    decoded_strings = []
    for seq in max_probs:
        prev = -1
        decoded = []
        for idx in seq:
            idx = idx.item()
            if idx != prev and idx != 0:
                decoded.append(int_to_char[idx])
            prev = idx
        decoded_strings.append("".join(decoded))
    return decoded_strings


# Reverses Normalize so images display correctly
def denormalize(img_tensor , mean= (0.485 , 0.456 , 0.406) , std= (0.229 , 0.224 , 0.225)):
    img = img_tensor.permute(1 , 2 , 0).cpu().numpy()  #CHW -> HWC
    img = (img * std) + mean 
    img = np.clip(img , 0 , 1) 
    return img

def visualize(model , test_loader  , int_to_char , device , num_img=5):
   model.eval()
   images_shown = 0
   fig , axes = plt.subplots(1 , num_img , figsize= (num_img * 3 ,3))
   if num_img == 1:
     axes = [axes]
   correct = 0
   total = 0

   with torch.no_grad():
     for images ,labels_concat , label_length in test_loader:
       images = images.to(device)
       outputs = model(images)
       label_lengths = label_length.tolist()
       #split the flat concatenated labels back into per-sample labels
       labels_split = torch.split(labels_concat , label_lengths)

       for i in range(images.size(0)):
         pred_text = ctc_decode(outputs[i].unsqueeze(0) , int_to_char)[0]
         true_text = "".join([int_to_char[idx.item()] for idx in labels_split[i]])
         total += 1
         if pred_text == true_text:
           correct += 1
         if images_shown < num_img:
           img_display = denormalize(images[i])
           axes[images_shown].imshow(img_display)
           axes[images_shown].set_title(f" GT: {true_text} \nPred: {pred_text}")
           axes[images_shown].axis("off")
           images_shown +=1


   plt.tight_layout()
   plt.show()

   accuracy = correct/total  * 100
   print(f"accuracy on test set: {accuracy:.2f}")
   return accuracy

model.load_state_dict(torch.load("best_model.pth"))
visualize(model , test_loader , int_to_char , device ,num_img= 5)


