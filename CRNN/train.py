import numpy as np
import torch
import torch.nn as nn
import os
import random
from torch.utils.data import Dataset , DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import string
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt
from model import model , device



# Set random seeds for reproducibility ensures the train/valid/test split
# add augmentation randomness are the same across runs
torch.manual_seed(42)
random.seed(42)

CHARS = string.ascii_lowercase + string.ascii_uppercase + string.digits
char_to_int = {char:idx + 1 for idx , char in enumerate(CHARS)} #start at 1
int_to_char = {idx:char for char , idx in char_to_int.items()}
num_classes = len(CHARS) + 1 # +1 for the CTC blank token, reserved at index 0



# Training augmentations - targets real-world plate degradation
# (motion blur from moving vehicles, compression from low-quality camera feeds)
# to fix the clean-to-blur domain shift found during evaluation
train_transforms = A.Compose([
    A.Resize(100, 200),
    A.MotionBlur(blur_limit = (5  , 9), p=0.6),
    A.GaussianBlur(blur_limit = (3, 7) , p=0.5),
    A.RandomBrightnessContrast(p =0.3),
    A.ImageCompression(quality_lower = 50 , quality_upper = 90 ,p =0.4),
    A.Affine(rotate = (-5 , 5) , shear= (-5 , 5) , p=0.5) ,
    A.RandomShadow(p =0.2),
    A.Normalize(mean = (0.485 , 0.456 , 0.406) , std = (0.229 , 0.224 , 0.225)),
    ToTensorV2(), # converts HWC numpy array into PyTorch tensor 
])


valid_transforms = A.Compose([
    A.Resize(100 , 200),
    A.Normalize(mean = (0.485 , 0.456 , 0.406) , std = (0.229 , 0.224 , 0.225)),
    ToTensorV2(),

])


class Custom(Dataset):
   def __init__(self , folder_path , set_type  , char_to_int , transforms= None):
     super(Custom , self).__init__()
     self.folder_path = folder_path
     self.set_type = set_type
     self.transforms = transforms
     self.char_to_int = char_to_int

     files = [f for f in os.listdir(folder_path) if f.endswith((".jpeg" , ".png" , ".jpg"))]
     random.shuffle(files)
     n= len(files)

     train_end = int(0.8 * n)
     valid_end = train_end + int(0.15 * n)


     # 80/15/5 - train/valid/test split by index boundaries
     if set_type == "train":
       selected = files[:train_end]
     elif set_type == "valid":
       selected = files[train_end:valid_end]
     else:
       selected = files[valid_end:]

     self.files = [os.path.join(folder_path , f) for f in selected]
     self.labels = [os.path.splitext(f)[0] for f in selected]



   def __len__(self):
     return len(self.files)

   def __getitem__(self , idx):
      image_path = self.files[idx]
      image_bgr = cv2.imread(image_path)
      image_rgb = cv2.cvtColor(image_bgr , cv2.COLOR_BGR2RGB)

      if self.transforms:
         image_rgb = self.transforms(image = image_rgb)["image"]

      label_text = self.labels[idx]
      label = torch.tensor([self.char_to_int[char] for char in label_text] , dtype= torch.long)

      return image_rgb , label


# Custom collate for CTC loss: images are stacked normally, but labels
# are concatenated into one flat tensor ,
# since CTC takes label_lengths to know where each sample's label starts/ends
def ctc_collate(batch):
   images , labels = zip(*batch)
   images = torch.stack(images , dim=0)
   label_lengths = torch.tensor([len(label) for label in labels] , dtype = torch.long)
   labels_concat = torch.cat(labels , dim=0)
   return images , labels_concat , label_lengths


train_dataset = Custom(folder_path , "train" , char_to_int , train_transforms)
valid_dataset = Custom(folder_path , "valid" , char_to_int , valid_transforms)
test_dataset = Custom(folder_path , "test" , char_to_int  , valid_transforms)

train_loader = DataLoader(train_dataset , batch_size = 16 , shuffle= True , collate_fn = ctc_collate, num_workers = 2 , pin_memory = True)
valid_loader = DataLoader(valid_dataset , batch_size = 8 , shuffle = False , collate_fn= ctc_collate , num_workers = 2 , pin_memory = True)
test_loader = DataLoader(test_dataset , batch_size = 2 , shuffle = False , collate_fn = ctc_collate , num_workers = 2 , pin_memory = True)

# Train and validate the model with early stopping
def run_model(model, criterion, optimizer, train_loader, valid_loader,scheduler , num_epochs=100, patience=10, device="cuda", output_file="best_model.pth"):
    train_losses = []
    valid_losses = []
    best_valid_loss = np.inf
    counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        total_train = 0

        for images, labels, label_lengths in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            label_lengths = label_lengths.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            outputs = outputs.permute(1, 0, 2)

            input_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long).to(device)

            loss = criterion(
                outputs.log_softmax(2),
                labels,
                input_lengths,
                label_lengths
            )

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            total_train += images.size(0)

        avg_train_loss = train_loss / total_train
        train_losses.append(avg_train_loss)

        # Validation phase
        model.eval()
        valid_loss = 0
        total_valid = 0

        with torch.no_grad():
            for images, labels, label_lengths in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                label_lengths = label_lengths.to(device)

                outputs = model(images)
                outputs = outputs.permute(1, 0, 2)

                input_lengths = torch.full(size=(outputs.size(1),), fill_value=outputs.size(0), dtype=torch.long).to(device)

                loss = criterion(
                    outputs.log_softmax(2),
                    labels,
                    input_lengths,
                    label_lengths
                )

                valid_loss += loss.item() * images.size(0)
                total_valid += images.size(0)

        avg_valid_loss = valid_loss / total_valid
        valid_losses.append(avg_valid_loss)
        scheduler.step(avg_valid_loss)

        print(
            f"Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {avg_train_loss:.4f} "
            f"Valid Loss: {avg_valid_loss:.4f}"
            f"LR: {optimizer.param_groups[0]["lr"]:.2e}"
        )

        # Save best model and check for early stopping
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            counter = 0
            torch.save(model.state_dict(), output_file)
            print("Saved best model")
        else:
            counter += 1
            print(f"No improvement ({counter}/{patience})")

        if counter >= patience:
            print("Early stopping triggered")
            break

    return train_losses, valid_losses


# Define loss function and optimizer
criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-4, alpha=0.9, momentum=0.9, weight_decay=1e-4)


scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer , mode= "min" , factor = 0.5 , patience = 3)
# Start training
train_losses, valid_losses = run_model(model, criterion, optimizer, train_loader, valid_loader, scheduler ,num_epochs=100, patience=10, device=device, output_file="best_model.pth")
