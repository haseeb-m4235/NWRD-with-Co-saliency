from torchvision import transforms
from BinaryDataset import get_label, BinaryDataset 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split, DataLoader
import torch
from PIL import Image
import glob
import os
import cv2


data = get_label('C:\\Users\\hasee\\Desktop\\Semester Internship\\128\\train')
    

# Assuming image_dict is already defined and CustomDataset is available

# 1. Preparing Data
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

dataset = BinaryDataset(image_dict=data, transform=transform)
print("Dataset loaded")

# Splitting dataset into training and validation sets (80-20 split as an example)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True)

# 2. Model Definition
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32*32*32, 256)
        self.fc2 = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))
        x = nn.functional.max_pool2d(x, 2)
        x = nn.functional.relu(self.conv2(x))
        x = nn.functional.max_pool2d(x, 2)
        x = x.view(-1, 32*32*32)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return self.sigmoid(x)

model = BinaryClassifier()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. Training Loop
best_val_acc = 0  # To keep track of the best validation accuracy
best_weights = None  # To store the best model weights
num_epochs = 15
for epoch in range(num_epochs):
    model.train()
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images).squeeze()
        loss = criterion(outputs, labels.float())
        loss.backward()
        optimizer.step()

    # Validation Loop
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images).squeeze()
            predicted = (outputs > 0.95).float()
            total += labels.size(0)
            correct += (predicted == labels.float()).sum().item()
    val_acc = 100 * correct / total
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_weights = model.state_dict()
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Accuracy: {val_acc:.2f}%, best accuracy is {best_val_acc}%")
    
model.load_state_dict(best_weights)

def predict_rust_or_nonrust(model, image_dir):
    transform = transforms.Compose([
        #transforms.Resize((128, 128)),
        transforms.ToTensor(),
        
    ])
    
    image_paths = glob.glob(os.path.join(image_dir, '*'))
    
    model.eval()
    predictions = {}
    with torch.no_grad():
        for image_path in image_paths:
            image = Image.open(image_path).convert('RGB')
            image = transform(image).unsqueeze(0)  # Add batch dimension
            output = model(image).squeeze()
            pred_label = (output > 0.95).item()
            predictions[image_path] = 'rust' if pred_label == 1 else 'non-rust'
    
    return predictions

images_paths = glob.glob('C:\\Users\\hasee\\Desktop\\Semester Internship\\128\\test\\nwrd_test_images_patches128\\*')
for image_path in images_paths:
    predictions = predict_rust_or_nonrust(model, image_path)
    patch_count=0
    grp_count = 0
    for path, pred in predictions.items():
        if pred == 'rust':
            #print(f"{path}: {pred}")
            imageNo = path.split('\\')[-2]
            patchNo = path.split('\\')[-1].split('.')[0]
            img = cv2.imread(path)
            patch_count+=1
            if (patch_count%40==0):
                grp_count+=1
            output_path = f"C:\\Users\\hasee\\Desktop\\Semester Internship\\128\\test\\binary\\{imageNo}\\{grp_count}\\{patchNo}.png"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, img)
    print(f"image no {imageNo} contians {patch_count} patches in {grp_count} groups")