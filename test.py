import torch
from torchvision import transforms
import torch
from PIL import Image
import glob
import os
import cv2
from torch import nn

model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False)
model.fc = nn.Linear(512, 2)

print("Using cuda" if torch.cuda.is_available() else "Using cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
#print("line 16")

custom_state_dict_path = "C:\\Users\\hasee\\Desktop\\Semester Internship\\FineLine\\BInaryClassifier\\Resnet\\resnet18_curriculum_patches_300_patch\\model_48.pth"  # Provide the path to your custom state dictionary
custom_state_dict = torch.load(custom_state_dict_path, map_location=torch.device('cpu'))
model.load_state_dict(custom_state_dict)
#print("line 20")

def predict_rust_or_nonrust(model, image_dir):
    transform_val = transforms.Compose([
    transforms.ToTensor(), 
    #transforms.ToPILImage(),        # Convert the image to a PIL image
    transforms.Resize((224, 224)),  # Resize the image to 224x224 pixels
    #transforms.ToTensor()           # Convert the image to a PyTorch tensor
])
    
    image_paths = glob.glob(os.path.join(image_dir, '*'))  
    predictions = {}
    #print("line 30")
    with torch.no_grad():
        for image_path in image_paths:
            image = Image.open(image_path)#.convert('RGB')
            image = transform_val(image).unsqueeze(0)  # Add batch dimension
            output = model(image).squeeze()
            # Assuming output is a tensor with two elements (binary classification)
            #positive_class_prob = output[1]  # Assuming index 1 represents the positive class
            pred_label = 1 if output[1] > output[0] else 0

            #pred_label = (output > 0.5).item()
            print(f'image path:{image_path} output:{output} prediction:{pred_label}')
            predictions[image_path] = 'rust' if pred_label == 1 else 'non-rust'
    return predictions

images_paths = glob.glob('C:\\Users\\hasee\\Desktop\\Semester Internship\\300\\test\\images\\*')
for image_path in images_paths:
    #print("line 43")
    predictions = predict_rust_or_nonrust(model, image_path)
    patch_count=0
    grp_count = 0
    for path, pred in predictions.items():
        if pred == 'rust':
            #print("line 49")
            ##print(f"{path}: {pred}")
            imageNo = path.split('\\')[-2]
            patchNo = path.split('\\')[-1].split('.')[0]
            img = cv2.imread(path)
            patch_count+=1
            if (patch_count%40==0):
                grp_count+=1
            output_path = f"C:\\Users\\hasee\\Desktop\\Semester Internship\\300\\binary\\{imageNo}\\{grp_count}\\{patchNo}.png"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, img)
    print(f"image no {imageNo} contians {patch_count} patches in {grp_count} groups")