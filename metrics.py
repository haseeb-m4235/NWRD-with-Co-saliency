

import os
import cv2
import numpy as np
from sklearn.metrics import precision_recall_fscore_support

# Define the paths to the directories containing ground truth and predicted masks
ground_truth_dir = 'C:\\Users\\hasee\\Desktop\\Semester Internship\\FineLine\\NWRD\\test\\masks'
predicted_dir = 'C:\\Users\\hasee\\Desktop\\Semester Internship\\300\\Results\\resnet18\\predictions'

# List all files in both directories
ground_truth_files = os.listdir(ground_truth_dir)
predicted_files = os.listdir(predicted_dir)

# Initialize lists to store results
f1_scores = []
recall_scores = []
precision_scores = []

# Loop through each pair of corresponding masks
for gt_file in ground_truth_files:
    if gt_file in predicted_files:
        gt_mask = cv2.imread(os.path.join(ground_truth_dir, gt_file), cv2.IMREAD_GRAYSCALE)
        predicted_mask = cv2.imread(os.path.join(predicted_dir, gt_file), cv2.IMREAD_GRAYSCALE)

        # Calculate precision, recall, and F1 score for each class
        precision, recall, f1, _ = precision_recall_fscore_support(gt_mask.flatten(), predicted_mask.flatten(), average='weighted')
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

# Calculate the mean scores
mean_precision = np.mean(precision_scores)
mean_recall = np.mean(recall_scores)
mean_f1_score = np.mean(f1_scores)

# Print the results
print(f"Mean Precision: {mean_precision:.2f}")
print(f"Mean Recall: {mean_recall:.2f}")
print(f"Mean F1 Score: {mean_f1_score:.2f}")

