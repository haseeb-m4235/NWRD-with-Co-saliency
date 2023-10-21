import cv2
import numpy as np
import glob

def binarize_image(img, threshold=150):
    return (img > threshold).astype(np.uint8)

def compute_metrics(gt, pred):
    TP = np.sum((gt == 1) & (pred == 1))
    FP = np.sum((gt == 0) & (pred == 1))
    FN = np.sum((gt == 1) & (pred == 0))
    TN = np.sum((gt == 0) & (pred == 0))
    
    precision = TP / (TP + FP) if (TP + FP) != 0 else 0
    recall = TP / (TP + FN) if (TP + FN) != 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
    
    return f1, recall, precision

# Paths to your ground truth and prediction images
biggts = glob.glob('C:\\Users\\hasee\\Desktop\\Semester Internship\\128\\rust128\\test_masks128\\*') #C:\\Users\\hasee\\Desktop\\Semester Internship\\FineLine\\NWRD\\test\\masks\\*
bigpreds = glob.glob('C:\\Users\\hasee\\Desktop\\Semester Internship\\128\\rust128\\results\\testing with best CoCA\\*') #'C:\\Users\\hasee\\Desktop\\Semester Internship\\FineLine\\first_results\\*'
#print("Number of ground truth images:", count)
#print("Number of prediction images:", len(pred_paths))
# Make sure you have the same number of ground truth and prediction images
assert len(biggts) == len(bigpreds)
count = 0
total_f1, total_recall, total_precision = 0, 0, 0
for biggt, bigpred in zip(biggts, bigpreds):
    #print(biggt)
    #print(bigpred)
    
    biggt = glob.glob(biggt+"\\*")
    bigpred = glob.glob(bigpred+"\\*")
    assert len(biggt) == len(bigpred)
    for gt_path, pred_path in zip(biggt, bigpred):
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        
        gt_binary = binarize_image(gt_img)
        pred_binary = binarize_image(pred_img)
        
        f1, recall, precision = compute_metrics(gt_binary, pred_binary)
        print(pred_path, f1, recall, precision)
        total_f1 += f1
        total_recall += recall
        total_precision += precision
        count+=1

print("count", count)
# Calculate average metrics
avg_f1 = total_f1 / count
avg_recall = total_recall / count
avg_precision = total_precision / count

print("Average F1 Score:", avg_f1)
print("Average Recall:", avg_recall)
print("Average Precision:", avg_precision)