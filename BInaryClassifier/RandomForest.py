from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from BinaryDataset import get_label
import glob
import cv2
import numpy as np
#from  imblearn import RandomOverSampler
model = RandomForestClassifier()
#sampler=RandomOverSampler()


train_data = get_label('C:\\Users\\hasee\\Desktop\\Semester Internship\\128\\train')

train_images = []
train_labels = []
#train_red_images = []
train_lab_images = []
# Use a loop to extract keys and values and populate the arrays
for key, value in train_data.items():
    image = cv2.imread(key)

    #red_channel = image[:, :, 2]
    #red_channel = np.squeeze(red_channel)
    #train_red_images.append(red_channel)

    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_image = lab_image.flatten()
    train_lab_images.append(lab_image)

    image = image.flatten()
    train_images.append(image)
    train_labels.append(value)

num_samples = len(train_images)  # Assuming both 'images' and 'labels' have the same length
print("NUmbe rof training samples:", num_samples)
index_array = np.arange(num_samples)
# Shuffle the index array randomly
np.random.shuffle(index_array)

# Use the shuffled index array to shuffle both 'images' and 'labels'
shuffled_train_images = [train_images[i] for i in index_array]
shuffled_train_lab_images = [train_lab_images[i] for i in index_array]
shuffled_train_labels = [train_labels[i] for i in index_array]

#shuffled_train_red_images = [train_red_images[i] for i in index_array]
#shuffled_train_red_images = [red_channel.reshape(-1) for red_channel in shuffled_train_red_images]


# Now, keys_array contains the keys, and values_array contains the values
#print(train_images[1])
#print(train_labels[1])

model.fit(shuffled_train_lab_images, shuffled_train_labels)

test_data = get_label('C:\\Users\\hasee\\Desktop\\Semester Internship\\128\\test')

test_images = []
test_labels = []
#test_red_images = []
test_lab_images = []

# Use a loop to extract keys and values and populate the arrays
for key, value in test_data.items():
    image = cv2.imread(key)
    
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab_image = lab_image.flatten()
    test_lab_images.append(lab_image)

    #red_channel = image[:, :, 2]
    #red_channel = np.squeeze(red_channel)
    #test_red_images.append(red_channel)

    image = image.flatten()
    test_images.append(image)
    test_labels.append(value)
#test_red_images = [red_channel.reshape(-1) for red_channel in test_red_images]

pred = model.predict(test_lab_images)

accuracy = accuracy_score(test_labels, pred)
report = classification_report(test_labels, pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)