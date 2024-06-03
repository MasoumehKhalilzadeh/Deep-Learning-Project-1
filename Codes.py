#!/usr/bin/env python
# coding: utf-8

# In[1]:


import itertools
import cv2
import os
import time

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras.applications.efficientnet import EfficientNetB7  as PretrainedModel, preprocess_input
from tensorflow.keras.layers import Input, Dense, Flatten, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from glob import glob


# In[3]:


from zipfile import ZipFile

# Specify the path to the ZIP file
data_path = '/home/ec2-user/SageMaker/Data.zip'

# Specify the extraction directory
extraction_path = '/home/ec2-user/SageMaker/extracted_data'

# Extract the ZIP file
with ZipFile(data_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_path)

print('The dataset has been extracted.')


# In[4]:


from zipfile import ZipFile
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


# In[5]:


# Construct the path to the image sets
path = os.path.join(extraction_path, 'lung_colon_image_set/lung_image_sets')

# List the categories (classes)
classes = os.listdir(path)
classes


# In[6]:


# Display random images from each category
for cat in classes:
    image_dir = os.path.join(path, cat)
    images = os.listdir(image_dir)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Images for {cat} category', fontsize=20)

    for i in range(3):
        k = np.random.randint(0, len(images))
        img = np.array(Image.open(os.path.join(path, cat, images[k])))
        ax[i].imshow(img)
        ax[i].axis('off')

    plt.show()


# In[7]:


IMG_SIZE = 256
SPLIT = 0.2
EPOCHS = 5
BATCH_SIZE = 64


img_label = {}

for cat in classes:
    for key, value in img_label.items():
        if key in cat:
            break
    else:
        img_label[cat] = len(img_label)

print("Class to Label Mapping:", img_label)


# In[8]:


import pandas as pd

# Construct the path to the image sets
path = os.path.join(extraction_path, 'lung_colon_image_set/lung_image_sets')

# List the categories (classes)
classes = os.listdir(path)


# In[9]:


# Initialize a dictionary to store the count of images in each class
class_count = {}

# Count the number of images in each class
for cat in classes:
    image_dir = os.path.join(path, cat)
    images = os.listdir(image_dir)
    class_count[cat] = len(images)
    
    
# Create a Pandas DataFrame to store the results
result_df = pd.DataFrame(list(class_count.items()), columns=['Class', 'Number of Images'])

# Display the DataFrame
print(result_df)


# In[10]:


import matplotlib.pyplot as plt

# Construct the path to the image sets
path = os.path.join(extraction_path, 'lung_colon_image_set/lung_image_sets')

# List the categories (classes)
classes = os.listdir(path)

# Initialize a dictionary to store the count of images in each class
class_count = {}

# Count the number of images in each class
for cat in classes:
    image_dir = os.path.join(path, cat)
    images = os.listdir(image_dir)
    class_count[cat] = len(images)
    
    # Create a bar graph
plt.figure(figsize=(10, 6))
plt.bar(class_count.keys(), class_count.values(), color='skyblue')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Number of Images per Class of Plant Disease')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better visibility
plt.tight_layout()

# Show the plot
plt.show()


# In[11]:


# Get the total number of images in the dataset
total_images = sum(class_count.values())

# Display the total number of images in the dataset
print(f'Total number of images in the dataset: {total_images}')


# In[12]:


# Display the number of unique labels in the dataset
unique_labels = len(class_count)
print(f'Total number of unique labels in the dataset: {unique_labels}')


# In[13]:


X = []
Y = []

for i, cat in enumerate(classes):
    images = glob(f'{path}/{cat}/*.jpeg')

    for image in images:
        img = cv2.imread(image)
        X.append(cv2.resize(img, (IMG_SIZE, IMG_SIZE)))
        Y.append(i)

X = np.asarray(X)
one_hot_encoded_Y = pd.get_dummies(Y).values


# In[14]:


X_normalized = preprocess_input(X)

from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(
    X_normalized, one_hot_encoded_Y, test_size=SPLIT, random_state=2022
)

# Print the size of the training set and testing set
print("Size of the Training Set:", X_train.shape[0])
print("Size of the Validation Set:", X_val.shape[0])


# In[15]:


def build_model(input_shape, num_classes):
    # Load the pre-trained EfficientNetB7 model without the top layer
    base_model = PretrainedModel(weights='imagenet', include_top=False, input_shape=input_shape)

    # Freeze the pre-trained layers
    for layer in base_model.layers:
        layer.trainable = False

    # Add custom layers on top of the pre-trained model
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dense(256, activation='relu')(x)
    x = Dense(num_classes, activation='softmax')(x)

    # Create the final model
    model = Model(inputs=base_model.input, outputs=x)
    
    return model



# In[16]:


# Build the model
model = build_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=len(classes))


# In[18]:


# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# In[19]:


model.summary()


# In[20]:


# Define early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


# In[ ]:


# Train the model
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stopping]
)


# In[23]:


# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.90, 1])  # Set the y-axis limits
plt.legend(loc='lower right')
plt.show()


# In[24]:


# Plot training history for cross-entropy loss
plt.plot(history.history['loss'], label='train_loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Cross-Entropy Loss')
plt.legend(loc='upper right')
plt.show()


# In[25]:


from sklearn.metrics import classification_report

# Predict classes for validation set
Y_val_pred = model.predict(X_val)
Y_val_pred_classes = np.argmax(Y_val_pred, axis=1)

# Convert one-hot encoded labels back to original labels
Y_val_true_classes = np.argmax(Y_val, axis=1)

# Generate classification report
report = classification_report(Y_val_true_classes, Y_val_pred_classes, target_names=classes)

# Print classification report
print(report)


# In[26]:


from sklearn.metrics import confusion_matrix
import seaborn as sns

# Generate confusion matrix
conf_matrix = confusion_matrix(Y_val_true_classes, Y_val_pred_classes)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:




