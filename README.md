# Deep-Learning-Project
Lung Cancer Prediction Using Deep Learning

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Development](#model-development)
5. [Experimental Results](#experimental-results)
6. [Discussion](#discussion)


## Introduction

This project presents a deep learning approach for lung cancer prediction using histopathological images. We design and evaluate transfer learning with the EfficientNetB7 convolutional neural network architecture pre-trained on ImageNet. The histopathological images are preprocessed to enhance their quality and standardized format. We propose a model architecture integrating global average pooling for feature extraction and fully connected layers for classification. Through extensive experimentation on a dataset of lung images, our approach demonstrates strong predictive performance in accurately identifying lung cancer presence. Additionally, we explore the performance of our model's predictions. Our research contributes to the field by offering a robust and interpretable framework for lung cancer prediction.


## Dataset

In this project, our dataset contains 15000 histopathological images obtained from Kaggle, a public data repository for datasets. This dataset is from a repository titled "Lung and Colon Cancer Histopathological Images" by Andrew Maas. These 15000 histopathological images are presented in three classes based on different lung conditions. These classes are normal lung tissue samples, lung tissue samples with adenocarcinomas which is a type of lung cancer, and lung tissue samples with squamous cell carcinomas which is another type of lung cancer. Each of these three classes contains 5000 images and has been developed by data augmentation from 250 images. So, there is no need to perform data augmentation on this data. Also, the dataset shows a balanced representation of the different lung conditions including 5000 images in each class.


```python

# Specify the extraction directory
extraction_path = 'extracted_data'

# Extract the ZIP file
with ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_path)
print('The data set has been extracted.')

# Construct the path to the image sets
path = os.path.join(extraction_path, 'lung_colon_image_set/lung_image_sets')

# List the categories (classes)
classes = os.listdir(path)
classes
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

```
![image](https://github.com/Masoumeh89/Deep-Learning-Project-1/assets/74910834/5a8a83c5-8f84-4805-8391-d0cf0dcf12b0)



## Data Preprocessing


**one hot encoding** which transforms categorical labels into a binary matrix where each class is represented by a unique column, and each row corresponds to a sample. 

```python
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

```
**Image resizing** has been performed by changing the size of the images to a uniform dimension of 224x224 pixels. This has been done by using the OpenCV library and the Numpy library in Python. This helps us to standardize the input dimension for the convolutional neural network (CNN) architecture.

**The number of epochs** which is defined as the many times the entire dataset will be passed forward and backward through the neural network during training is considered as 5 epochs. The dataset is divided into batches during training, and each batch is processed by the neural network before updating the weights. This constant defines the number of samples in each batch. A smaller batch size often provides more frequent updates but might be computationally expensive. Here, the batch size is 5 in this study.  



```python
IMG_SIZE = 224
SPLIT = 0.4
EPOCHS = 5
BATCH_SIZE = 16
```
**Images normalization** has been performed by using a preprocess_input function provided by the TensorFlow framework.

**The split,** which is the constant that determines the ratio in which our dataset will be split into training and validation sets is defined as 0.4. This split has been performed with a ratio of 60:40 to be sure that there is an adequate amount of data in both training and validation categories. So, we can say that the size of the training set is 9000 images, and the size of the validation set is 6000 images. 

```python
X_normalized = preprocess_input(X)

from sklearn.model_selection import train_test_split

X_train, X_val, Y_train, Y_val = train_test_split(
    X_normalized, one_hot_encoded_Y, test_size=SPLIT, random_state=2022
)

# Print the size of the training set and testing set
print("Size of the Training Set:", X_train.shape[0])
print("Size of the Validation Set:", X_val.shape[0])
```

Size of the Training Set: 9000
Size of the Validation Set: 6000


**Exploratory data analysis** has been performed on data to find the distribution of images according to the three classes. 

```python
import pandas as pd

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
    
    
# Create a Pandas DataFrame to store the results
result_df = pd.DataFrame(list(class_count.items()), columns=['Class', 'Number of Images'])

# Display the DataFrame
print(result_df)

```

<img width="179" alt="image" src="https://github.com/Masoumeh89/Deep-Learning-Project-1/assets/74910834/4aca2231-9ba1-41ff-a712-0bd0a15d51f8">




```python
img_label = {}

for cat in classes:
    for key, value in img_label.items():
        if key in cat:
            break
    else:
        img_label[cat] = len(img_label)
print("Class to Label Mapping:", img_label)
```

**Class to Label Mapping: {'lung_aca': 0, 'lung_n': 1, 'lung_scc': 2}**

```python
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



```
![image](https://github.com/Masoumeh89/Deep-Learning-Project-1/assets/74910834/f7ac775e-451c-4b35-bb03-70736a63ab19)




## Model Development

In this project, the model architecture was implemented using the Functional API of Keras. I have implemented EfficientNetB7 architecture as the base model for feature extraction. EfficientNetB7 is a convolutional neural network (CNN) architecture that is very efficient in image classification. There are some key components of the model architecture that have been used in this model:

**A global average pooling layer** is used to compute the average value of each feature map along its spatial dimensions (height and width). It replaces each 2D feature map with a single value, which is the average of all the values in that feature map. This operation results in a fixed-size output, where the length of the output vector is equal to the depth (number of channels) of the input feature maps, regardless of the input spatial dimensions. 

**Fully connected layers (also known as dense layers)** are added on top of the feature representation obtained from the pre-trained base model. 

**The final output layer** of the model produces soft probabilities for the target classes using a SoftMax activation function. These probabilities represent the likelihood of each class, allowing for multi-class classification of lung cancer.






```python
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
```


```python
# Build the model
model = build_model(input_shape=(IMG_SIZE, IMG_SIZE, 3), num_classes=len(classes))
```

```python

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

**Callback** is used to monitor the model improvement during each epoch. So, an early stopping is employed to prevent overfitting after optimizing the model using the Adam optimizer. We put callback as 90% which means the algorithm stops when it reaches the best performance to prevent overfitting. 

```python
import tensorflow as tf
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Define custom callback
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('val_accuracy') > 0.90:
            print('\nValidation accuracy has reached 90%, so stopping further training.')
            self.model.stop_training = True

# Instantiate custom callback
my_callback = myCallback()

# Instantiate standard callbacks
es = EarlyStopping(patience=3, monitor='val_accuracy', restore_best_weights=True)
lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, verbose=1)

```


```python
history = model.fit(X_train, Y_train,
validation_data = (X_val, Y_val),
batch_size = BATCH_SIZE,
epochs = EPOCHS,
verbose = 1,
callbacks = [es, lr, myCallback()])

```


## Experimental Results

In this training process, the final training accuracy was approximately 93% and a training loss was 0.1579. Also, the validation accuracy reached a high value of 96% with a validation loss of 0.0794. Our classification results, as shown in the classification report, underscore the performance of our model across all classes, including lung_scc, lung_aca, and lung_n. As can be seen from the results, the precision, recall, and F1-scores of our model demonstrate almost perfect performance. Achieving an overall accuracy of 97% on the validation dataset shows the robustness and efficacy of our proposed model. These results suggest that our model has a significant ability to diagnose and classify lung cancer which leads us to accurate patient care decisions.


**The model performance** is evaluated using evaluation metrics such as accuracy, precision, recall, and F1-score.

```python

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
```

<img width="444" alt="image" src="https://github.com/Masoumeh89/Deep-Learning-Project-1/assets/74910834/74a9df68-6de6-4559-bc6e-5236f195e83c">

<img width="413" alt="image" src="https://github.com/Masoumeh89/Deep-Learning-Project-1/assets/74910834/8bfe49e9-9bc4-4e1f-9459-64ef003697f4">



## Discussion

The results of our study demonstrate the great performance of our deep learning model in predicting lung cancer. The classification report shows high precision, recall, and F1-score values across all classes, including lung_scc, lung_aca, and lung_n. These metrics indicate the model's ability to accurately classify instances of each category. The overall accuracy of 97% on the validation dataset emphasizes the robustness of our model in distinguishing between different lung cancer categories. 
The success of our model can be because of several factors. Firstly, the utilization of transfer learning with the EfficientNetB7 architecture enables our model to leverage knowledge learned from a large-scale dataset (ImageNet) to effectively extract relevant features from medical images. Also, the global average pooling and dense layers are used to perform the transformation of extracted features into meaningful predictions, enhancing the model's performance. Furthermore, early stopping regularization helps prevent overfitting, ensuring that our model generalizes well to unseen data. 












































