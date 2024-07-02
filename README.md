# Deep-Learning-Project-1
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


![image](https://github.com/Masoumeh89/Deep-Learning-Project-1/assets/74910834/5a8a83c5-8f84-4805-8391-d0cf0dcf12b0)


## Data Preprocessing

**Label encoding** is to convert categorical labels into numerical values, and we use label encoding because most machine learning algorithms require numerical inputs. 

**Exploratory data analysis** has been performed on data to find the distribution of images according to the three classes. 

![image](https://github.com/Masoumeh89/Deep-Learning-Project-1/assets/74910834/f7ac775e-451c-4b35-bb03-70736a63ab19)


**Image resizing** has been performed by changing the size of the images to a uniform dimension of 224x224 pixels. This has been done by using the OpenCV library and the Numpy library in Python. This helps us to standardize the input dimension for the convolutional neural network (CNN) architecture.

 **Images normalization** has been performed by using a preprocess_input function provided by the TensorFlow framework.

**The split,** which is the constant that determines the ratio in which our dataset will be split into training and validation sets is defined as 0.4. This split has been performed with a ratio of 60:40 to be sure that there is an adequate amount of data in both training and validation categories. So, we can say that the size of the training set is 9000 images, and the size of the validation set is 6000 images. 

**The number of epochs** which is defined as the many times the entire dataset will be passed forward and backward through the neural network during training is considered as 5 epochs. The dataset is divided into batches during training, and each batch is processed by the neural network before updating the weights. This constant defines the number of samples in each batch. A smaller batch size often provides more frequent updates but might be computationally expensive. So, the batch size is 16 in this study.  

**one hot encoding** which transforms categorical labels into a binary matrix where each class is represented by a unique column, and each row corresponds to a sample. 




## Model Development

In this project, the model architecture was implemented using the Functional API of Keras. I have implemented EfficientNetB7 architecture as the base model for feature extraction. EfficientNetB7 is a convolutional neural network (CNN) architecture that is very efficient in image classification. There are some key components of the model architecture that have been used in this model:

**A global average pooling layer** is used to compute the average value of each feature map along its spatial dimensions (height and width). It replaces each 2D feature map with a single value, which is the average of all the values in that feature map. This operation results in a fixed-size output, where the length of the output vector is equal to the depth (number of channels) of the input feature maps, regardless of the input spatial dimensions. 

**Fully connected layers (also known as dense layers)** are added on top of the feature representation obtained from the pre-trained base model. 

**The final output layer** of the model produces soft probabilities for the target classes using a SoftMax activation function. These probabilities represent the likelihood of each class, allowing for multi-class classification of lung cancer.

**Callback** is used to monitor the model improvement during each epoch. So, an early stopping is employed to prevent overfitting after optimizing the model using the Adam optimizer. We put callback as 90% which means the algorithm stops when it reaches the best performance to prevent overfitting. 

**The model performance** is evaluated using evaluation metrics such as accuracy, precision, recall, and F1-score.


## Experimental Results

In this training process, the final training accuracy was approximately 93% and a training loss was 0.1579. Also, the validation accuracy reached a high value of 96% with a validation loss of 0.0794. Our classification results, as shown in the classification report, underscore the performance of our model across all classes, including lung_scc, lung_aca, and lung_n. As can be seen from the results, the precision, recall, and F1-scores of our model demonstrate almost perfect performance. Achieving an overall accuracy of 97% on the validation dataset shows the robustness and efficacy of our proposed model. These results suggest that our model has a significant ability to diagnose and classify lung cancer which leads us to accurate patient care decisions.


<img width="432" alt="image" src="https://github.com/Masoumeh89/Deep-Learning-Project-1/assets/74910834/b8def7e1-4309-4b8b-950d-bd28990e95b0">


![image](https://github.com/Masoumeh89/Deep-Learning-Project-1/assets/74910834/98767ef7-822a-4d2e-b63b-8d43c6dfbd3d)


## Discussion

The results of our study demonstrate the great performance of our deep learning model in predicting lung cancer. The classification report shows high precision, recall, and F1-score values across all classes, including lung_scc, lung_aca, and lung_n. These metrics indicate the model's ability to accurately classify instances of each category. The overall accuracy of 97% on the validation dataset emphasizes the robustness of our model in distinguishing between different lung cancer categories. 
The success of our model can be because of several factors. Firstly, the utilization of transfer learning with the EfficientNetB7 architecture enables our model to leverage knowledge learned from a large-scale dataset (ImageNet) to effectively extract relevant features from medical images. Also, the global average pooling and dense layers are used to perform the transformation of extracted features into meaningful predictions, enhancing the model's performance. Furthermore, early stopping regularization helps prevent overfitting, ensuring that our model generalizes well to unseen data. 












































