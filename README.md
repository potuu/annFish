# annFish

Kaggle link: [https://www.kaggle.com/code/adilabdullayev/ann-model](https://www.kaggle.com/code/adilabdullayev/ann-model?scriptVersionId=203200824)](https://www.kaggle.com/code/adilabdullayev/ann-model?scriptVersionId=203200824)
# Önemli Bilgilendirme
Ana dosya "ann-model-fish-kaggle.ipynb" dosyası. Diğer, ann-model-fish.ipynb dosyası modeli kaydetme sürecinde çeşitli token ve yetkilendirme hataları aldığım için kaydettiğim ilk dosya.


# Fish Classification Model
## Overview
This project aims to classify various species of fish using a deep learning model. The dataset consists of images of different fish species, and the model is designed to recognize and classify these images into corresponding categories. It utilizes an Artificial Neural Network (ANN) architecture, which is a common approach for classification tasks.

## Table of Contents
- [Dataset Description](#dataset-description)
- [Environment Setup](#environment-setup)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluation Metrics](#evaluation-metrics)
- [Usage](#usage)
- [Conclusion](#conclusion)


## Dataset Description
The dataset is a large-scale fish dataset containing images of different species. Each species is stored in a separate directory, with the images being in PNG format. The dataset is organized as follows:


/Fish_Dataset
    ├── Species1
    │   ├── image1.png
    │   ├── image2.png
    │   └── ...
    ├── Species2
    │   ├── image1.png
    │   ├── image2.png
    │   └── ...
    └── ...
The dataset has been divided into training, validation, and test sets.

## Environment Setup
To run this project, you need the following libraries:

TensorFlow
Keras
Pandas
Matplotlib
scikit-learn

## Data Preprocessing
The data preprocessing steps include:

Loading the Dataset: The images and their corresponding labels are loaded into a Pandas DataFrame.
Splitting the Dataset: The dataset is split into training (80%) and test (20%) sets.
Image Augmentation: The training images are rescaled and split into training and validation sets using ImageDataGenerator.

## Model Architecture
The model is a feedforward neural network (ANN) designed with the following layers:

Input Layer: Flattens the input image (28x28x3).
Hidden Layers:
1st Layer: 1024 neurons with ReLU activation, L2 regularization, Batch Normalization, and Dropout.
2nd Layer: 512 neurons with ReLU activation and L2 regularization.
3rd Layer: 256 neurons with ReLU activation, L2 regularization, and Dropout.
4th Layer: 128 neurons with ReLU activation and L2 regularization.
5th Layer: 64 neurons with ReLU activation and Dropout.
6th Layer: 32 neurons with ReLU activation and L2 regularization.
7th Layer: 16 neurons with ReLU activation and L2 regularization.
Output Layer: 9 neurons with softmax activation for multi-class classification.

## Training the Model
The model is trained using the following parameters:

* Optimizer: Adagrad with a learning rate of 0.01.
* Loss Function: Categorical crossentropy.
* Metrics: Accuracy, Precision, Recall, and F1 Score.

### Callbacks
*Early Stopping: Stops training if validation loss does not improve for 5 epochs.
*Learning Rate Scheduler: Reduces the learning rate after 20 epochs.


## Evaluation Metrics
The model evaluates performance using:

* Accuracy: The proportion of correct predictions.
* Precision: The ratio of correctly predicted positive observations to the total predicted positives.
* Recall: The ratio of correctly predicted positive observations to all actual positives.
* F1 Score: The weighted average of Precision and Recall.


## Usage
To use this model:

1. Ensure the dataset is correctly set up.
2. Run the preprocessing script to prepare the data.
3. Define and compile the model.
4. Train the model using the training script.
5. Evaluate the model on the test set to measure performance.
Feel free to modify the model architecture and hyperparameters to improve performance based on your specific requirements!

## Conclusion
This project demonstrates how to build an image classification model using deep learning techniques with TensorFlow and Keras. With proper tuning and training, this model can effectively classify various fish species based on their images. The use of an ANN architecture provides a solid foundation for further experimentation and optimization.
