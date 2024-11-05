
# Airline Passenger Satisfaction Prediction using Artificial Neural Network (ANN)

This project aims to predict airline passenger satisfaction based on various features using an Artificial Neural Network (ANN). The model is built using TensorFlow and Keras, and various data preprocessing and feature engineering techniques are applied to improve accuracy.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Code Overview](#code-overview)
  - [Data Preprocessing](#data-preprocessing)
  - [Feature Engineering](#feature-engineering)
  - [Building and Training the ANN Model](#building-and-training-the-ann-model)
  - [Evaluation](#evaluation)
- [Results](#results)
- [License](#license)

## Overview
The objective of this project is to classify passengers as either satisfied or dissatisfied based on demographic, travel, and service-related information. A neural network model is built to predict satisfaction levels, achieving high accuracy on the test set.

## Dataset
The dataset used is titled **Airline Passenger Satisfaction**. It includes features such as:
- Demographics: `Gender`, `Customer Type`, `Class`
- Travel details: `Type of Travel`, `Arrival Delay`, `Flight Distance`
- Service features: `Inflight WiFi Service`, `Seat Comfort`, `On-board Service`, etc.

The dataset has been cleaned, and categorical variables are encoded for model compatibility.

## Installation
To run this project, you will need to have Python installed with the following libraries:
- `pandas`
- `numpy`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `tensorflow`
  
You can install the required libraries using:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow
```

## Project Structure
- **airline_passenger_satisfaction.csv**: The dataset file.
- **main.py**: The main code file which includes data preprocessing, model building, and evaluation.
- **README.md**: Project documentation.

## Code Overview
### Data Preprocessing
1. **Load Dataset**: Load the CSV file using Pandas.
2. **Initial Analysis**: Display dataset statistics, check for null values, and basic information.
3. **Missing Values**: Replace missing values in the `Arrival Delay` column with the median.
4. **Label Encoding**: Convert categorical features (`Gender`, `Customer Type`, `Type of Travel`, `Class`, and `Satisfaction`) into numerical values using `LabelEncoder`.
5. **Outlier Removal**: Apply the Interquartile Range (IQR) method to remove outliers from numeric columns.

### Feature Engineering
1. **Feature and Target Variables**: Separate features (`X`) and target (`y`).
2. **Standardization**: Standardize the feature data using `StandardScaler`.

### Building and Training the ANN Model
1. **Model Architecture**:
   - Input layer with 32 neurons, ReLU activation.
   - Two hidden layers with 16 and 8 neurons respectively, using ReLU activation.
   - Output layer with 1 neuron and sigmoid activation for binary classification.
2. **Compile Model**: Use `binary_crossentropy` as the loss function and `Adam` as the optimizer with a learning rate of 0.001.
3. **Train Model**: Train the model for 25 epochs with a batch size of 32, using a 75-25 train-test split.

### Evaluation
1. **Predictions**: Predictions on the test set are thresholded at 0.5 to obtain binary values.
2. **Accuracy and Classification Report**: Calculate accuracy and display a classification report for performance metrics.
3. **Confusion Matrix**: Visualize the confusion matrix using Seaborn to assess true positives, false positives, true negatives, and false negatives.

## Results
The model's accuracy and classification report provide insight into its performance. The confusion matrix helps visualize how well the model distinguishes between satisfied and dissatisfied passengers.
