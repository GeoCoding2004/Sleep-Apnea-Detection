# Logistic Regression for Sleep Apnea Prediction

### This repository contains a Python implementation of a logistic regression model designed to predict the presence of sleep apnea based on input features extracted from patient data. The code includes data preprocessing, model training, and evaluation using validation and test datasets.

## Features
Input Features: The model uses the following features: <br>
Neck circumference (Neckcircum) <br>
Body Mass Index (BMI) <br>
STOP-BANG score (STOPBANG_total) <br>
Epworth Sleepiness Scale score (ESS>11) <br>

## Output Label:
The apnea-hypopnea index (AHI) is used to classify whether a patient has sleep apnea. <br>
A threshold of AHI >= 15.0 is used to label a patient as having sleep apnea (1) or not (0). <br>

## Data
The code expects an Excel file named datasheet.xlsx containing the following columns:
Neckcircum <br>
BMI <Br>
STOPBANG_total <br>
ESS>11 <br>
AHI <br>
Rows with missing values in the Neckcircum column are excluded during preprocessing. <br>

## Steps in the Code
### Data Preprocessing:
Reads the input features and output labels from the Excel file. <br>
Removes rows with missing Neckcircum values. <br>
Converts the AHI column into a binary classification label (0 or 1). <br>

### Data Splitting:
Splits the dataset into training (70%), validation (15%), and test (15%) sets using train_test_split. <br>

### Model Training:
Trains a logistic regression model using the training set. <br>

### Validation and Test Evaluation:
Validates the model using the validation set and calculates accuracy and mean squared error (MSE). <br>
Evaluates the final model on the test set and prints accuracy and MSE. <br>

### Model Insights:
Prints the model's coefficients and intercept for interpretability. <br>

### Outputs
Validation Accuracy and MSE: Evaluates model performance on unseen data. <br>
Test Accuracy and MSE: Final evaluation metrics for the model. <br>
Model Coefficients and Intercept: <br>
The learned parameters of the logistic regression model, which can be used to understand the influence of each feature. <br>
