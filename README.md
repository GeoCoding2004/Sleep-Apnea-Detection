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
