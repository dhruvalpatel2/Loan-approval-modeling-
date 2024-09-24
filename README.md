# Loan Default Prediction Project

## Overview
This project aims to predict loan defaults using machine learning techniques. The dataset contains information on customer demographics, loan attributes, and historical loan performance. The goal is to build a predictive model that accurately identifies high-risk borrowers, which can be used by financial institutions to mitigate potential losses.

## Table of Contents
- [Project Description](#project-description)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Model Development](#model-development)
- [Model Evaluation](#model-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Future Work](#future-work)
- [Installation & Usage](#installation--usage)

## Project Description
The project leverages various machine learning algorithms such as Random Forest, Neural Networks, Convolutional Neural Networks (CNNs), and Long Short-Term Memory (LSTM) networks to predict whether a borrower will default on a loan. The primary objective is to provide a reliable tool for financial institutions to identify high-risk customers and take appropriate measures to mitigate risk.

## Dataset
The dataset used for this project contains the following features:
- **Customer Demographics:** Age, Income, Home Ownership, Employment Duration.
- **Loan Attributes:** Loan Amount, Loan Intent, Interest Rate, Loan Grade, Term Length.
- **Historical Data:** Credit History Length, Previous Default Status, Loan Status.

### Data Source
- The dataset is sourced from a public loan dataset (or specify the source if available).

## Data Preprocessing
- **Data Cleaning:** 
  - Removed or imputed missing values using the median strategy.
  - Standardized numerical columns and encoded categorical features using OneHotEncoder.
- **Feature Scaling:** 
  - Applied StandardScaler to normalize the data, ensuring all features contribute equally to the model.

## Feature Engineering
- **Income to Loan Ratio:** Created a feature representing the ratio of loan amount to customer income.
- **Age Groups:** Binned customers into categories such as Young, Middle-Aged, and Senior based on their age.
- **Loan Amount and Interest Rate Binning:** Categorized loan amounts and interest rates into groups to capture non-linear relationships.
- **Interaction Terms:** Created interaction features such as loan intent combined with credit history length to capture complex patterns.

## Model Development
- **Random Forest Classifier:**
  - Performed K-Fold Cross-Validation to evaluate model stability.
  - Achieved an average cross-validation score of 92.62%.
- **Neural Networks:**
  - Implemented a neural network with multiple dense layers and dropout for regularization.
  - Achieved an accuracy of 88.83% and a ROC-AUC score of 0.89.
- **CNN & LSTM Hybrid Model:**
  - Developed a hybrid model combining CNN and LSTM layers for capturing both temporal and spatial features in the data.
  - Achieved an accuracy of 89.46% and a ROC-AUC score of 0.89.

## Model Evaluation
- **Performance Metrics:**
  - Evaluated models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC.
- **Confusion Matrix:**
  - Analyzed true positives, false positives, true negatives, and false negatives to understand model performance.
- **ROC Curve:**
  - Visualized the trade-off between sensitivity and specificity for different models.

## Results
- The best-performing model was the Random Forest Classifier, with an accuracy of 92% and a ROC-AUC score of 0.92.
- The neural network model also performed well, but slightly lower than the Random Forest in terms of recall for defaulters.

## Conclusion
The project successfully built and evaluated multiple models for loan default prediction. The Random Forest model proved to be the most effective, followed closely by the neural network. The feature engineering efforts significantly contributed to improving model performance.

## Future Work
- **Feature Enrichment:** Explore additional features such as customer transaction history, loan repayment patterns, and external financial indicators.
- **Hyperparameter Tuning:** Implement advanced techniques like Bayesian Optimization for hyperparameter tuning.
- **Deployment:** Develop a user-friendly API for real-time loan default prediction.

## Installation & Usage
### Prerequisites
- Python 3.6 or higher
- Libraries: Pandas, NumPy, Scikit-Learn, TensorFlow, Matplotlib, Seaborn
