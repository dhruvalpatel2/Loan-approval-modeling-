Loan Default Prediction Project
Overview
This project aims to predict the default risk of loans using various machine learning models, including Random Forest, Neural Networks (CNN and LSTM), and Convolutional Neural Networks (CNNs). The dataset contains information about customers, their financial history, and loan characteristics. The main objective is to classify loans into "Default" or "No Default" categories.

Dataset
The dataset consists of several features, including:

customer_id: Unique identifier for each customer.
customer_age: Age of the customer.
customer_income: Annual income of the customer.
home_ownership: Ownership status of the customer's home (Rent, Own, Mortgage).
employment_duration: Duration of employment in months.
loan_intent: Purpose of the loan.
loan_grade: Credit grade assigned to the loan.
loan_amnt: Loan amount in GBP.
loan_int_rate: Interest rate of the loan.
term_years: Loan term in years.
historical_default: Whether the customer has defaulted before (Y/N).
cred_hist_length: Length of credit history in years.
Current_loan_status: Target variable indicating the current loan status (Default/No Default).
Data Cleaning and Preprocessing
Handling Missing Values:

Imputed missing values for employment_duration and loan_int_rate with the median.
Replaced missing values in historical_default with 'N' for no default.
Feature Engineering:

Created new features like loan_to_income_ratio, age_group, and interaction terms between loan_intent and cred_hist_length.
Binned loan amount and interest rate into categories for better interpretability.
Data Encoding and Normalization:

Encoded categorical variables using OneHotEncoder and LabelEncoder.
Standardized the data using StandardScaler to have mean 0 and variance 1.
Exploratory Data Analysis (EDA)
Correlation Analysis: Created a detailed correlation heatmap to understand the relationships between different features.
Distribution Analysis: Analyzed the distribution of loan amounts and interest rates after filtering outliers.
Relationship Plots: Created scatter plots and boxplots to explore relationships between variables like loan_amnt, loan_int_rate, and customer_income.
Machine Learning Models
Random Forest Classifier
Cross-Validation: Performed K-fold cross-validation to evaluate the model's performance.
Hyperparameter Tuning: Used GridSearchCV for finding the best hyperparameters.
Evaluation Metrics: Achieved a cross-validation accuracy of ~92.6%. The ROC-AUC score was ~92.4%.
Neural Networks (Sequential Model)
Model Architecture:
Input layer followed by two hidden layers with ReLU activation and dropout regularization.
Output layer with a sigmoid activation function for binary classification.
Evaluation Metrics: Achieved an accuracy of ~88.8% and an ROC-AUC score of ~89.2%.
Convolutional Neural Networks (CNN) and Long Short-Term Memory (LSTM) Networks
Model Architecture:
Combined LSTM and CNN layers to capture temporal dependencies and extract spatial features from the data.
Fully connected layers followed by dropout regularization.
Evaluation Metrics: Achieved an accuracy of ~90.5% and an ROC-AUC score of ~90.1%.
Model Evaluation
Confusion Matrix: Analyzed the performance of models using confusion matrices and classification reports.
ROC Curve: Plotted ROC curves to visualize the trade-off between true positive rate and false positive rate.
Future Work
Model Optimization: Further hyperparameter tuning using RandomizedSearchCV and GridSearchCV.
Feature Selection: Use SHAP values and permutation importance to identify the most significant features.
Model Deployment: Deploy the model using a REST API for real-time predictions.
