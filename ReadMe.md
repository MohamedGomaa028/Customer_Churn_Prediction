# Project Name:
## Customer Churn Prediction
# Project overview 
Customer churn prediction is a critical focus for businesses across all industries.
The goal of this project is to build a machine learning model that predicts whether a customer will continue their subscription or cancel it. By detecting at-risk customers early, businesses can take proactive measures to enhance retention and minimize revenue loss.

#### Source : (Telco Customer Churn)
Link : https://www.kaggle.com/datasets/blastchar/telco-customer-churn
#### Rows: 7043
#### Columns: 21
#### Target Variable: Churn (Yes)Or (No)
## Project Workflow:
### 1- Data Loading & Inspection

### 2-Exploratory Data Analysis (EDA)
2.1) Identify missing values (df.isna().sum())

2.2) Detect duplicates (df.duplicated().sum())

2.3) Visualize class imbalance in target variable

### 3- Data Preprocessing
3.1) Drop irrelevant columns 
3.2) Handle Missing data 
3.3) Convert categorical data from (Yes or No) --> (0 or 1)
3.4) Apply one-hot code for multi category features
3.5) Scale numerical features using StandardScaler
### 4- Feature Analysis
4.1) Check churn rate by each feature using groupby (churn_precentage() function)
### 5-Data Splitting
5.1) Split into Train (60%), Cross-validation (20%), Test (20%) using train_test_split
### 6- Modeling
#### 6.1) Logistic Regression
Train & evaluate baseline model

Compare metrics: Accuracy, MSE, Classification Report
#### 6.2)Logistic Regression + SMOTE
Handle class imbalance with SMOTE oversampling

Train logistic regression with balanced weights

#### 6.3) Random Forest Classifier
Hyperparameter tuning for min_samples_split, max_depth, n_estimators

Visualize accuracy curves

Select best parameters
#### 6.4)XGBoost
Train with early stopping

Compare train vs. test accuracy

Classification report
## Technologies Used
1) Python  
2) Pandas
3) Matplotlib
4) Scikit-learn  
5) XGBoost  
6) Jupyter Notebook  

## Author
### Name : Mohamed Gomaa 



