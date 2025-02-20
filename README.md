# Heart Disease Prediction - Machine Learning Project

## **Project Overview**
This project involves building a machine learning model to predict the presence of heart disease based on a set of health-related features. The dataset contains numerical features, and the target variable is binary (1 for heart disease, 0 for no heart disease). The goal is to create a model that can accurately predict whether a patient has heart disease.

The dataset used for this project can be found on Kaggle: [Heart Disease Dataset](https://www.kaggle.com/datasets/yasserh/heart-disease-dataset/data).

## **Dataset Description**
- **Target Variable**: `target` (1 for heart disease, 0 for no heart disease)
- **Features**: Various numerical attributes including age, cholesterol levels, blood pressure, etc.
- **Usability Rating**: 10/10 (The dataset is clean, well-structured, and easy to work with.)

## **Steps Taken**

### **1. Data Preparation**
- **Data Loading**: The dataset was loaded into a Jupyter notebook for analysis.
- **Data Cleaning**: Column names were cleaned by renaming them to more readable and understandable formats, replacing abbreviations with full names.
- **Missing Data Check**: Checked the dataset for any missing values. There were no null values found, so no data imputation was required.
- **Feature Selection**: Examined the data types of the columns to determine which were suitable for the machine learning model. The target variable (`target`) was separated from the feature variables. Features were selected as all columns except `target`.

### **2. Model Development**
- **K-Nearest Neighbors (KNN) Model**: Used the **K-Nearest Neighbors (KNN)** classification algorithm to predict the target variable.
- **Model Evaluation**: Evaluated the model using several key metrics:
    - **Accuracy**: To assess overall model performance.
    - **Recall**: To measure the model's ability to correctly identify instances of heart disease.
    - **Precision**: To measure the model's ability to avoid false positives.
    - **F1-Score**: The harmonic mean of precision and recall, offering a balanced metric.
    - **Support**: The number of actual occurrences of each class in the dataset.

### **3. Advanced Techniques and Findings**

#### **Standardization Improved Model Performance**
- **Standardization (Z-Score Scaling)**: Improved model accuracy and reduced the impact of feature magnitudes.
    - Initially, the KNN model achieved an accuracy of **0.7368**.
    - After **standardization**, accuracy improved to **0.8553**.
- **Decision**: We proceeded with **standardization** as it significantly improved model performance.

#### **Feature Selection**
- Applied a **correlation matrix** to identify redundant features.
- All features had similar correlations with the target variable, so **no columns were dropped**.

#### **Grid Search Cross-Validation (GridSearchCV)**
- Optimized the KNN model by systematically testing different hyperparameters.

    **Parameters Tuned**:
  - `n_neighbors` (number of neighbors)
  - `p` (distance metric: Manhattan or Euclidean)
  - `weights` (uniform vs. distance-based weighting)

    **Best Parameters Found**:
  - `n_neighbors=21`
  - `p=1` (Manhattan distance)
  - `weights='uniform'`

    **Performance**:
  - **Best F1-Score**: 0.8593
  - **Outcome**: This slightly improved accuracy while **reducing precision** but **increased recall by nearly 10%**.
  
In this medical context, **recall is more important** because it reduces **false negatives**, meaning fewer cases of heart disease are missed.
