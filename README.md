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
- **Normalization and Standardization**:
    - **Normalization (Min-Max Scaling)** decreased model accuracy. This may be because absolute values in medical datasets carry clinical meaning. Normalizing these values could remove critical scale differences.
    - **Standardization** also reduced accuracy. Some features (e.g., cholesterol) are more informative. By standardizing, these informative features were downplayed, reducing model effectiveness.
    - **Decision**: Avoid using normalization or standardization to retain important feature scales.
    
- **Feature Selection**:
    - Applied a **correlation matrix** to identify redundant features.
    - All features had similar correlations with the target variable, so **no columns were dropped**.
    
- **Grid Search Cross-Validation (GridSearchCV)**:
    - Optimized the KNN model by systematically testing different hyperparameters.
    
    - **Parameters Tuned**:
      - `n_neighbors` (number of neighbors)
      - `p` (distance metric: Manhattan or Euclidean)
      - `weights` (uniform vs. distance-based weighting)
    
    - **Best Parameters Found**:
      - `n_neighbors=21`, `p=1`, `weights='distance'`
    
    - **Performance**:
      - Improved **F1-Score: 0.73** (better balance between precision and recall).
    
    - **Outcome**: This improved model accuracy and ensured better generalization across datasets.

### **4. Future Steps**
- Experiment with other classification algorithms like **Logistic Regression**, **Decision Trees**, and **Random Forests**.
- Implement additional hyperparameter tuning and advanced feature engineering.
- Explore techniques like **SMOTE** to handle class imbalance (if present).
- Evaluate performance using more robust validation strategies such as **Stratified K-Folds**.
