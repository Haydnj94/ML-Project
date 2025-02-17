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

### **3. Future Steps**
- Experiment with different classification algorithms such as **Logistic Regression**, **Decision Trees**, and **Random Forests**.
- Explore hyperparameter tuning for the KNN model to optimize performance.
- Perform further evaluation with additional metrics or techniques (e.g., cross-validation) to improve model reliability.

---

## **How to Run the Code**
1. Clone this repository.
2. Install the necessary dependencies using `pip` or `conda`:
    - `pandas`
    - `numpy`
    - `scikit-learn`
    - `matplotlib`
    - `seaborn`
3. Open the `heart_disease_prediction.ipynb` file in Jupyter Notebook.
4. Run the code cells to load the data, clean it, and fit the KNN model.

---

Feel free to update this README as the project progresses. It provides an overview of the work completed so far, and outlines the next steps for further development.
