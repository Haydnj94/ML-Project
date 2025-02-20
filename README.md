# Creating the updated README content as a string
updated_readme_content = """
# **Heart Disease Prediction - Machine Learning Project**

## **Project Overview**

This project involves building a machine learning model to predict the presence of heart disease based on a set of health-related features. The dataset contains numerical features, and the target variable is binary (1 for heart disease, 0 for no heart disease). The goal is to create a model that can accurately predict whether a patient has heart disease.

The dataset used for this project can be found on Kaggle: [Heart Disease Dataset](https://www.kaggle.com/datasets/yasserh/heart-disease-dataset/data).

## **Dataset Description**

- **Target Variable**: `target` (1 for heart disease, 0 for no heart disease)
- **Features**: Various numerical attributes including age, cholesterol levels, blood pressure, etc.
- **Usability Rating**: 10/10 (The dataset is clean, well-structured, and easy to work with.)

## **Dependencies**

The following Python libraries are required to run the code in this project:

- `pandas`: For data manipulation and analysis.
- `numpy`: For numerical computing.
- `scikit-learn` (specifically):
    - `train_test_split`: For splitting data into training and testing sets.
    - `KNeighborsClassifier`: For the K-Nearest Neighbors classification model.
    - `KNeighborsRegressor`: (While imported, it's not actually used in this classification project, consider removing it).
    - `GridSearchCV`: For hyperparameter tuning.
    - `DecisionTreeClassifier`: For the Decision Tree classification model.
    - `RandomForestClassifier`: For the Random Forest classification model.
    - `accuracy_score`, `precision_score`, `recall_score`: For model evaluation metrics.
    - `StandardScaler`, `MinMaxScaler`: For feature scaling (StandardScaler is used).
    - `export_text`, `plot_tree`: For visualizing the decision tree (though not used in the final model selection).
- `seaborn`: For data visualization (specifically the correlation matrix).
- `matplotlib`: For plotting (though not extensively used in the final model selection).

## **Steps Taken**

### **1. Data Preparation**

- **Data Loading**: The dataset was loaded into a Jupyter notebook for analysis.
- **Data Cleaning**: Column names were cleaned by renaming them to more readable and understandable formats, replacing abbreviations with full names.
- **Missing Data Check**: Checked the dataset for any missing values. There were no null values found, so no data imputation was required.
- **Feature Selection**: Examined the data types of the columns to determine which were suitable for the machine learning model. The target variable (`target`) was separated from the feature variables. Features were selected as all columns except `target`.

### **2. Model Development and Evaluation**

Several models were explored and evaluated, with a focus on maximizing recall due to the importance of minimizing false negatives in a medical context.

#### **K-Nearest Neighbors (KNN)**

- **Initial Model**:  A KNN model was initially trained and achieved an accuracy of 0.7368.
- **Standardization**: Feature standardization (Z-score scaling) significantly improved performance, boosting accuracy to 0.8553. This step was crucial for KNN due to its sensitivity to feature scales.
- **Hyperparameter Tuning (GridSearchCV)**:  GridSearchCV was used to optimize the following hyperparameters:
    - `n_neighbors` (number of neighbors)
    - `p` (distance metric: Manhattan or Euclidean)
    - `weights` (uniform vs. distance-based weighting)
- **Best Parameters**: `n_neighbors=21`, `p=1` (Manhattan distance), `weights='uniform'`
- **Best KNN Performance**:
    - Accuracy: 0.8684
    - Precision: 0.8444
    - Recall: 0.9268
    - F1-Score: 0.8833 (calculated for completeness)

#### **Decision Tree**

- **Model**: A decision tree model was trained to address potential overfitting in previous models.
- **Performance**:
    - Accuracy: 0.8026
    - Precision: 0.8250
    - Recall: 0.8049

#### **Random Forest**

- **Model**: A random forest model was trained to address the potential overfitting of the decision tree.
- **Performance**:
    - Accuracy: 0.8158
    - Precision: 0.8140
    - Recall: 0.8537
- **Hyperparameter Tuning (GridSearchCV)**:  
  GridSearchCV was used to optimize the following hyperparameters for the **Random Forest** model:
    - **`max_depth`**: Maximum depth of trees  
    - **`max_features`**: Number of features to consider at each split  
    - **`min_samples_leaf`**: Minimum samples required at a leaf node  
    - **`min_samples_split`**: Minimum samples required to split an internal node  
    - **`n_estimators`**: Number of trees in the forest  
- **Best Parameters**:
    - `max_depth=10`
    - `max_features='sqrt'`
    - `min_samples_leaf=4`
    - `min_samples_split=2`
    - `n_estimators=300`

- **Best Random Forest Performance**

After performing **GridSearchCV** with the optimized hyperparameters, the model achieved the following performance metrics:

    - **Accuracy**: 0.8421  
    - **Precision**: 0.8222  
    - **Recall**: 0.9024  

### **3. Conclusion**

While the Random Forest showed improvement over the single Decision Tree, the K-Nearest Neighbors model, after standardization and hyperparameter tuning, ultimately delivered the best performance, especially in terms of recall, which is the most critical metric for this project.  Therefore, the KNN model was selected as the final model for heart disease prediction.
