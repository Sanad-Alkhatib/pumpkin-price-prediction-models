# Pumpkin Price Classification using Multiple Machine Learning Models
# Random Forest vs. SVM vs. KNN vs. Naive Bayes

## Project Overview
This project focuses on classifying pumpkin prices into three categories:
- Low
- Medium
- High
using multiple machine learning algorithms to compare performance and select the best model.

- The project includes:
- Data preprocessing
- Feature engineering
- Multiple ML models
- Model comparison
- Visualization (Accuracy + Confusion Matrix)
- Hyperparameter tuning


### Goal
Build a machine learning pipeline that:
- Predicts pumpkin price category
- Compares multiple algorithms fairly
- Improves performance using tuning
- Provides clear interpretability and evaluation


### Models Used

This project compares 4 machine learning models:
- Random Forest Classifier
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes

### Why Multiple Models?
We used multiple algorithms to:
- Compare performance fairly
- Understand model behavior differences
- Avoid relying on a single approach
- Identify the best trade-off between accuracy and generalization


### Dataset
- Rows: 1757
- Columns: 26 → cleaned to 17
['City Name', 'Type', 'Package', 'Variety', 'Sub Variety', 'Date', 'Low Price', 'High Price', 'Mostly Low', 'Mostly High', 'Origin', 'Origin District', 'Item Size', 'Color', 'Unit of Sale', 'Repack', 'Unnamed: 25']
- This project uses the dataset from [Microsoft ML-For-Beginners](https://github.com/microsoft/ML-For-Beginners).

---

### Phase 1: Data Exploration (EDA)

'''python
print(df.head())
print(df.dtypes)
print(df.describe())
print(df.isnull().sum())
'''
### Why?
- Understand structure
- Detect missing values
- Explore feature types

---

### Phase 2: Data Cleaning

### Remove empty columns
'''python
data_cleaned = df.dropna(axis=1, how='all').copy()
'''
### Why?
- Remove useless features
- Reduce noise in data

### Fill missing values
Categorical
- fillna('Unknown')
  
Numerical
- median()
  
## Why?
- Median is robust against outliers
- Keeps dataset consistent












