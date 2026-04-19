# Pumpkin Price Classification using Multiple Machine Learning Models

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

---

### Phase 3: Encoding
'''python
LabelEncoder()
'''
### Why?
- Convert categorical text → numeric
- Required for ML models

---

Phase 4: Feature Engineering
Average Price
'''python
data_encoded['Avg Price'] = (Low Price + High Price) / 2
'''

Price Categorization
Instead of regression → classification:
'''python
Low / Medium / High (based on quantiles)
'''

### Why?
- Converts regression problem into classification
- Balanced classes using quantiles

--- 

### Phase 5: Train/Test Split
'''python
train_test_split(test_size=0.2, random_state=42)
'''

### Why?
- 80% training / 20% testing
- Ensures reproducibility

---
### Phase 6: Models Implementation
1. Random Forest
'''python
RandomForestClassifier(random_state=42)
'''
Strengths :
- High accuracy
- Reduces overfitting
- Feature importance available

2. SVM
'''python
SVC()
'''
Important:
- Requires scaling (StandardScaler used)
Strengths:
- Strong decision boundaries
- Good for complex data

3. KNN
 '''python
KNeighborsClassifier()
'''
Important:
- Requires scaling
Strengths:
- Simple logic
- No training phase

4. Naive Bayes
'''python
GaussianNB()
'''
Strengths:
- Fast
- Works well with probabilistic data

---

### Phase 7: Model Comparison
Accuracy Comparison

Displayed using bar plot:
<img width="1001" height="530" alt="image" src="https://github.com/user-attachments/assets/2302db4f-9951-4a5e-b643-069c5eb6ed09" />

---

### Phase 8: Confusion Matrix
Each model evaluated using:
rf_tuned = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
Insights:
- Shows correct vs incorrect predictions
- Helps understand model confusion between classes

<img width="1331" height="1190" alt="image" src="https://github.com/user-attachments/assets/0af2fe93-519c-4dbf-9c69-0c56b3a17ba7" />

---

### Phase 9: Hyperparameter Tuning
1. Random Forest (Tuned)
'''python
rf_tuned = RandomForestClassifier(
    n_estimators=200,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
'''
Result:
- Improved generalization
- Reduced overfitting
- accuracy_tuned =  0.9659
- 

2. SVM (Tuned)
'''python
C=10, gamma=0.1
'''
Result:
- Better margin control
- Improved accuracy
- accuracy_tuned = 0.9204545454545454

4. KNN (Tuned)
'''python
n_neighbors=7
weights='distance'
'''
Result:
- Better neighbor weighting
- More stable predictions
- accuracy_tuned = 0.9630681818181818

4. Naive Bayes (Tuned)
'''python
var_smoothing=1e-8
'''
Result:
- Slight stability improvement
- accuracy_tuned = 0.53125

### Final Results Summary
1. Random Forest (Tuned): 0.9659
2. KNN (Tuned): 0.9630681818181818
3. SVM (Tuned): 0.9204545454545454
4. NB (Tuned): 0.53125

### Key Insights

# Most Important Features:
- Package (strongest predictor)
- Variety
- City Name
- Item Size
  
# Observations:
- Random Forest achieved the best overall performance, making it the most reliable model for this dataset.
- KNN performed very closely to Random Forest, showing strong capability after tuning and scaling.
- SVM delivered good results but was slightly behind tree-based and distance-based models, indicating moderate suitability for this dataset.
- Naive Bayes showed significantly lower accuracy, suggesting that its probabilistic assumptions are not well-suited for this feature distribution.

# Important Notes
- Scaling is required for SVM and KNN
- Feature engineering significantly improved performance
- Hyperparameter tuning improved generalization

# Final Conclusion

This project demonstrates:
- Strong comparison between 4 ML models
- Importance of preprocessing & encoding
- Impact of scaling on model performance
- Effectiveness of hyperparameter tuning

# Final Insight:
Random Forest is the best performing model for pumpkin price classification in this dataset.
















