# Red Wine Quality Prediction Using Machine Learning

## a. Problem Statement

The objective of this project is to predict the quality of red wine based on its physicochemical properties using Machine Learning techniques. The task is formulated as a binary classification problem where wine samples are classified as either Good Quality or Poor Quality.

Multiple Machine Learning models are implemented and compared to identify the best-performing model.

---

## b. Dataset Description

The dataset used is the Red Wine Quality dataset.

- Source: UCI Machine Learning Repository  
- Number of samples: 1599  
- Number of features: 11  
- Target variable: Quality  

Input features include:

- Fixed acidity  
- Volatile acidity  
- Citric acid  
- Residual sugar  
- Chlorides  
- Free sulfur dioxide  
- Total sulfur dioxide  
- Density  
- pH  
- Sulphates  
- Alcohol  

The quality column is converted into binary form:
- Good quality: quality ≥ 7  
- Poor quality: quality < 7  

---

## c. Models Used and Evaluation Metrics

The following Machine Learning models were implemented:

1. Logistic Regression  
2. K-Nearest Neighbors  
3. Naive Bayes  
4. Decision Tree  
5. Random Forest  
6. XGBoost  

The models were evaluated using:

- Accuracy  
- AUC  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

---

## Model Comparison Table

| ML Model Name       | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------      |----------|-----|-----------|--------|----|-----|
| Logistic Regression |          |     |           |        |    |     |
| KNN                 |          |     |           |        |    |     |
| Naive Bayes         |          |     |           |        |    |     |
| Decision Tree       |          |     |           |        |    |     |
| Random Forest       |          |     |           |        |    |     |
| XGBoost             |          |     |           |        |    |     |

---

## Conclusion

Multiple models were trained and evaluated. Based on comparison metrics, Random Forest/XGBoost showed better overall performance. The best model was selected for prediction in the application.

---

## How to Run

1. Install required libraries: pip install -r requirements.txt

2. Run application: python app.py

3. Notebook implementation is available inside the Model folder.
