# Project 5 — Ensemble Models for Wine Quality Prediction

**Author:** Deb St. Cyr  
**Course:** Applied Machine Learning  
**Date:** November 2025

## Overview

This project explores ensemble machine learning models to predict the quality of red wine based on eleven physicochemical measurements. The original quality score (0–10) is grouped into three classes:

- **Low** (3–4)
- **Medium** (5–6)
- **High** (7–8)

Ensemble methods such as Random Forest, Bagging, AdaBoost, Gradient Boosting, and Voting Classifiers were evaluated and compared using accuracy, F1 score, and train–test gap metrics.

## Key Results

The best-performing models were:

### **1. Random Forest (100 Trees)**
- Test Accuracy: **0.8875**
- Test F1 Score: **0.8661**
- Highest predictive performance overall
- Provides clear and interpretable feature importances

### **2. Gradient Boosting (100 Trees)**
- Test Accuracy: **0.8562**
- Test F1 Score: **0.8411**
- Better regularization and lower overfitting than bagged models

## Confusion Matrix Insights

Both models performed best on the dominant **medium** class. The **high** class saw the most misclassification, typically being predicted as medium due to overlapping chemical profiles and class imbalance.

## Important Features

Across models, the most influential features were consistently:

- **Alcohol**
- **Sulphates**
- **Volatile Acidity**
- **Density**
- **Citric Acid**

These findings align with known characteristics of wine quality assessment.

## Repository Structure
project05/
│
├── data/
│ ├── winequality-red.csv
│ └── winequality.names
│
├── ensemble_stcyr.ipynb
└── README.md


## Conclusion

Tree-based ensemble models are highly effective for predicting wine quality.  
Random Forest offers the best accuracy, while Gradient Boosting provides strong generalization and slightly better stability.

Future work could include hyperparameter optimization, cross-validation, class balancing, and testing more advanced algorithms such as XGBoost or CatBoost.

