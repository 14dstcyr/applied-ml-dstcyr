# Project 5 — Ensemble Models for Wine Quality Prediction

**Author:** Deb St. Cyr  
**Course:** Applied Machine Learning  
**Date:** November 2025

## Overview

This repository contains the Jupyter notebook, dataset, and documentation for Project 5 of the Applied Machine Learning course, focusing on ensemble methods for wine quality prediction.


## Dataset Summary

The wine quality dataset includes eleven physicochemical measurements such as acidity, alcohol, density, and sulphates. The original quality score (0–10) was grouped into three classes:

- **Low** (3–4)  
- **Medium** (5–6)  
- **High** (7–8)

These transformed labels help simplify the prediction problem while maintaining meaningful categories.

## Ensemble Methods Evaluated

- Random Forest  
- Gradient Boosting  
- AdaBoost  
- Bagging  
- Two Voting Classifiers (DT/SVM/NN and RF/LR/KNN)

Performance was compared using accuracy, F1 scores, and analysis of confusion matrices.

## Key Results

### **1. Random Forest (100 Trees) — Best Overall**
- **Test Accuracy:** 0.8875  
- **Test F1 Score:** 0.8661  
- Strongest predictive performance  
- Good interpretability due to feature importance scores  

### **2. Gradient Boosting (100 Trees)**
- **Test Accuracy:** 0.8562  
- **Test F1 Score:** 0.8411  
- Slightly more stable generalization than bagged models  
- Sensitive to learning rate and tree depth  

## Confusion Matrix Insights

The **medium** wine-quality class was predicted most accurately because it dominates the dataset and has more distinct chemical signatures.  
The **high** class experienced the most confusion, often being mislabeled as medium due to overlapping feature ranges and class imbalance.

## Most Influential Features

Across multiple ensemble models, these features consistently ranked highest:

- **Alcohol**  
- **Sulphates**  
- **Volatile Acidity**  
- **Density**  
- **Citric Acid**

These factors align well with known determinants of wine quality in real-world enology studies.

## Repository Structure
```
project05/
│
├── data/
│ ├── winequality-red.csv
│ └── winequality.names
│
├── ensemble_stcyr.ipynb
└── README.md
```

## How to Run

1. Clone the repository: git clone https://github.com/YOUR-USERNAME/applied-ml-dstcyr.git
2. Navigate to the project05 folder: cd applied-ml-dstcyr/notebooks/project05
3. Install dependencies from `requirements.txt` (if included)  
4. Open `ensemble_stcyr.ipynb` in Jupyter or VS Code  
5. Run all cells to reproduce the analysis


## Conclusion

Tree-based ensemble models are highly effective for predicting wine quality.  
Random Forest achieved the best accuracy and most stable results, while Gradient Boosting showed strong generalization and competitive performance.

Future improvements could include extensive hyperparameter tuning, stratified cross-validation, addressing class imbalance, or testing advanced gradient boosting methods such as XGBoost or CatBoost.
