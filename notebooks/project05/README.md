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

This project explores the effectiveness of ensemble machine learning models compared to traditional baseline classifiers. After preparing the dataset and establishing baseline performance with kNN and a Decision Tree, I evaluated several ensemble approaches including Random Forest, Gradient Boosting, Voting Classifiers, and Bagging.

The results demonstrated a clear improvement when using ensemble methods. Random Forest (100 trees) achieved the highest test accuracy at 0.8875, outperforming all baseline models. Gradient Boosting also performed well, reaching 0.8562. The Voting Classifier combining a Decision Tree, SVM, and Neural Network achieved 0.8594, improving on the baseline but not surpassing Random Forest.

This project highlights how ensemble techniques reduce variance, create smoother decision boundaries, and deliver more robust predictions. Random Forest was the top performing and most stable model in this analysis.
