# Project 3 — Building a Classifier (Titanic)

**Notebook:** `ml03_stcyr.ipynb`  
**Goal:** Predict `survived` using three models (Decision Tree, SVM, Neural Network) across three feature sets (cases), evaluate performance, and reflect.

## Contents
1. Import & Inspect (Seaborn Titanic)
2. Data Prep (impute `age`, create `family_size`, encode categories)
3. Feature Selection  
   - Case 1: `alone`  
   - Case 2: `age`  
   - Case 3: `age`, `family_size`
4. Decision Tree: split, train, reports, confusion matrices, tree plots
5. SVC & NN: evaluation; SVC support vectors; NN 2D decision surface
6. Summary table + reflections

## How to Run
1. Open the repo folder in VS Code.
2. Start Jupyter / run notebook.
3. Execute cells top-to-bottom.

## Notes
- Uses `StratifiedShuffleSplit` for consistent train/test class balance.
- Align `y` with `X` indices after `dropna`.
- Optional experiments: try different SVC kernels, tune tree depth / MLP layers, add features (`sex`, `pclass`, `fare`), and standardize when adding continuous features for SVM/NN.

## Summary of Model Performance (Test Data)

| Model Type | Case | Features Used | Accuracy | Precision | Recall | F1-Score | Notes |
|-------------|------|---------------|-----------|------------|---------|-----------|-------|
|  Decision Tree | Case 1 | alone | 62.57% | 51.28% | 57.97% | 54.42% | - ||
|  | Case 2 | age | 61.45% | 50.00% | 17.39% | 25.81% | - |
|  | Case 3 | age + family_size | 59.22% | 46.15% | 34.78% | 39.67% | - |
| SVM (RBF Kernel) | Case 1 | alone | 62.57% | 51.28% | 57.97% | 54.42% | - |
|  | Case 2 | age | 63.13% | 71.43% | 7.25% | 13.16% | - |
|  | Case 3 | age + family_size | 63.13% | 71.43% | 7.25% | 13.16% | - |
| Neural Network (MLP) | Case 3 | age + family_size | 65.92% | 57.14% | 46.38% | 51.20% | - |
