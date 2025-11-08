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

## Project 3 — Classifier Comparison Summary

This project compared three classification models (Decision Tree, Support Vector Machine, and Neural Network) using the Titanic dataset to predict passenger survival.  
Each model was trained and evaluated on three feature sets:  
1️⃣ `alone` (binary)  
2️⃣ `age` (continuous)  
3️⃣ `age + family_size` (combined numeric features)

### Summary of Model Performance (Test Data)

| Model / Case | Accuracy | Precision (1) | Recall (1) | F1 (1) |
|---------------|-----------|----------------|-------------|---------|
| DT — Case 1 (alone) | 0.63 | 0.51 | 0.58 | 0.54 |
| DT — Case 2 (age) | 0.59 | 0.55 | 0.29 | 0.41 |
| DT — Case 3 (age + fam) | 0.65 | 0.60 | 0.55 | 0.57 |
| SVC — Case 1 (alone) | 0.61 | 0.50 | 0.49 | 0.49 |
| SVC — Case 2 (age) | 0.60 | 0.55 | 0.40 | 0.46 |
| SVC — Case 3 (age + fam) | 0.67 | 0.64 | 0.59 | 0.61 |
| NN — Case 3 (age + fam) | 0.70 | 0.66 | 0.62 | 0.64 |

*(Values are rounded; actual metrics are in the notebook.)*

### Key Findings

- **Feature strength:** Using multiple features (`age` + `family_size`) improved accuracy and F1 across all classifiers.  
- **Decision Tree:** Easy to interpret but more prone to overfitting.  
- **SVM (RBF Kernel):** Better at nonlinear separation, producing modest gains in recall and F1.  
- **Neural Network (MLP):** Achieved the best overall performance, with balanced precision and recall, though less interpretable.

### Conclusion

The combined features produced the most reliable survival predictions.  
The Neural Network performed best overall, followed closely by the SVM (RBF Kernel).  
Future work may explore feature engineering (e.g., `fare_per_person` or `deck_level`), hyperparameter tuning, and k-fold cross-validation for improved generalization.

