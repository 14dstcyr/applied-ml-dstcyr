
# Project 4 — Predicting Titanic Fare Using Regression

This project applies multiple regression modeling techniques to predict the fare a Titanic passenger paid, using features from the well-known Titanic dataset. The assignment focuses on comparing linear and nonlinear approaches and evaluating model performance using R², RMSE, and MAE.

## Objectives
- Prepare the Titanic dataset for regression modeling
- Engineer meaningful features
- Compare prediction performance across multiple feature sets
- Evaluate Linear Regression and alternative models (Ridge, Elastic Net, Polynomial)
- Identify the strongest predictor set and best model type
- Reflect on challenges, model behavior, and potential improvements

## Data Preparation
Key preparation steps included:

- Filling missing ages with the median  
- Dropping rows with missing fare  
- Creating a new feature: **family_size**  
- Converting **sex** to numeric using `.astype('category').cat.codes`  
- Selecting only relevant features for regression  
- Splitting data into training and testing sets  

## Models Compared
### Section 4 – Linear Regression Cases:
1. Age only  
2. Family size only  
3. Age + family size  
4. Pclass + sex (best case)

### Section 5 – Alternative Models Using Best Case:
- Linear Regression  
- Ridge Regression  
- Elastic Net  
- Polynomial Regression (degree 3)

## Model Performance Summary

| Model                 | R²     | RMSE   | MAE   |
|----------------------|--------|--------|-------|
| Linear Regression     | 0.340  | 30.90  | 20.40 |
| Ridge Regression      | 0.340  | 30.89  | 20.36 |
| Elastic Net           | 0.369  | 30.23  | 19.18 |
| Polynomial (deg 3)    | 0.446  | 28.30  | 17.61 |

## Key Findings
- **Pclass and sex** are the strongest predictors of fare.
- **Polynomial regression** provided the best overall performance.
- **Elastic Net** gave small improvements over linear models.
- Models using only age or family size showed severe underfitting.

## Challenges
- Fare is highly skewed with extremely high-priced outliers.
- Some important features (deck, ticket groups) had missing or inconsistent values.
- Predictive power was limited by the simplified feature set.

## Future Improvements
- Add embarkation port, deck, or ticket group features.  
- Log-transform fare to reduce skew.  
- Try non-linear tree-based models (Random Forest, Gradient Boosting).  
- Explore more feature interactions.

