# Project Summary – Regression Analysis (Medical Cost Dataset)

**Author:** Deb St. Cyr

## Dataset Overview

The Medical Cost dataset includes basic demographic and health-related information along with each person’s medical insurance charges. The features include age, sex, BMI, number of children, smoking status, and region. The target variable is *charges*, which is a continuous value, making regression the right choice for this project.

There were no missing values, so most of the preparation focused on exploring the data, encoding categorical features, and understanding how each variable contributes to medical charges.

---------------------------------------

### Exploration & Preprocessing

Before building any models, I created several visualizations to understand the data:

- **Histograms** to see distributions

- **Boxplots** to spot outliers

- **Count plots** for categorical features

- **A correlation heatmap** for numerical relationships

A few things stood out:

- Smokers have much higher charges compared to non-smokers.

- BMI has some outliers, but they appear valid.

- Region and sex don’t seem as influential on their own.

- Charges are heavily skewed because a small group of individuals have very high costs.

After that, I used one-hot encoding (get_dummies) to convert the categorical variables into numeric form. No other major cleaning steps were required.

------------------------------------------------

### Feature Selection

I included all available features in the model because medical costs are influenced by multiple factors. I expect **age**, **BMI**, and **smoker status** to be the strongest predictors, and the data exploration supported that. Region and sex may not be as strong, but they still provide additional context.

The target variable **(charges)** reflects actual medical costs, which makes it ideal for regression modeling.

------------------------------------------

### Models & Performance

I trained three models:

**1. Baseline Linear Regression**

- Test R² ≈ 0.78

- MAE ≈ $4,181

- RMSE ≈ $5,796

This gave me a strong baseline and showed that a simple model could already explain a good portion of the variance.

---------------------------------------------------

**2. Pipeline 1 – Scaled Linear Regression**

- Identical results to the baseline

- Scaling didn’t change anything, which makes sense for linear regression

---------------------------------------------------

**3. Pipeline 2 – Polynomial Regression (Degree 3)**

- Train R² ≈ 0.85

- Test R² ≈ 0.85

- Much lower MAE and RMSE

The polynomial model clearly captured more complex relationships without showing major signs of overfitting. This was the strongest model overall.

---------------------------------------------------

### Insights

- Smoking status is by far the biggest driver of higher medical charges.

- BMI and age also contribute meaningfully.

- Linear regression alone can only capture part of the pattern.

- Polynomial features helped the model generalize to patterns that weren’t purely linear.

If I continued this project, I would experiment with models like Random Forest, Gradient Boosting, and regularization methods such as Ridge and Lasso.

--------------------------------------------------------

### Reflection

This project helped me put everything together — from exploring the data to building and comparing different models. I’m still learning, but seeing how each step affects the results made things much clearer for me. The biggest surprise was how much better the polynomial model performed. It showed me that sometimes a simple linear model isn’t enough to capture what’s going on underneath the surface.