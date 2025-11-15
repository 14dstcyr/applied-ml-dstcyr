# P4 – Building a Regression Model

This folder contains the full work for the **P4: Building a Regression Model** assignment in the Applied Machine Learning course. The project follows the structure of the Howell regression example but applies the workflow to the **Titanic dataset**, including multiple feature cases, model evaluation, and documentation.

---

## Project Contents

* **p4_regression_model.ipynb** – Main notebook containing:

  * Introduction
  * Data loading and exploration
  * Feature engineering for Cases 1–4
  * Regression modeling (Linear, Ridge, ElasticNet, Polynomial)
  * Evaluation metrics and comparison table
  * Reflections section with screenshots

* **index.md** – Documentation file linked to the hosted site.

* **images/** – Folder for all screenshots included in Section 6 of the notebook.

---

## Project Overview

The goal of this assignment is to build and compare several regression models on a continuous target variable. The Titanic dataset was used to predict **Fare**, and multiple feature sets (Cases 1–4) were engineered to explore how additional features influence model performance.

The following models were evaluated:

* Linear Regression
* Ridge Regression
* ElasticNet
* Polynomial Regression (degree 2)

Evaluation metrics include:

* MAE (Mean Absolute Error)
* MSE (Mean Squared Error)
* RMSE (Root Mean Squared Error)
* R² (Coefficient of Determination)

---

## Feature Cases Summary

**Case 1 – Base Features:**

* `pclass`, `sibsp`, `parch`

**Case 2 – Engineered Feature:**

* Adds `FamilySize = sibsp + parch + 1`

**Case 3 – Additional Engineered Feature:**

* Adds `FarePerPerson = fare / FamilySize`

**Case 4 – Best Feature Set:**

* Selected based on performance from Cases 1–3

---

## Summary Table

A complete summary table comparing all models across all four cases is included in the notebook under **Section 5**.

---

## Reflections

The notebook concludes with a reflections section addressing:

* Best-performing features
* Best model
* Challenges encountered
* Time spent
* Screenshots of all plots and final results

---

## Documentation Link

This project is published on the hosted documentation site and can be accessed through:

**Projects → P4 Regression Model**

---

