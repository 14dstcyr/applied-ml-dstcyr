# Lab 2 Howell Project Overview

**Author:** Deb St. Cyr  
**Course:** Applied Machine Learning – Module 2  
**Institution:** Northwest Missouri State University  
**Date:** October 2025  

This lab explores the Howell dataset, a classic anthropometric dataset widely used in data analysis and statistical modeling.
The goal of the lab is to:

* Explore and visualize relationships between variables

* Clean and prepare the dataset for modeling

* Engineer new features (BMI and BMI categories)

* Split the dataset for training/testing using both random and stratified techniques
  
  ---------

### Folder Structure
```text
lab2_howell/
│
├── Howell.csv              # Dataset file (semicolon-separated)
├── howell_lab2.ipynb       # Jupyter notebook for Lab 2
└── README.md               # This file



----------

# Environment Setup
**Dependencies** (install via `pip` or within your `.venv` environment):

```
pip install pandas numpy matplotlib scikit-learn
```

**Recommended Environment:**

  * Python 3.10+

  * JupyterLab or VS Code with Jupyter extension

-----------

# Notebook Sections
**Section 1: Import and Inspect the Data**

- Load the Howell dataset using `pandas.read_csv()` with `sep=";"`.

- Display data info, summary statistics, and correlations.

- Identify feature names, missing values, data types, and correlations.

- Reflection questions guide your interpretation of dataset structure and units (height in cm, weight in kg, age in years).

------------

# Section 2: Data Exploration and Preparation
**2.1 Scatter Matrix and Initial Visualizations**
- Visualize relationships between height, weight, and age.

- Describe distributions (skewness, bimodality, and normality).

- Create scatter plots for height–weight and age–height relationships, with gender coloring.

**2.2 Data Cleaning**

- Demonstrate how to add and remove columns with `DataFrame.drop()`.

- Compute medians and means.

- Fill missing values using `.fillna()` (dataset had none, but the process was shown).

**2.3 Feature Engineering**

- Create a **BMI** feature using the metric formula `10000 * weight / height²`.

- Add a **BMI** category feature with four classes (Underweight, Normal, Overweight, Obese).

- Visualize **Age vs BMI**, colored by gender, to explore growth and body composition trends.

**2.4 Plot with Masking**

- Use **NumPy masked arrays** to selectively plot male and female adults on the same graph.

- Demonstrate how to compare subsets visually while preserving all data.

- Show gender-specific height/weight clustering.

------------

# Section 3: Split the Data for Training and Testing

**3.1 Adult/Child Split**

- Separate dataset into adults (`age > 18`) and children (`age ≤ 18`).

**3.2 Basic Train/Test Split**

- Use `train_test_split()` (80/20) to divide adult data.

- Compare gender ratios between training and test sets.

**3.3 Stratified Train/Test Split**

- Use `StratifiedShuffleSplit()` to preserve gender balance in both training and test sets.

- Compare male/female ratios across original, training, and test subsets.

- Explain why stratification is important for model fairness and evaluation.

------------

# How to Run
1. Launch JupyterLab or VS Code.

2. Open howell_lab2.ipynb inside the lab2_howell/ folder.

3. Run all cells in order (Kernel → Restart & Run All).

4. Verify all plots render successfully, and reflection answers appear below each section.

----------

# References

- Howell, N. (1976). Demography of the Dobe !Kung. Academic Press.

- McElreath, R. (2020). Statistical Rethinking (2nd ed.). CRC Press.

- Scikit-learn Documentation – Train/Test Split
