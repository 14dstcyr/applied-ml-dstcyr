"""ml02.py - Just the code.

This script provides a simple example of exploratory data analysis and modeling
using the Titanic dataset.

The dataset is loaded, cleaned, and visualized to explore relationships between
passenger characteristics and survival outcomes. Key features such as class, age,
sex, and fare are examined to identify trends and correlations.

A logistic regression model is then trained to predict passenger survival.
Several visualizations are included to illustrate data distributions, survival patterns,
and model insights.
"""

# Imports (single cell, import each lib only once)
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

# -----------------------------------------------------------------------------------------
# Display and theme settings (must come after imports, before main code)
# -----------------------------------------------------------------------------------------
pd.set_option("display.max_columns", 100)
sns.set_theme()

#########################################
# Section 1. Load and explore the dataset.
#########################################

# Load the Titanic Dataset
titanic = sns.load_dataset("titanic")

# Basic info
titanic.info()

# Display the first 10 rows
print(titanic.head(10))

# Missing values per column
print("Missing Values:")
print(titanic.isnull().sum())


# Summary statistics
print(titanic.describe())

# Correlations
print(titanic.corr(numeric_only=True))

# Compute correlation matrix (numeric only)
corr_matrix = titanic.corr(numeric_only=True)

# Unstack and sort correlations
corr_pairs = corr_matrix.unstack().sort_values(ascending=False)

# Remove self-correlations (feature correlated with itself)
corr_pairs = corr_pairs[corr_pairs < 1]

# Show the top 5 strongest correlations
corr_pairs.head(5)


#########################################
# Section 2. Visualize Feature Distributions
#########################################

# Scatter matrix for age, fare, and pclass
attributes = ["age", "fare", "pclass"]
axes = scatter_matrix(titanic[attributes], figsize=(12, 12), color="#8e2fc5a2", alpha=0.7)

# Loop through the diagonal subplots and recolor the histograms
n = len(attributes)
for i in range(n):
    ax = axes[i, i]
    for patch in ax.patches:
        patch.set_facecolor("#85f07ba6")
        patch.set_edgecolor("black")
        patch.set_alpha(0.7)

plt.show()


# Scatterplot: age vs fare
male = titanic[titanic["sex"] == "male"]
female = titanic[titanic["sex"] == "female"]

plt.scatter(male["age"], male["fare"], color="green", alpha=0.6, label="Male")
plt.scatter(female["age"], female["fare"], color="purple", alpha=0.6, label="Female")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.title("Age vs Fare by Gender")
plt.legend()
plt.show()

# Histogram of age
sns.set_palette("deep")

# Histogram + KDE on same scale (count)
plt.figure(figsize=(8, 6))
sns.histplot(
    data=titanic,
    x="age",
    kde=True,  # enables the smooth line
    bins=30,  # number of histogram bars
    color="purple",  # fill color
    edgecolor="black",  # thin black edges
    linewidth=0.9,
    alpha=0.8,  # slight transparency
)

plt.title("Age Distribution", fontsize=14)
plt.xlabel("Age", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.show()


# Count plot: class by survival
sns.countplot(x="class", hue="survived", data=titanic)
plt.title("Class Distribution by Survival")
plt.xlabel("Passenger Class")
plt.ylabel("Count")
plt.show()

# counts by class × survived
ct = pd.crosstab(titanic["class"], titanic["survived"])

# row-wise percentages
pct = ct.div(ct.sum(axis=1), axis=0).mul(100)

# (optional) if you only want Third class quickly:
third_pct = pct.loc["Third"]
print(third_pct)  # survived=0 and 1 percentages


# Handle Missing Values and Clean Data
# Impute missing values for age with median
titanic["age"] = titanic["age"].fillna(titanic["age"].median())

# Fill missing values for embark_town with mode
titanic["embark_town"] = titanic["embark_town"].fillna(titanic["embark_town"].mode()[0])


# Feature Engineering

# Family size = sibsp + parch + 1 (self)
titanic["family_size"] = titanic["sibsp"] + titanic["parch"] + 1

# Encode categoricals to numeric (new columns to preserve originals)
titanic["sex_num"] = titanic["sex"].map({"male": 0, "female": 1})
titanic["embarked_num"] = titanic["embarked"].map({"C": 0, "Q": 1, "S": 2})

# Binary 'alone' feature (already exists as bool -> convert to int copy)
titanic["alone_num"] = titanic["alone"].astype(int)

titanic[["family_size", "sex_num", "embarked_num", "alone_num"]].head()


#########################################
# Section 3. Feature Selection and Justification
#########################################

# We'll use **survived** as the classification target and select several intuitive predictors.
# Select features (X) and target (y)
feature_cols = ["age", "fare", "pclass", "sex_num", "family_size"]
X = titanic[feature_cols].copy()
y = titanic["survived"].copy()

X.head(), y.head()

#########################################
# Section 4. Train a Linear Regression Model
#########################################

# Compare a **basic train/test split** with a **stratified** split (stratified by the target `survived`).
# Basic train/test split
X_train_basic, X_test_basic, y_train_basic, y_test_basic = train_test_split(
    X, y, test_size=0.2, random_state=123
)

print("Basic Split — Train size:", len(X_train_basic), " Test size:", len(X_test_basic))


# Stratified split by the target 'survived'
splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=123)
for train_idx, test_idx in splitter.split(X, y):
    X_train_strat = X.iloc[train_idx]
    X_test_strat = X.iloc[test_idx]
    y_train_strat = y.iloc[train_idx]
    y_test_strat = y.iloc[test_idx]

print("Stratified Split — Train size:", len(X_train_strat), " Test size:", len(X_test_strat))


# Compare class distributions (original vs. each split)
print("Original Class Distribution (survived):")
print(y.value_counts(normalize=True).sort_index())

print("\nBasic Split — Class Distribution (survived):")
print(y_train_basic.value_counts(normalize=True).sort_index())
print(y_test_basic.value_counts(normalize=True).sort_index())

print("\nStratified Split — Class Distribution (survived):")
print(y_train_strat.value_counts(normalize=True).sort_index())
print(y_test_strat.value_counts(normalize=True).sort_index())
