# Banknote Authentication — Midterm Classification Project

- **Author:** Deb St. Cyr
- **Date:** November 2025

## Objective

This project applies supervised machine learning classification methods to identify whether a banknote is authentic or forged using statistical features calculated from digital image data.

## Quick Links
- **Notebook:** (clickable)  
  `notebooks/classification_dstcyr.ipynb`
- **Peer Review:** (clickable)  
  `peer_review.md`

## Results (Summary)
- **Best model:** {{e.g., Random Forest}}
- **Accuracy:** 99.4% | **Precision:** 98.7% | **Recall:** 100% | **F1-score:** 99.3%
- **Notes:** The Random Forest model achieved near-perfect performance, effectively capturing subtle nonlinear interactions between features. 
  ROC analysis confirmed strong class separation (AUC ≈ 1.00).

---

## Setup & Run
# Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS/Linux
source .venv/bin/activate

# Install dependencies and open the notebook
pip install -r requirements.txt
jupyter lab  # or open the notebook in VS Code

---

# Dataset Overview

**Source:** UCI Banknote Authentication Dataset

Each record represents a set of statistical measurements extracted from an image of a banknote.
The features describe patterns in pixel intensity that help distinguish genuine notes from counterfeit ones.

**Features:**

- variance — Variation in pixel intensity across the image

- skewness — Measure of asymmetry in intensity distribution

- curtosis — Measure of how sharp or flat the intensity distribution is

- entropy — Randomness or complexity in the image pattern

**Target:**

- class → 0 = authentic, 1 = forged

---

# Key Insights
Exploration of the dataset revealed a balanced distribution between authentic and forged banknotes, with no missing values or anomalies.
All four numerical features showed distinct ranges and strong separation between the two classes.
Notably, variance and skewness appeared to have the greatest influence on classification, while entropy showed more overlap between categories.
The clean, well-structured nature of this dataset made it an excellent candidate for testing multiple machine learning classifiers and comparing their performance.