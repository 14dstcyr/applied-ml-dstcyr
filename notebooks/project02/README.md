# Project 2: Titanic Dataset Exploration

This folder contains a ready-to-edit Jupyter Notebook for your Project 2 analysis.

## Files
- `ml02_stcyr.ipynb` — your notebook starter with numbered sections, reflections, and working code cells.
- `README.md` — this file with instructions and tips.
- `m102.py` - working script with just the code.

## Where to put this in your repo
Place this folder under your course repository at:
```
notebooks/
  project02/
    README.md
    m102.py
    ml02_stcyr.ipynb
```
> If your repo already has a `notebooks/project02` folder, you can drop these files into it.

## How to Run
1. Open your course repo in VS Code.
2. Ensure your Python environment (the same as Project 1) is activated.
3. Open `ml02_stcyr.ipynb` and run cells top-to-bottom.

## What to Edit
- Update the **title block** (name, date) in the first cell as needed.
- Add short **Reflection** answers after each section.
- Improve charts or add more (e.g., survival by `sex`, `class`, `embarked`, etc.).

## Notes
- The dataset is loaded from `seaborn.load_dataset("titanic")`, so no CSV is required.
- Correlations use `numeric_only=True` to avoid warnings.
- The stratified split is stratified by the **target (`survived`)**; compare class distributions to confirm.

## ✅ Final Verification & Commit Workflow

This sequence ensures that all dependencies are up to date, code is cleanly formatted, and all quality checks pass before committing to GitHub.

### 1. Update and sync dependencies (safe to run anytime)
uv sync --extra dev --extra docs --upgrade

### 2. Stage only Titanic project files
git add notebooks/project02 m102.py

### 3. Run Ruff for formatting and linting fixes
uvx ruff check --fix

### 4. Run all pre-commit quality checks
uv run pre-commit run --all-files

### 5. Stage any remaining auto-fixes (Ruff or pre-commit may reformat files)
git add .

### 6. Commit clean code
git commit -m "Clean Titanic project — all Ruff and pre-commit checks passed"

### 7. Push to GitHub
git push
