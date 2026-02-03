# DLC Coating Performance Prediction

**Short description**

This repository contains data processing, analysis, and machine learning models for *predicting the performance of Diamond-Like Carbon (DLC) coatings*, notably **friction (CoF)** and **wear**, based on experimental input parameters.

---

## Project goals

- Explore and visualize relationships between inputs and coating performance
- Build a reproducible data pipeline to process raw experimental measurements
- Train and compare machine learning models (Random Forest, XGBoost, etc.) to predict CoF and wear
- Provide reproducible figures and metrics for comparative evaluation

---

## Repository structure

- `data/` — Raw and processed data plus scripts for selection and reduction
  - `raw.csv` — Original raw dataset
  - `processed/processing.py` — Data cleaning & preprocessing
  - `selected/selection.py` — Selection and scenario datasets

- `data_analisis/` — Scripts and output for plotting and exploratory figures
  - `figures/` — Generated figures
  - `make_figures/` — Scripts to generate plots (composition, ternary diagrams, feature presence, etc.)
    - `compo_dataset.py` - generates a bar with the data repartition (also with abscence) of each feature
    - `Data_quantity.py`- generates a graph of number of point in function the value for each feature
    - `Fric_wear_input.py` - generates two graphs : CoF in function of input and Wear in fonction of input for some each feature
    - `input1_input2_compo.py` - mean value of input1 and input2 in function of each element
    - `input1_input2_density.py` - input1 in function of input2 with data quantity in color
    - `input1_input2_family.py`- input1 in function of input2 with DLC group in color
    - `Ternary_diagram.py` - some ternary diagrams
    - `presence_per_feature.py` - data quantity for each feature

- `pred_comparison/` — Code to train and compare multiple models
  - `main.py` — Orchestrates model training and evaluation
  - `results/` — Contains `figures/` and `metrics/` per scenario
  - `src/` — Utilities, preprocessing, models and helpers

- `pred_extra_trees/` — Extra Trees experiments (best-performing models; primary focus)  
  - `main.py` — run Extra Trees experiments and evaluation
  - `results/CoF/`, — CoF prediction results
  - `results/Wear/` — Wear prediction results

- `pred_rd_forest/` — Random Forest experiments (alternative ensemble)
  - `main.py` — run model optimisation (with optuna) and evaluation
  - `results/CoF/` — CoF prediction results (per scenario)
  - `results/Wear/` — Wear prediction results (per scenario)

- `pred_xgboost/` — XGBoost experiments (explored but later deprioritized because Random Forest produced better results)
  - `main.py` — run model optimisation (with optuna) and evaluation
  - `results_RMSE/CoF/` — CoF prediction results
  - `results_RMSE/Wear/` — Wear prediction results
  
  **Note:** **_Extra Trees (`pred_rd_forest/`) yielded the best performance in our evaluations; XGBoost experiments were kept for comparison but are no longer the focus._**

- `presentation.pdf` - Entire presentation of the project.


---

## Quick start

> **Note:** The repository assumes a Python 3.8+ environment. Adjust commands to your chosen environment manager (venv/conda).

1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
```

2. Install required packages

```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn jupyter
```

3. Process data

```bash
python data/01-processed/processing.py
python data/02-selected/selection.py
python data/03-reduced/reduction.py
```

4. Generate figures for data analisis of raw.csv

```bash
python data_analisis/make_figures/Data_quantity.py
# or any other scripts in data_analisis/make_figures/
```

5. Train & compare models

```bash
python pred_comparison/main.py
# or run model-specific scripts in `pred_rd_forest/`, `pred_extra_trees/`, or `pred_xgboost/`
```

6. Inspect results

- `pred_comparison/results/metrics/` — CSV files for scenario metrics
- `pred_comparison/results/figures/` — Plots comparing predictions and ground truth


---

## Conclusion

- Extra Trees has proven to be the most effecient prediction method. 
- Scenraio 2 (inputs = Sliding velocity - Humidity - Ball hardness - Load - Temperature - Sp2/Sp3 - DLC groupe - Film hardness - Doped - H) has the best results 

