TR-Data-Soph — plotting & data utilities
=================================================

Summary
-------
This folder contains data-cleaning and plotting scripts used to explore a tribology dataset (wear, friction, DLC types and properties). Scripts live in the top-level `DLC-Coating-Performance-Prediction/` folder and a set of plotting helpers are under `DLC-Coating-Performance-Prediction/make_figures/`.

Goals
-----
- Provide reproducible plotting scripts that read CSVs from `DLC-Coating-Performance-Prediction/data/` and save figures into `DLC-Coating-Performance-Prediction/make_figures/figures/`.
- Keep the CSV schema stable: do not rename CSV headers without a coordinated change across scripts.
- Make scripts consistent: English docstrings, snake_case variables, consistent colormaps (viridis) and robust handling of missing data.

Quick start
-----------
1. Create a Python virtual environment and install minimal dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install pandas numpy matplotlib scipy seaborn ternary
```

2. Run a single plotting script from the `DLC-Coating-Performance-Prediction/` working directory. For example, to create the friction/wear per-input figures:

```bash
python3 make_figures/Fric_wear_input.py
```

3. Output images will be saved under `make_figures/figures/` in subfolders. Check console output for generated filenames.

Data format
-----------
- Input CSVs are in `DLC-Coating-Performance-Prediction/data/` (many scripts expect `data/cleaned_dataset.csv`).
- Reader options used: `sep=';'`, `decimal=','`, `encoding='utf-8'` — keep that format when updating raw CSV files.

What I standardized
-------------------
- Module docstrings in English and consistent naming (snake_case: `col_name`, `is_continuous`, `is_log`).
- Perceptually-uniform colormaps (`viridis`) for continuous color encodings.
- Correct use of `matplotlib.cm.ScalarMappable` for colorbars (set_array with the data array).
- Safer handling of NaNs and empty bins when computing statistics and plotting.

Files of interest
-----------------
- `data/cleaned_dataset.csv` — primary dataset used by plotting scripts.
- `make_figures/` — plotting scripts and their output directory `make_figures/figures/`.

