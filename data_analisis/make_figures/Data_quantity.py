"""Data quantity plots

This script computes and saves plots showing the number of available
data points for many input columns (categorical and continuous).

"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from scipy import stats


info_input = [
    # Each entry: column name, is_continuous (bool), is_log_scaled (bool)
    {'column': 'DLC groupe', 'is_continuous': False, 'is_log': False},
    {'column': 'C_content', 'is_continuous': True, 'is_log': False},
    {'column': 'H_content', 'is_continuous': True, 'is_log': False},
    {'column': 'Doped', 'is_continuous': False, 'is_log': False},
    {'column': 'Sp2/Sp3', 'is_continuous': True, 'is_log': False},
    {'column': 'Film elastic modulus (GPa)', 'is_continuous': True, 'is_log': False},
    {'column': 'Film hardness (GPa)', 'is_continuous': True, 'is_log': False},
    {'column': 'Ball elastic modulus (GPa)', 'is_continuous': True, 'is_log': False},
    {'column': 'Ball hardness (GPa)', 'is_continuous': True, 'is_log': False},
    {'column': 'Load (N)', 'is_continuous': True, 'is_log': False},
    {'column': 'Temperature', 'is_continuous': True, 'is_log': False},
    {'column': 'Humidity', 'is_continuous': False, 'is_log': False},
    {'column': 'E*', 'is_continuous': True, 'is_log': False},
    {'column': 'Hertz pressure (Gpa)', 'is_continuous': True, 'is_log': False},
    {'column': 'log10(Sliding distance (m))', 'is_continuous': True, 'is_log': True},
    {'column': 'log10(Wear Volume V)', 'is_continuous': True, 'is_log': True},
    {'column': 'log10(Sliding velocity (m/s))', 'is_continuous': True, 'is_log': True},
    {'column': 'log10(Rq (nm))', 'is_continuous': True, 'is_log': True},
    {'column': 'Friction coefficient', 'is_continuous': True, 'is_log': False},
    {'column': 'log10(Wear rate)', 'is_continuous': True, 'is_log': True}

]

input_dir = 'project/data/processed'
output_dir = 'project/data_analisis/figures/Data_quantity'

# =========================
# Chargement des données
# =========================
df = pd.read_csv(
    f"{input_dir}/processed.csv",
    sep=";",
    decimal=",",
    encoding="utf-8"
)

for info in info_input:

    col_name = info['column']
    is_log = info.get('is_log', False)
    is_continuous = info.get('is_continuous', False)
    n_bins = 50

    input_fname = (col_name[6:-1] if is_log else col_name).replace('/', '_').replace(' ', '_')
    file_name = f'{input_fname}.png'

    if os.path.exists(f"{output_dir}/{input_fname}.png") :
        print(f'{col_name} already done')
        continue
    else :

        if is_continuous:
            # take the non-null series (preserve original df untouched)
            data = df[col_name].dropna()

            # handle empty series
            if data.empty:
                print(f"{col_name}: no data, skipped")
                continue

            # BINNING: create bin edges and label each data point
            val_min, val_max = data.min(), data.max()
            bin_edges = np.linspace(val_min, val_max, n_bins + 1)
            bins = pd.cut(data, bin_edges, labels=False, include_lowest=True)

            # Build df_cut (one row per bin) and compute counts using the 'bins' series
            df_cut = pd.DataFrame({'bin': range(n_bins)})
            counts = bins.value_counts(sort=False)
            df_cut['n_points'] = df_cut['bin'].map(counts).fillna(0).astype(int)
            df_cut['x_left'] = bin_edges[:-1]
            df_cut['x_right'] = bin_edges[1:]
            df_cut['x_center'] = (df_cut['x_left'] + df_cut['x_right']) / 2

            # For plotting, convert back from log-space to real scale when needed
            if is_log:
                df_cut['x_plot_left'] = 10 ** df_cut['x_left']
                df_cut['width'] = 10 ** df_cut['x_right'] - 10 ** df_cut['x_left']
            else:
                df_cut['x_plot_left'] = df_cut['x_left']
                df_cut['width'] = df_cut['x_right'] - df_cut['x_left']

            # Candidate distributions to test (fit to the data)
            distributions = {
                "normal": stats.norm,
                "lognormal": stats.lognorm,
                "exponential": stats.expon,
                "gamma": stats.gamma,
                "weibull": stats.weibull_min
            }

            def aic(log_likelihood, k):
                """Compute Akaike Information Criterion.

                k is the number of parameters in the model.
                """
                return 2 * k - 2 * log_likelihood

            results = []

            for name, dist in distributions.items():
                try:
                    params = dist.fit(data)
                    log_likelihood = np.sum(dist.logpdf(data, *params))
                    k = len(params)
                    results.append((name, params, aic(log_likelihood, k)))
                except Exception:
                    # skip distributions that fail to fit
                    continue

            if results:
                best_dist = min(results, key=lambda x: x[2])
                best_name, best_params, _ = best_dist
                best_distribution = distributions[best_name]
            else:
                best_name, best_params, best_distribution = (None, None, None)

            x = np.linspace(min(data), max(data), 1000)
            if best_distribution is not None:
                plt.plot(x, best_distribution.pdf(x, *best_params),
                        label=f"Best fit: {best_name}", linewidth=2)

                # Average bin width (in data units)
                bin_width = (val_max - val_min) / n_bins

                # Scale PDF to histogram counts
                pdf = best_distribution.pdf(x, *best_params)
                pdf_scaled = pdf * len(data) * bin_width
            else:
                pdf_scaled = np.zeros(1000)

            # For plotting on the real scale
            x_fit_real = 10 ** x if is_log else x

            # =========================
            # Plot
            # =========================
            # Normalize colors across bins; guard against constant counts
            vmin = int(df_cut['n_points'].min())
            vmax = int(df_cut['n_points'].max())
            if vmin == vmax:
                vmax = vmin + 1
            norm_colors = Normalize(vmin=vmin, vmax=vmax)
            colors = plt.cm.viridis(norm_colors(df_cut['n_points']))

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(
                df_cut['x_plot_left'],
                df_cut['n_points'],
                width=df_cut['width'],
                color=colors,
                align='edge'
            )

            ax.plot(x_fit_real, pdf_scaled, color='black', linewidth=2, label=f"Best fit: {best_name}")

            ax.set_xlabel(col_name[6:-1] if is_log else col_name)
            ax.set_ylabel("Number of points")
            ax.set_title("Available data count")
            if is_log:
                ax.set_xscale('log')

            ax.legend()
            sm = ScalarMappable(norm=norm_colors, cmap=plt.cm.viridis)
            # set_array should receive the data shown by the colormap
            sm.set_array(df_cut['n_points'].values)
            fig.colorbar(sm, ax=ax, label="Number of points")

            plt.tight_layout()

        else:
            data = df[col_name].dropna()

            if data.empty:
                print(f"{col_name}: no data, skipped")
                continue

            # Count categories
            counts = data.value_counts().sort_index()
            df_cut = counts.reset_index()
            df_cut.columns = [col_name, 'n_points']

            # Color normalization
            vmin = int(df_cut['n_points'].min())
            vmax = int(df_cut['n_points'].max())
            if vmin == vmax:
                vmax = vmin + 1

            norm_colors = Normalize(vmin=vmin, vmax=vmax)
            colors = plt.cm.viridis(norm_colors(df_cut['n_points']))

            # Plot
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(
                df_cut[col_name].astype(str),
                df_cut['n_points'],
                color=colors
            )

            ax.set_xlabel(col_name[6:-1] if is_log else col_name)
            ax.set_ylabel("Number of points")
            ax.set_title("Available data count")

            ax.tick_params(axis='x', rotation=45)

            sm = ScalarMappable(norm=norm_colors, cmap=plt.cm.viridis)
            sm.set_array(df_cut['n_points'].values)
            fig.colorbar(sm, ax=ax, label="Number of points")

            plt.tight_layout()
        
        # Save figure
        plt.savefig(f'{output_dir}/{file_name}', dpi=300, bbox_inches='tight')
        plt.close()
        print(f'{col_name} done')


print('TASK FINISHED')
