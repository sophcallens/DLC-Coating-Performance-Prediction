""" Number of data for each input

This script computes and saves bars showing the number of available and unavailable
data points for many input columns (categorical and continuous).

"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, Normalize
import os

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
output_dir = 'project/data_analisis/figures/compo_dataset'

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

    input_column = info['column']
    use_log_bins = info.get('is_log', False)
    is_continuous = info.get('is_continuous', False)
    n_bins = 50
    unknown_label = 'unknown'
    min_bar_percentage_display = 3

    input_fname = (input_column[6:-1] if use_log_bins else input_column).replace('/', '_').replace(' ', '_')
    file_name = f'{input_fname}.png'

    if os.path.exists(f"{output_dir}/{input_fname}.png") :
        print(f'{input_column} already done')
        continue

    else :

        if is_continuous:

            # Font sizes for labels inside bars
            font_min = 5
            font_max = 50

            # Separate known / unknown
            df_known = df[df[input_column].notna()].copy()
            df_unknown = df[df[input_column].isna()].copy()

            values = df_known[input_column].values
            n = len(values)

            # Build bins
            if use_log_bins:
                log_min = np.floor(values.min())
                log_max = np.ceil(values.max())
                bins = np.arange(log_min, log_max + 1)
            else:
                q25, q75 = np.percentile(values, [25, 75])
                iqr = q75 - q25
                if iqr == 0:
                    n_bins = max(3, int(np.sqrt(n)))
                else:
                    bin_width = 2 * iqr / (n ** (1 / 3))
                    n_bins = int(np.ceil((values.max() - values.min()) / bin_width))
                n_bins = np.clip(n_bins, 3, 15)
                bins = np.linspace(values.min(), values.max(), n_bins + 1)

            df_known['Zone'] = pd.cut(df_known[input_column], bins=bins, include_lowest=True)

            # Counts per zone
            df_group = df_known['Zone'].value_counts(sort=False).reset_index()
            df_group.columns = ['Zone', 'N points']

            # Add unknown bucket
            df_group = pd.concat([
                df_group,
                pd.DataFrame({'Zone': [unknown_label], 'N points': [len(df_unknown)]}),
            ], ignore_index=True)

            total = df_group['N points'].sum()
            df_group['Percentage'] = 100 * df_group['N points'] / total

            df_group_known = df_group[df_group['Zone'] != unknown_label].copy()
            df_group_known['Zone_left'] = df_group_known['Zone'].apply(lambda x: x.left)
            df_group_known = df_group_known.sort_values(by='Zone_left')

            df_group_unknown = df_group[df_group['Zone'] == unknown_label]
            df_group_sorted = pd.concat([df_group_known, df_group_unknown], ignore_index=True)

            # Colors (light red -> light yellow)
            cmap = LinearSegmentedColormap.from_list('red_yellow', ['#ff9999', '#ffff99'])
            norm = Normalize(vmin=df_group_known['Percentage'].min(), vmax=df_group_known['Percentage'].max())

            colors = [
                'lightgray' if row['Zone'] == unknown_label else cmap(norm(row['Percentage']))
                for _, row in df_group_sorted.iterrows()
            ]

            # Plot
            fig, ax = plt.subplots(figsize=(12, 3.5))
            left = 0
            for (_, row), color in zip(df_group_sorted.iterrows(), colors):
                ax.barh(y=0, width=row['Percentage'], left=left, color=color, edgecolor='white', height=0.2)

                if row['Percentage'] > min_bar_percentage_display:
                    x_center = left + row['Percentage'] / 2
                    fontsize = font_min + (font_max - font_min) * (row['Percentage'] / 100)
                    fontsize = np.clip(fontsize, font_min, font_max)

                    if row['Zone'] == unknown_label:
                        label = unknown_label
                    else:
                        l = row['Zone'].left
                        r = row['Zone'].right
                        if use_log_bins:
                            label = f"$10^{{{int(l)}}}$ – $10^{{{int(r)}}}$"
                        else:
                            label = f"{l:.3g} – {r:.3g}"

                    ax.text(x_center, 0.01, label, ha='center', va='bottom', fontsize=fontsize, fontweight='bold')
                    ax.text(x_center, -0.01, f"{row['Percentage']:.1f}%", ha='center', va='top', fontsize=max(fontsize * 0.45, 8), fontweight='bold')
                    ax.text(x_center, -0.035, f"{row['N points']}", ha='center', va='top', fontsize=max(fontsize * 0.4, 7))

                left += row['Percentage']

            display_name = (input_column[6:-1] if use_log_bins else input_column).replace('/', '_').replace(' ', '_')
            ax.set_title(f"Presence of {display_name} by zones", pad=18, fontsize=16)
            ax.set_xlim(0, 100)
            ax.set_ylim(-0.2, 0.2)
            ax.axis('off')
            plt.tight_layout()

        else:
            # Categorical
            df[input_column] = df[input_column].fillna(unknown_label)
            df_group = df[input_column].value_counts().reset_index()
            df_group.columns = [input_column, 'N points']
            total = df_group['N points'].sum()
            df_group['Percentage'] = 100 * df_group['N points'] / total

            known = df_group[df_group[input_column] != unknown_label].sort_values('Percentage', ascending=False)
            unknown = df_group[df_group[input_column] == unknown_label]
            df_group_sorted = pd.concat([known, unknown], ignore_index=True)

            cmap = LinearSegmentedColormap.from_list('red_yellow', ['#ff9999', '#ffff99'])
            norm = Normalize(vmin=df_group_sorted[df_group_sorted[input_column] != unknown_label]['Percentage'].min(), vmax=df_group_sorted[df_group_sorted[input_column] != unknown_label]['Percentage'].max())

            colors = [
                'lightgray' if row[input_column] == unknown_label else cmap(norm(row['Percentage']))
                for _, row in df_group_sorted.iterrows()
            ]

            fig, ax = plt.subplots(figsize=(12, 3.5))
            left = 0
            for _, row in df_group_sorted.iterrows():
                ax.barh(y=0, width=row['Percentage'], left=left, color=colors.pop(0), edgecolor='white', height=0.2)

                if row['Percentage'] > min_bar_percentage_display:
                    x_center = left + row['Percentage'] / 2
                    ax.text(x_center, 0.01, str(row[input_column]), ha='center', va='bottom', fontsize=18, fontweight='bold')
                    ax.text(x_center, -0.01, f"{row['Percentage']:.1f}%", ha='center', va='top', fontsize=10, fontweight='bold')
                    ax.text(x_center, -0.035, f"{row['N points']}", ha='center', va='top', fontsize=9)

                left += row['Percentage']

            ax.set_title(f"Distribution of categories for '{input_column[6:-1] if use_log_bins else input_column}' in the dataset", pad=18, fontsize=16)
            ax.set_xlim(0, 100)
            ax.set_ylim(-0.2, 0.2)
            ax.axis('off')
            plt.tight_layout()

        # Save
        plt.savefig(f'{output_dir}/{file_name}', dpi=300, bbox_inches='tight')
        plt.close()
        print(f'{input_column} done')


print('TASK FINISHED')