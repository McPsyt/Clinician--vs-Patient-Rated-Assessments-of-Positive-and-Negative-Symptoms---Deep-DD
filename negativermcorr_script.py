import pandas as pd
import numpy as np
import pingouin as pg
import matplotlib.pyplot as plt
from scipy.stats import linregress
import os

#load data
filepath = '/Users/jean-francoispouliot/Desktop/negativesymptoms.csv'
df = pd.read_csv(filepath)

# Rename 'Subject' to 'ID' if needed
df.rename(columns={'Subjects': 'ID'}, inplace=True)

# --- rm_corr calculations ---

# 1. PVSS-21 vs SANS
df_pvss_21_sans = df[['ID', 'PVSS_21', 'SANS']].dropna()
rm_pvss_21_sans = pg.rm_corr(
    data=df_pvss_21_sans,
    x='PVSS_21',
    y='SANS',
    subject='ID'
)
print("PVSS_21 vs SANS:\n", rm_pvss_21_sans)

# 2. PVSS-21 vs PANSS
df_pvss_21_panss = df[['ID', 'PVSS_21', 'PANSS']].dropna()
rm_pvss_21_panss = pg.rm_corr(
    data=df_pvss_21_panss,
    x='PVSS_21',
    y='PANSS',
    subject='ID'
)
print("PVSS_21 vs PANSS:\n", rm_pvss_21_panss)

# 3. SHAPS vs SANS
df_shaps_sans = df[['ID', 'SHAPS', 'SANS']].dropna()
rm_shaps_sans = pg.rm_corr(
    data=df_shaps_sans,
    x='SHAPS',
    y='SANS',
    subject='ID'
)
print("SHAPS vs SANS:\n", rm_shaps_sans)

# 4. SHAPS vs PANSS
df_shaps_panss = df[['ID', 'SHAPS', 'PANSS']].dropna()
rm_shaps_panss = pg.rm_corr(
    data=df_shaps_panss,
    x='SHAPS',
    y='PANSS',
    subject='ID'
)
print("SHAPS vs PANSS:\n", rm_shaps_panss)


# --- Plotting ---

df['ID'] = df['ID'].astype(str)

pairs = [
    ('PVSS_21', 'SANS'),
    ('PVSS_21', 'PANSS'),
    ('SHAPS', 'SANS'),
    ('SHAPS', 'PANSS'),
]

# Color mapping
ids = df['ID'].unique()
id_map = {id_: idx for idx, id_ in enumerate(ids)}
colors = plt.cm.tab10(np.linspace(0, 1, len(ids)))
legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=id_,
                              markerfacecolor=colors[idx], markersize=8)
                   for id_, idx in id_map.items()]

# Subplots
nrows, ncols = 2, 2
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 5 * nrows))
axes = axes.flatten()

# Plot loop
for idx, (x, y) in enumerate(pairs):
    ax = axes[idx]
    subdf = df[['ID', x, y]].dropna()
    rm = pg.rm_corr(data=subdf, subject='ID', x=x, y=y)
    corr, pval = rm['r'].values[0], rm['pval'].values[0]

    # Color by ID
    color_vals = subdf['ID'].map(id_map)
    ax.scatter(subdf[x], subdf[y],
               c=[colors[i] for i in color_vals], s=30)

    # Linear regression (for plotting line only)
    slope, intercept, *_ = linregress(subdf[x], subdf[y])
    x_line = np.linspace(subdf[x].min(), subdf[x].max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color='black',
            label=f'rm_corr r = {corr:.2f}, p = {pval:.3f}', linewidth=2)

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()

# Final touches
fig.legend(handles=legend_elements, loc='lower right', title='Participant ID')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.suptitle("Positive Symptoms Repeated Measures Correlation", fontsize=16)

save_dir = os.path.dirname(filepath)
plt.savefig(f'{save_dir}/regression.png')
print("saved negative regression.png to current directory")