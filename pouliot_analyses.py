# Hi Marianne
# This script has 2 parts
# 1. Perform rm corr
# 2. Plot the results



import pandas as pd
import numpy as np
import pingouin as pg
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load data
filepath = r'C:\Users\dylan\OneDrive - McGill University\McPsyt Lab Douglas\marianne_pouliot'
df = pd.read_csv(f"{filepath}/Book1.csv")


# Rename 'Subject' to 'ID' if needed
df.rename(columns={'Subject': 'ID'}, inplace=True)

# 1. LSHS vs SAPS
df_lshs_saps = df[['ID', 'LSHS', 'SAPS']].dropna()
rm_lshs_saps = pg.rm_corr(
            data=df,
            x='LSHS',
            y='SAPS',
            subject='ID'
        )
print("LSHS vs SAPS:\n", rm_lshs_saps)

# 2. LSHS vs PANSS
df_lshs_panss = df[['ID', 'LSHS', 'PANSS']].dropna()
rm_lshs_panss = pg.rm_corr(data=df_lshs_panss, subject='ID', x='LSHS', y='PANSS')
print("LSHS vs PANSS:\n", rm_lshs_panss)

# 3. AHRS vs SAPS
df_ahrs_saps = df[['ID', 'AHRS', 'SAPS']].dropna()
rm_ahrs_saps = pg.rm_corr(data=df_ahrs_saps, subject='ID', x='AHRS', y='SAPS')
print("AHRS vs SAPS:\n", rm_ahrs_saps)

# 4. AHRS vs PANSS
df_ahrs_panss = df[['ID', 'AHRS', 'PANSS']].dropna()
rm_ahrs_panss = pg.rm_corr(data=df_ahrs_panss, subject='ID', x='AHRS', y='PANSS')
print("AHRS vs PANSS:\n", rm_ahrs_panss)







########################################### 2. Now we will plot

df['ID'] = df['ID'].astype(str)

# Define variable pairs
pairs = [
    ('LSHS', 'SAPS'),
    ('LSHS', 'PANSS'),
    ('AHRS', 'SAPS'),
    ('AHRS', 'PANSS')
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
plt.suptitle("Repeated Measures Correlations", fontsize=16)
plt.savefig(f'{filepath}/regressions.png')
print("saved negative regression.png to current directory")