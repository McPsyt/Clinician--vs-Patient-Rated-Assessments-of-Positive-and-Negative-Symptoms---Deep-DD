import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, linregress

# Load data
filepath = r'C:\Users\dylan\OneDrive - McGill University\McPsyt Lab Douglas\marianne_pouliot'
df = pd.read_csv(f"{filepath}/Book1.csv")

# Rename 'Subject' to 'ID' if needed
df.rename(columns={'Subject': 'ID'}, inplace=True)
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

    # Compute Spearman correlation
    corr, pval = spearmanr(subdf[x], subdf[y])

    # Color by ID
    color_vals = subdf['ID'].map(id_map)
    ax.scatter(subdf[x], subdf[y],
               c=[colors[i] for i in color_vals], s=30)

    # Regression line (optional, for visualization)
    slope, intercept, *_ = linregress(subdf[x], subdf[y])
    x_line = np.linspace(subdf[x].min(), subdf[x].max(), 100)
    y_line = slope * x_line + intercept
    ax.plot(x_line, y_line, color='black',
            label=f"Spearman r = {corr:.2f}, p = {pval:.3f}", linewidth=2)

    ax.set_title(f"{x} vs {y}")
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.legend()

# Final touches
fig.legend(handles=legend_elements, loc='lower right', title='Participant ID')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.suptitle("Spearman Correlations", fontsize=16)
plt.savefig(f'{filepath}/spearman_regressions.png')

