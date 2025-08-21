import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, linregress
import os

# Load data
filepath ='/Users/jean-francoispouliot/Desktop/negativesymptoms.csv'
df = pd.read_csv(filepath)

# Rename 'Subject' to 'ID' if needed
df.rename(columns={'Subjects': 'ID'}, inplace=True)
df['ID'] = df['ID'].astype(str)

# Data analysis 

# Assuming your data is in a DataFrame called df
# and columns are named as shown below:
# 'PVSS_21', 'SANS', 'PANSS', 'SHAPS'



# Define variable pairs
pairs = [
    ('PVSS_21', 'SANS'),
    ('PVSS_21', 'PANSS'),
    ('SHAPS', 'SANS'),
    ('SHAPS', 'PANSS')
]

# Compute and print Spearman's correlation for each pair
for var1, var2 in pairs:
    # Drop rows with missing data for the pair
    data = df[[var1, var2]].dropna()
    rho, pval = spearmanr(data[var1], data[var2])
    print(f"Spearman correlation between {var1} and {var2}:")
    print(f"  rho = {rho:.3f}, p-value = {pval:.4f}")
    print()

# Plotting Data ----- 


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

save_dir = os.path.dirname(filepath)
plt.savefig(os.path.join(save_dir, 'spearman_regressions.png'))
print("Saved 'spearman_regressions.png' to current directory.")

