"""
Diabetes Prediction Project
File: 05_multivariate_analysis.py
Purpose: Create multivariate visualizations (Visualizations 19-23)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# ============================
# CONFIGURATION
# ============================
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Define output directories
output_dir = '../outputs/figures/multivariate'
os.makedirs(output_dir, exist_ok=True)

# ============================
# LOAD DATA
# ============================
df = pd.read_csv('../diabetes_data_upload.csv')

# Create numeric version for calculations
df_numeric = df.copy()
df_numeric['Gender'] = df_numeric['Gender'].map({'Male': 1, 'Female': 0})
df_numeric['class'] = df_numeric['class'].map({'Positive': 1, 'Negative': 0})

symptom_cols = ['Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 
                'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 
                'Irritability', 'delayed healing', 'partial paresis', 
                'muscle stiffness', 'Alopecia', 'Obesity']

for col in symptom_cols:
    df_numeric[col] = df_numeric[col].map({'Yes': 1, 'No': 0})

# Add symptom count to both df and df_numeric
df['symptom_count'] = df[symptom_cols].apply(lambda x: (x == 'Yes').sum(), axis=1)
df_numeric['symptom_count'] = df_numeric[symptom_cols].sum(axis=1)

# ============================
# CALCULATE TOP FEATURES
# ============================
target_corr = df_numeric[symptom_cols + ['class']].corr()['class'].drop('class').sort_values()
top_5_features = target_corr.nlargest(5).index.tolist()

# ============================
# VISUALIZATION 19: Pairplot of Top 5 Features
# ============================
print("Creating Visualization 19: Pairplot of Top 5 Features...")
pairplot_fig = sns.pairplot(df_numeric[top_5_features + ['class']], 
                            hue='class', diag_kind='kde',
                            palette={1: 'lightcoral', 0: 'lightblue'},
                            plot_kws={'alpha':0.6, 's':50, 'edgecolor':'black'},
                            diag_kws={'alpha':0.6})
pairplot_fig.fig.suptitle('19. Pairplot of Top 5 Most Predictive Features', fontsize=16, fontweight='bold', y=1.02)
plt.setp(pairplot_fig._legend.get_texts(), fontsize='12')
plt.setp(pairplot_fig._legend.get_title(), fontsize='12', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'pairplot_top_features.png'), dpi=300, bbox_inches='tight')
plt.show()

# ============================
# VISUALIZATION 20: Age vs Symptom Count
# ============================
plt.figure(figsize=(14,8))
for status, color, marker in zip(['Positive','Negative'], ['red','blue'], ['o','s']):
    subset = df[df['class']==status]
    plt.scatter(subset['Age'], subset['symptom_count'], 
                c=color, label=status, alpha=0.6, s=100, marker=marker, edgecolor='black')

# Regression lines
for status in ['Positive','Negative']:
    subset = df[df['class']==status]
    z = np.polyfit(subset['Age'], subset['symptom_count'], 1)
    p = np.poly1d(z)
    plt.plot(sorted(subset['Age']), p(sorted(subset['Age'])), 
             linewidth=3, linestyle='--', color='red' if status=='Positive' else 'blue', alpha=0.8)

plt.xlabel('Age', fontsize=14)
plt.ylabel('Number of Symptoms', fontsize=14)
plt.title('20. Age vs Number of Symptoms (Colored by Diabetes Status)', fontsize=16, fontweight='bold')
plt.legend(fontsize=12, loc='upper right')
plt.grid(True, alpha=0.3)

# Correlation text box
correlation = df_numeric.groupby('class')[['Age','symptom_count']].corr().iloc[0::2,-1]
textstr = f'Correlation (Age vs Symptoms):\nPositive: {correlation[1]:.3f}\nNegative: {correlation[0]:.3f}'
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(output_dir,'age_vs_symptom_count.png'), dpi=300, bbox_inches='tight')
plt.show()

# ============================
# VISUALIZATION 21: 3D Scatter Plots
# ============================
print("Creating Visualization 21: 3D Scatter Plots...")
fig = plt.figure(figsize=(16,8))
top_3 = target_corr.nlargest(3).index.tolist()
ax1 = fig.add_subplot(121, projection='3d')

colors_map = {'Positive':'red','Negative':'blue'}
for status,color in colors_map.items():
    subset = df_numeric[df_numeric['class']==(1 if status=='Positive' else 0)]
    ax1.scatter(subset[top_3[0]], subset[top_3[1]], subset[top_3[2]], c=color, label=status, alpha=0.6, s=50)

ax1.set_xlabel(top_3[0])
ax1.set_ylabel(top_3[1])
ax1.set_zlabel(top_3[2])
ax1.set_title('21a. 3D Scatter Plot Top 3 Features', fontsize=14, fontweight='bold')
ax1.legend()

# Second 3D plot
ax2 = fig.add_subplot(122, projection='3d')
top_3b = target_corr.nlargest(4).index.tolist()[1:4]
for status,color in colors_map.items():
    subset = df_numeric[df_numeric['class']==(1 if status=='Positive' else 0)]
    ax2.scatter(subset[top_3b[0]], subset[top_3b[1]], subset[top_3b[2]], c=color, label=status, alpha=0.6, s=50)

ax2.set_xlabel(top_3b[0])
ax2.set_ylabel(top_3b[1])
ax2.set_zlabel(top_3b[2])
ax2.set_title('21b. 3D Scatter Plot Next 3 Features', fontsize=14, fontweight='bold')
ax2.legend()

plt.suptitle('21. 3D Scatter Plots of Top Predictive Features', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(output_dir,'3d_scatter_plots.png'), dpi=300, bbox_inches='tight')
plt.show()

# ============================
# VISUALIZATION 22: Symptom Co-occurrence
# ============================
print("Creating Visualization 22: Symptom Co-occurrence Analysis...")
fig, axes = plt.subplots(1,2,figsize=(18,8))

# Heatmap of correlation
ax1 = axes[0]
symptom_corr = df_numeric[symptom_cols].corr()
mask = np.triu(np.ones_like(symptom_corr, dtype=bool))
sns.heatmap(symptom_corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, linewidths=0.5, ax=ax1, cbar_kws={"shrink":0.8}, annot_kws={"size":8})
ax1.set_title('22a. Symptom Co-occurrence Correlation Matrix', fontsize=14, fontweight='bold')
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')

# Heatmap of co-occurrence percentages
ax2 = axes[1]
cooccurrence = np.zeros((len(symptom_cols), len(symptom_cols)))
for i,sym1 in enumerate(symptom_cols):
    for j,sym2 in enumerate(symptom_cols):
        if i<=j:
            both = ((df[sym1]=='Yes') & (df[sym2]=='Yes')).sum()
            cooccurrence[i,j]=both
            cooccurrence[j,i]=both

cooccurrence_df = pd.DataFrame((cooccurrence/len(df))*100, index=symptom_cols, columns=symptom_cols)
sns.heatmap(cooccurrence_df, annot=True, fmt='.1f', cmap='YlOrRd', square=True, linewidths=0.5, ax=ax2, cbar_kws={"shrink":0.8, "label":"Co-occurrence (%)"}, annot_kws={"size":8})
ax2.set_title('22b. Symptom Co-occurrence Percentages', fontsize=14, fontweight='bold')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45, ha='right')
ax2.set_yticklabels(ax2.get_yticklabels(), rotation=0)

plt.suptitle('22. Symptom Co-occurrence Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(output_dir,'symptom_cooccurrence.png'), dpi=300, bbox_inches='tight')
plt.show()

# ============================
# VISUALIZATION 23: PCA Analysis
# ============================
print("Creating Visualization 23: PCA Analysis...")
fig, axes = plt.subplots(1,2,figsize=(16,6))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric[symptom_cols])
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# PCA scatter plot
ax = axes[0]
for status,color in zip(['Positive','Negative'], ['red','blue']):
    mask = df_numeric['class']==(1 if status=='Positive' else 0)
    ax.scatter(X_pca[mask,0], X_pca[mask,1], c=color, label=status, alpha=0.6, s=50, edgecolor='black')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('23a. PCA Projection (First 2 Components)', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# Explained variance ratio
ax = axes[1]
explained_variance = pca.explained_variance_ratio_*100
cumulative_variance = np.cumsum(explained_variance)
x_pos = range(1,len(explained_variance)+1)
ax.bar(x_pos, explained_variance, alpha=0.7, label='Individual', color='skyblue', edgecolor='black')
ax.plot(x_pos, cumulative_variance, 'ro-', linewidth=2, label='Cumulative', markersize=8)
ax.set_xlabel('Principal Component')
ax.set_ylabel('Variance Explained (%)')
ax.set_title('23b. PCA Explained Variance Ratio', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xticks(x_pos)

plt.suptitle('23. Principal Component Analysis (PCA) Results', fontsize=16, fontweight='bold', y=1.05)
plt.tight_layout()
plt.savefig(os.path.join(output_dir,'pca_analysis.png'), dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "="*60)
print("MULTIVARIATE ANALYSIS COMPLETE - All visualizations saved in", output_dir)
print("="*60)