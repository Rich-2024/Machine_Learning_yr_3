"""
Diabetes Prediction Project
File: 04_correlation_analysis.py
Purpose: Create correlation visualizations (Visualizations 15–22)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

# =====================================================
# CONFIGURATION
# =====================================================
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Base directories
BASE_OUTPUT = '../data_exploration/outputs'
CORR_OUTPUT = os.path.join(BASE_OUTPUT, 'correlation')

# Create directories if they don't exist
os.makedirs(BASE_OUTPUT, exist_ok=True)
os.makedirs(CORR_OUTPUT, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================
df = pd.read_csv('../diabetes_data_upload.csv')

# Convert categorical columns to numeric
df_numeric = df.copy()
df_numeric['Gender'] = df_numeric['Gender'].map({'Male': 1, 'Female': 0})
df_numeric['class'] = df_numeric['class'].map({'Positive': 1, 'Negative': 0})

symptom_cols = [
    'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness',
    'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching',
    'Irritability', 'delayed healing', 'partial paresis',
    'muscle stiffness', 'Alopecia', 'Obesity'
]

for col in symptom_cols:
    df_numeric[col] = df_numeric[col].map({'Yes': 1, 'No': 0})

# =====================================================
# HELPER FUNCTION TO SAVE FIGURES
# =====================================================
def save_fig(fig, filename):
    fig_path = os.path.join(CORR_OUTPUT, filename)
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {fig_path}")
    plt.close(fig)

# =====================================================
# VISUALIZATION 15: Complete Correlation Heatmap
# =====================================================
fig = plt.figure(figsize=(18, 14))
corr_matrix = df_numeric.corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', 
            center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8},
            annot_kws={"size": 8})
plt.title('15. Complete Correlation Heatmap of All Features', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
save_fig(fig, 'complete_correlation_heatmap.png')

# =====================================================
# VISUALIZATION 16: Correlation with Target (Horizontal Bar)
# =====================================================
fig = plt.figure(figsize=(14, 10))
target_corr = corr_matrix['class'].drop('class').sort_values()
colors = ['red' if x < 0 else 'green' for x in target_corr.values]
bars = plt.barh(range(len(target_corr)), target_corr.values, color=colors, alpha=0.7)
plt.yticks(range(len(target_corr)), target_corr.index, fontsize=10)
plt.xlabel('Correlation with Diabetes', fontsize=12)
plt.title('16. Feature Correlations with Diabetes Status', fontsize=16, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.3, axis='x')
# Add labels
for bar, val in zip(bars, target_corr.values):
    plt.text(val + (0.01 if val >= 0 else -0.05), bar.get_y() + bar.get_height()/2, f'{val:.3f}', va='center', fontweight='bold')
# Legend
legend_elements = [Patch(facecolor='green', alpha=0.7, label='Positive Correlation'),
                   Patch(facecolor='red', alpha=0.7, label='Negative Correlation')]
plt.legend(handles=legend_elements, loc='lower right')
plt.tight_layout()
save_fig(fig, 'target_correlations.png')

# =====================================================
# VISUALIZATION 17: Top Positive Correlations Heatmap
# =====================================================
fig = plt.figure(figsize=(12, 10))
top_features = target_corr.nlargest(8).index.tolist()
top_features_with_target = top_features + ['class']
top_corr = df_numeric[top_features_with_target].corr()
sns.heatmap(top_corr, annot=True, fmt='.2f', cmap='YlOrRd', square=True,
            linewidths=1, cbar_kws={"shrink": 0.8}, annot_kws={"size": 10})
plt.title('17. Top 8 Features Most Correlated with Diabetes', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
save_fig(fig, 'top_features_heatmap.png')

# =====================================================
# VISUALIZATION 18: Bottom/Negative Correlations Heatmap
# =====================================================
fig = plt.figure(figsize=(12, 10))
bottom_features = target_corr.nsmallest(8).index.tolist()
bottom_features_with_target = bottom_features + ['class']
bottom_corr = df_numeric[bottom_features_with_target].corr()
sns.heatmap(bottom_corr, annot=True, fmt='.2f', cmap='Blues', square=True,
            linewidths=1, cbar_kws={"shrink": 0.8}, annot_kws={"size": 10})
plt.title('18. Features Least Correlated with Diabetes', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
save_fig(fig, 'bottom_features_heatmap.png')

# =====================================================
# ADDITIONAL CORRELATION ANALYSES (19–22)
# =====================================================
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 19. Symptom-Symptom Correlations
ax = axes[0,0]
symptom_corr = df_numeric[symptom_cols].corr()
mask = np.tril(np.ones_like(symptom_corr, dtype=bool))
sns.heatmap(symptom_corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink":0.8})
ax.set_title('19. Symptom-Symptom Correlations', fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# 20. Correlation with Age
ax = axes[0,1]
age_corr = df_numeric[symptom_cols + ['Gender', 'class']].corrwith(df_numeric['Age']).sort_values()
colors_age = ['red' if x<0 else 'green' for x in age_corr.values]
age_corr.plot(kind='barh', ax=ax, color=colors_age, alpha=0.7)
ax.set_title('20. Feature Correlations with Age', fontweight='bold')
ax.set_xlabel('Correlation with Age')
ax.grid(True, alpha=0.3, axis='x')

# 21. Correlation Difference (Positive - Negative)
ax = axes[1,0]
pos_data = df_numeric[df_numeric['class']==1][symptom_cols]
neg_data = df_numeric[df_numeric['class']==0][symptom_cols]
corr_diff = pos_data.corr() - neg_data.corr()
sns.heatmap(corr_diff, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink":0.8})
ax.set_title('21. Correlation Difference (Positive - Negative)', fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# 22. Correlation with Symptom Count
ax = axes[1,1]
df_numeric['symptom_count'] = df_numeric[symptom_cols].sum(axis=1)
count_corr = df_numeric[symptom_cols + ['Age', 'Gender', 'class']].corrwith(df_numeric['symptom_count']).sort_values()
count_corr.plot(kind='bar', ax=ax, color='purple', alpha=0.7)
ax.set_title('22. Feature Correlations with Total Symptom Count', fontweight='bold')
ax.set_ylabel('Correlation with Symptom Count')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('ADDITIONAL CORRELATION ANALYSES', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
save_fig(fig, 'additional_correlations.png')

# =====================================================
# SUMMARY PRINTS
# =====================================================
print("\n" + "="*60)
print("CORRELATION ANALYSIS COMPLETE")
print("="*60)
print("\nTop 5 Features Most Correlated with Diabetes:")
print(target_corr.tail(5).to_string())
print("\nTop 5 Features Least Correlated with Diabetes:")
print(target_corr.head(5).to_string())

# Strong inter-symptom correlations (>0.5)
strong_pairs = [(i,j,symptom_corr.loc[i,j]) for i in symptom_cols for j in symptom_cols if i<j and abs(symptom_corr.loc[i,j])>0.5]
print("\nFeatures with Strongest Inter-correlations:")
if strong_pairs:
    for pair in strong_pairs:
        print(f"  {pair[0]} & {pair[1]}: {pair[2]:.3f}")
else:
    print("  No strong correlations (>0.5) between symptoms")

print("\nAll correlation visualizations saved in:", CORR_OUTPUT)
print("="*60)