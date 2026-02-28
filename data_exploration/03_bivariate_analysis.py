"""
Diabetes Prediction Project
File: 03_bivariate_analysis.py
Purpose: Create bivariate visualizations (Visualizations 8–18)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =====================================================
# CONFIGURATION
# =====================================================

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Base directories
BASE_OUTPUT = '../data_exploration/outputs'
BIVARIATE_OUTPUT = os.path.join(BASE_OUTPUT, 'bivariate')

# Create directories if they do not exist
os.makedirs(BASE_OUTPUT, exist_ok=True)
os.makedirs(BIVARIATE_OUTPUT, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv('../diabetes_data_upload.csv')

symptom_cols = [
    'Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness',
    'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching',
    'Irritability', 'delayed healing', 'partial paresis',
    'muscle stiffness', 'Alopecia', 'Obesity'
]

# =====================================================
# HELPER FUNCTION
# =====================================================

def save_figure(fig, path, filename):
    os.makedirs(path, exist_ok=True)
    fig.savefig(os.path.join(path, filename),
                dpi=300, bbox_inches='tight')
    print(f"Saved: {os.path.join(path, filename)}")

# =====================================================
# VISUALIZATIONS 8–14 (PART 1)
# =====================================================

fig1, axes1 = plt.subplots(3, 3, figsize=(18, 15))
axes1 = axes1.ravel()

for i, symptom in enumerate(symptom_cols[:9]):
    ax = axes1[i]
    ct = pd.crosstab(df[symptom], df['class'])

    ct.plot(kind='bar', stacked=False, ax=ax,
            color=['lightcoral', 'lightblue'],
            edgecolor='black', width=0.8)

    ax.set_title(f'8.{i+1}. {symptom} vs Diabetes',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel(symptom)
    ax.set_ylabel('Count')
    ax.set_xticklabels(['No', 'Yes'], rotation=0)
    ax.legend(['Positive', 'Negative'], title='Class')
    ax.grid(True, alpha=0.3, axis='y')

    for container in ax.containers:
        ax.bar_label(container, fontweight='bold')

plt.suptitle('BIVARIATE ANALYSIS - Symptom vs Diabetes (Part 1/3)',
             fontsize=16, fontweight='bold')

plt.tight_layout()
save_figure(fig1, BASE_OUTPUT,
            'bivariate_symptoms_vs_diabetes_part1.png')
plt.close(fig1)

# =====================================================
# VISUALIZATIONS 15–19 (PART 2)
# =====================================================

fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
axes2 = axes2.ravel()

for i, symptom in enumerate(symptom_cols[9:]):
    ax = axes2[i]
    ct = pd.crosstab(df[symptom], df['class'])

    ct.plot(kind='bar', stacked=False, ax=ax,
            color=['lightcoral', 'lightblue'],
            edgecolor='black', width=0.8)

    ax.set_title(f'8.{i+10}. {symptom} vs Diabetes',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel(symptom)
    ax.set_ylabel('Count')
    ax.set_xticklabels(['No', 'Yes'], rotation=0)
    ax.legend(['Positive', 'Negative'], title='Class')
    ax.grid(True, alpha=0.3, axis='y')

    for container in ax.containers:
        ax.bar_label(container, fontweight='bold')

axes2[5].set_visible(False)

plt.suptitle('BIVARIATE ANALYSIS - Symptom vs Diabetes (Part 2/3)',
             fontsize=16, fontweight='bold')

plt.tight_layout()
save_figure(fig2, BASE_OUTPUT,
            'bivariate_symptoms_vs_diabetes_part2.png')
plt.close(fig2)

# =====================================================
# ADDITIONAL BIVARIATE ANALYSIS (PART 3)
# =====================================================

fig3, axes3 = plt.subplots(2, 3, figsize=(18, 12))
axes3 = axes3.ravel()

# 1️⃣ Symptom Prevalence by Gender
ax = axes3[0]
symptom_by_gender = df.groupby('Gender')[symptom_cols] \
                       .apply(lambda x: (x == 'Yes').sum())

symptom_by_gender.T.plot(kind='bar', ax=ax,
                         color=['lightblue', 'lightcoral'],
                         edgecolor='black')

ax.set_title('15. Symptom Prevalence by Gender',
             fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(),
                   rotation=45, ha='right')
ax.grid(True, alpha=0.3, axis='y')

# 2️⃣ Symptom Prevalence by Age Group
ax = axes3[1]

df['Age_Group'] = pd.cut(df['Age'],
                         bins=[0, 30, 40, 50, 60, 100],
                         labels=['<30', '30-40',
                                 '40-50', '50-60', '60+'])

symptom_by_age = df.groupby('Age_Group')[symptom_cols] \
                    .apply(lambda x: (x == 'Yes').mean()) * 100

sns.heatmap(symptom_by_age,
            annot=True, fmt='.1f',
            cmap='YlOrRd',
            ax=ax,
            cbar_kws={'label': 'Percentage (%)'})

ax.set_title('16. Symptom Prevalence by Age Group (%)',
             fontweight='bold')
ax.set_xticklabels(ax.get_xticklabels(),
                   rotation=45, ha='right')

# 3️⃣ Average Age Difference
ax = axes3[2]

avg_age_diff = []
for symptom in symptom_cols:
    with_s = df[df[symptom] == 'Yes']['Age'].mean()
    without_s = df[df[symptom] == 'No']['Age'].mean()
    avg_age_diff.append(with_s - without_s)

bars = ax.barh(range(len(symptom_cols)),
               avg_age_diff,
               color=['red' if x > 0 else 'blue'
                      for x in avg_age_diff])

ax.set_yticks(range(len(symptom_cols)))
ax.set_yticklabels(symptom_cols, fontsize=8)
ax.axvline(x=0, color='black')
ax.set_title('17. Avg Age Difference (With - Without)',
             fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

for bar, val in zip(bars, avg_age_diff):
    ax.text(val, bar.get_y() + bar.get_height()/2,
            f'{val:.1f}', va='center')

# 4️⃣ Gender-Age Interaction
ax = axes3[3]

for gender in df['Gender'].unique():
    subset = df[df['Gender'] == gender]
    for status in df['class'].unique():
        status_subset = subset[subset['class'] == status]
        ax.hist(status_subset['Age'],
                bins=15,
                alpha=0.5,
                label=f'{gender}-{status}',
                density=True)

ax.set_title('18. Age Distribution by Gender & Diabetes',
             fontweight='bold')
ax.legend(bbox_to_anchor=(1.05, 1),
          loc='upper left')
ax.grid(True, alpha=0.3)

axes3[4].set_visible(False)
axes3[5].set_visible(False)

plt.suptitle('ADDITIONAL BIVARIATE ANALYSIS (Part 3/3)',
             fontsize=16, fontweight='bold')

plt.tight_layout()
save_figure(fig3, BIVARIATE_OUTPUT,
            'bivariate_symptoms_vs_diabetes_part3.png')
plt.close(fig3)

print("\n" + "="*60)
print("BIVARIATE ANALYSIS COMPLETE")
print("="*60)