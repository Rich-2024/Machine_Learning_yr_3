"""
Diabetes Prediction Project
File: 02_univariate_analysis.py
Purpose: Create univariate visualizations (Visualizations 1-7)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load data
df = pd.read_csv('../diabetes_data_upload.csv')

# Ensure output directory exists
output_dir = '../data_exploration/outputs'
os.makedirs(output_dir, exist_ok=True)

# Create figure for 7 visualizations
fig = plt.figure(figsize=(20, 15))

# ============================================
# VISUALIZATION 1: Age Distribution Histogram
# ============================================
ax1 = plt.subplot(3, 3, 1)
ax1.hist(df['Age'], bins=20, edgecolor='black', alpha=0.7, color='skyblue')
ax1.set_title('1. Age Distribution of Patients', fontsize=14, fontweight='bold')
ax1.set_xlabel('Age', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.grid(True, alpha=0.3)

# Add mean and median lines
mean_age = df['Age'].mean()
median_age = df['Age'].median()
ax1.axvline(mean_age, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_age:.1f}')
ax1.axvline(median_age, color='green', linestyle='--', linewidth=2, label=f'Median: {median_age:.1f}')
ax1.legend()

# ============================================
# VISUALIZATION 2: Age Distribution by Gender
# ============================================
ax2 = plt.subplot(3, 3, 2)
colors = {'Male': 'lightblue', 'Female': 'lightcoral'}
for gender in ['Male', 'Female']:
    subset = df[df['Gender'] == gender]['Age']
    ax2.hist(subset, bins=20, alpha=0.6, label=gender, color=colors[gender])
ax2.set_title('2. Age Distribution by Gender', fontsize=14, fontweight='bold')
ax2.set_xlabel('Age', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.legend()
ax2.grid(True, alpha=0.3)

# ============================================
# VISUALIZATION 3: Gender Distribution (Pie Chart)
# ============================================
ax3 = plt.subplot(3, 3, 3)
gender_counts = df['Gender'].value_counts()
colors_pie = ['lightblue', 'lightcoral']
explode = (0.05, 0.05)
ax3.pie(gender_counts.values, explode=explode, labels=gender_counts.index, 
        colors=colors_pie, autopct='%1.1f%%', shadow=True, startangle=90)
ax3.set_title('3. Gender Distribution', fontsize=14, fontweight='bold')

# ============================================
# VISUALIZATION 4: Target Variable Distribution
# ============================================
ax4 = plt.subplot(3, 3, 4)
class_counts = df['class'].value_counts()
colors_bar = ['lightcoral' if x == 'Positive' else 'lightblue' for x in class_counts.index]
bars = ax4.bar(class_counts.index, class_counts.values, color=colors_bar, edgecolor='black')
ax4.set_title('4. Diabetes Status Distribution', fontsize=14, fontweight='bold')
ax4.set_xlabel('Diabetes Status', fontsize=12)
ax4.set_ylabel('Count', fontsize=12)

# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 5,
             f'{int(height)}', ha='center', va='bottom', fontweight='bold')

# Add percentage labels inside bars
total = len(df)
for i, (index, value) in enumerate(class_counts.items()):
    percentage = (value / total) * 100
    ax4.text(i, value / 2, f'{percentage:.1f}%', ha='center', va='center',
             color='white', fontweight='bold', fontsize=11)

# ============================================
# VISUALIZATION 5: Age Box Plot by Diabetes Status
# ============================================
ax5 = plt.subplot(3, 3, 5)
data_to_plot = [df[df['class'] == 'Positive']['Age'], 
                df[df['class'] == 'Negative']['Age']]
bp = ax5.boxplot(data_to_plot, labels=['Positive', 'Negative'], 
                 patch_artist=True, showmeans=True)
bp['boxes'][0].set_facecolor('lightcoral')
bp['boxes'][1].set_facecolor('lightblue')
bp['means'][0].set_markerfacecolor('black')
bp['means'][1].set_markerfacecolor('black')
ax5.set_title('5. Age Distribution by Diabetes Status', fontsize=14, fontweight='bold')
ax5.set_xlabel('Diabetes Status', fontsize=12)
ax5.set_ylabel('Age', fontsize=12)
ax5.grid(True, alpha=0.3, axis='y')

# ============================================
# VISUALIZATION 6: Gender-wise Diabetes Count
# ============================================
ax6 = plt.subplot(3, 3, 6)
gender_class = pd.crosstab(df['Gender'], df['class'])
gender_class.plot(kind='bar', ax=ax6, color=['lightcoral', 'lightblue'], edgecolor='black')
ax6.set_title('6. Diabetes Status by Gender', fontsize=14, fontweight='bold')
ax6.set_xlabel('Gender', fontsize=12)
ax6.set_ylabel('Count', fontsize=12)
ax6.legend(title='Class')
ax6.set_xticklabels(['Female', 'Male'], rotation=0)
ax6.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for container in ax6.containers:
    ax6.bar_label(container, fontweight='bold')

# ============================================
# VISUALIZATION 7: Top Symptoms Frequency
# ============================================
ax7 = plt.subplot(3, 3, 7)
symptom_cols = ['Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 
                'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 
                'Irritability', 'delayed healing', 'partial paresis', 
                'muscle stiffness', 'Alopecia', 'Obesity']

symptom_freq = df[symptom_cols].apply(lambda x: (x == 'Yes').sum()).sort_values(ascending=True)
colors_symptoms = plt.cm.viridis(np.linspace(0.2, 0.9, len(symptom_freq)))
bars = ax7.barh(range(len(symptom_freq)), symptom_freq.values, color=colors_symptoms)
ax7.set_yticks(range(len(symptom_freq)))
ax7.set_yticklabels(symptom_freq.index, fontsize=10)
ax7.set_title('7. Symptom Frequency (All Patients)', fontsize=14, fontweight='bold')
ax7.set_xlabel('Frequency', fontsize=12)
ax7.set_ylabel('Symptoms', fontsize=12)
ax7.set_xlim(0, symptom_freq.values.max() * 1.1)  # Add padding for label visibility

# Add value labels
for i, (bar, val) in enumerate(zip(bars, symptom_freq.values)):
    ax7.text(val + 2, bar.get_y() + bar.get_height()/2, str(val), 
             va='center', fontweight='bold', fontsize=10)

# Add a text box with total count
total_patients = len(df)
ax7.text(0.95, 0.05, f'Total Patients: {total_patients}', 
         transform=ax7.transAxes, ha='right', va='bottom',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Adjust layout and title
plt.suptitle('UNIVARIATE ANALYSIS - Visualizations 1-7', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.subplots_adjust(top=0.9, hspace=0.4, wspace=0.3)

# Save figure
plt.savefig(os.path.join(output_dir, 'univariate_analysis.png'), dpi=300, bbox_inches='tight')

# Show figure
plt.show()

print("\n" + "="*60)
print("UNIVARIATE ANALYSIS COMPLETE - 7 Visualizations Created")
print("="*60)
print(f"Files saved:\n  -- {os.path.join(output_dir, 'univariate_analysis.png')}")