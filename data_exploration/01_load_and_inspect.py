"""
Diabetes Prediction Project
File: 01_load_and_inspect.py
Purpose: Load dataset and perform initial inspection
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load the data
print("="*60)
print("DIABETES PREDICTION PROJECT - DATA INSPECTION")
print("="*60)

df = pd.read_csv('../diabetes_data_upload.csv')

# 1. Basic dataset information
print("\n" + "="*50)
print("BASIC DATASET INFORMATION")
print("="*50)
print(f"Dataset Shape: {df.shape}")
print(f"\nNumber of Rows: {df.shape[0]}")
print(f"Number of Columns: {df.shape[1]}")

print("\n" + "-"*40)
print("COLUMN NAMES:")
print("-"*40)
for i, col in enumerate(df.columns, 1):
    print(f"{i}. {col}")

print("\n" + "-"*40)
print("DATA TYPES:")
print("-"*40)
print(df.dtypes)

# 2. First few rows
print("\n" + "-"*40)
print("FIRST 5 ROWS:")
print("-"*40)
print(df.head())

# 3. Last few rows
print("\n" + "-"*40)
print("LAST 5 ROWS:")
print("-"*40)
print(df.tail())

# 4. Random sample
print("\n" + "-"*40)
print("RANDOM SAMPLE OF 5 ROWS:")
print("-"*40)
print(df.sample(5, random_state=42))

# 5. Missing values check
print("\n" + "="*50)
print("MISSING VALUES CHECK")
print("="*50)
missing_values = df.isnull().sum()
print(missing_values)
print(f"\nTotal Missing Values: {missing_values.sum()}")

# 6. Basic statistics for numeric columns
print("\n" + "="*50)
print("NUMERIC COLUMN STATISTICS")
print("="*50)
print("Age Statistics:")
print(df['Age'].describe())

# 7. Target variable distribution
print("\n" + "="*50)
print("TARGET VARIABLE DISTRIBUTION")
print("="*50)
target_counts = df['class'].value_counts()
print(target_counts)
print(f"\nPercentage Distribution:")
print(df['class'].value_counts(normalize=True) * 100)

# 8. Gender distribution
print("\n" + "="*50)
print("GENDER DISTRIBUTION")
print("="*50)
gender_counts = df['Gender'].value_counts()
print(gender_counts)
print(f"\nPercentage Distribution:")
print(df['Gender'].value_counts(normalize=True) * 100)

# 9. Identify column types
print("\n" + "="*50)
print("COLUMN CLASSIFICATION")
print("="*50)
symptom_cols = ['Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 
                'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 
                'Irritability', 'delayed healing', 'partial paresis', 
                'muscle stiffness', 'Alopecia', 'Obesity']

print(f"Demographic Features: Age, Gender")
print(f"Symptom Features ({len(symptom_cols)}):")
for i, symptom in enumerate(symptom_cols, 1):
    print(f"  {i}. {symptom}")
print(f"Target Variable: class (Positive/Negative)")

# 10. Quick summary statistics for categorical columns
print("\n" + "="*50)
print("CATEGORICAL COLUMNS SUMMARY")
print("="*50)
for col in ['Gender'] + symptom_cols + ['class']:
    print(f"\n{col}:")
    print(df[col].value_counts())

# Save basic info to a text file for reference
with open('../data_exploration/outputs/dataset_info.txt', 'w') as f:
    f.write("DIABETES PREDICTION DATASET INFORMATION\n")
    f.write("="*50 + "\n")
    f.write(f"Dataset Shape: {df.shape}\n")
    f.write(f"\nColumns:\n")
    for col in df.columns:
        f.write(f"  - {col}: {df[col].dtype}\n")
    f.write(f"\nMissing Values: {missing_values.sum()}\n")
    f.write(f"\nTarget Distribution:\n")
    f.write(f"  Positive: {target_counts['Positive']} ({target_counts['Positive']/len(df)*100:.1f}%)\n")
    f.write(f"  Negative: {target_counts['Negative']} ({target_counts['Negative']/len(df)*100:.1f}%)\n")

print("\n" + "="*60)
print("DATA INSPECTION COMPLETE - Check outputs/dataset_info.txt")
print("="*60)