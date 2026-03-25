"""
Diabetes Prediction Project - PU Learning
File: 01_all_pu_learning.py
Purpose: Implement Positive-Unlabeled (PU) Learning algorithms without errors
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, roc_curve, auc
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create output directories automatically
output_dir = '../outputs/figures/pu_learning'
os.makedirs(output_dir, exist_ok=True)

# Load and prepare data
print("="*60)
print("POSITIVE-UNLABELED (PU) LEARNING - DIABETES PREDICTION")
print("="*60)

df = pd.read_csv('../diabetes_data_upload.csv')

# Numeric conversion
df_numeric = df.copy()
df_numeric['Gender'] = df_numeric['Gender'].map({'Male': 1, 'Female': 0})
df_numeric['class'] = df_numeric['class'].map({'Positive': 1, 'Negative': 0})

symptom_cols = ['Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 
                'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 
                'Irritability', 'delayed healing', 'partial paresis', 
                'muscle stiffness', 'Alopecia', 'Obesity']

for col in symptom_cols:
    df_numeric[col] = df_numeric[col].map({'Yes': 1, 'No': 0})

# Feature matrix
X = df_numeric[symptom_cols].values
y_true = df_numeric['class'].values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nDataset shape: {X.shape}")
print(f"Total positives: {sum(y_true)}")
print(f"Total negatives: {len(y_true)-sum(y_true)}")

# ============================================
# CREATE PU DATASET
# ============================================
def create_pu_dataset(X, y, n_positive_labeled=50, random_state=42):
    np.random.seed(random_state)
    pos_idx = np.where(y==1)[0]
    neg_idx = np.where(y==0)[0]
    np.random.shuffle(pos_idx)
    labeled_pos = pos_idx[:n_positive_labeled]
    unlabeled = np.concatenate([pos_idx[n_positive_labeled:], neg_idx])
    hidden_pos = pos_idx[n_positive_labeled:]
    y_pu = -np.ones(len(y))
    y_pu[labeled_pos] = 1
    return y_pu, labeled_pos, unlabeled, hidden_pos

n_labeled_pos = 50
y_pu, pos_indices, unlabeled_indices, hidden_pos_indices = create_pu_dataset(X_scaled, y_true, n_positive_labeled=n_labeled_pos)

print("\nPU Dataset Summary:")
print(f"Labeled positives: {len(pos_indices)}")
print(f"Unlabeled: {len(unlabeled_indices)}")
print(f"Hidden positives: {len(hidden_pos_indices)}")
print(f"True negatives in unlabeled: {len(unlabeled_indices)-len(hidden_pos_indices)}")

# ============================================
# METHOD 1: TWO-STEP PU LEARNING
# ============================================
print("\n--- Two-Step PU Learning ---")

# Step 1: Train classifier on positive + tiny negative sample to avoid single-class
fallback_neg_idx = np.random.choice(unlabeled_indices, size=min(5,len(unlabeled_indices)), replace=False)
X_step1 = np.vstack([X_scaled[pos_indices], X_scaled[fallback_neg_idx]])
y_step1 = np.hstack([np.ones(len(pos_indices)), np.zeros(len(fallback_neg_idx))])

two_step_clf = RandomForestClassifier(n_estimators=100, random_state=42)
two_step_clf.fit(X_step1, y_step1)

# Predict probabilities safely
if len(np.unique(y_step1))==1:
    unlabeled_probs = np.zeros(len(unlabeled_indices))
else:
    unlabeled_probs = two_step_clf.predict_proba(X_scaled[unlabeled_indices])[:,1]

# Identify reliable negatives
threshold = np.percentile(unlabeled_probs, 30)
reliable_neg_idx = unlabeled_indices[unlabeled_probs<threshold]

# Step 2: Train final classifier on positive + reliable negatives
X_final = np.vstack([X_scaled[pos_indices], X_scaled[reliable_neg_idx]])
y_final = np.hstack([np.ones(len(pos_indices)), np.zeros(len(reliable_neg_idx))])

# Avoid single-class issue
if len(np.unique(y_final))<2:
    print("⚠ Warning: Only one class found in final training set.")
    two_step_pred = np.zeros(len(X_scaled))
else:
    two_step_clf_final = RandomForestClassifier(n_estimators=100, random_state=42)
    two_step_clf_final.fit(X_final, y_final)
    two_step_pred = two_step_clf_final.predict(X_scaled)
    two_step_proba = two_step_clf_final.predict_proba(X_scaled)[:,1]

# ============================================
# METHOD 2: PU BAGGING
# ============================================
print("\n--- PU Bagging ---")
n_bags = 20
bag_predictions = np.zeros((len(X_scaled), n_bags))
bag_probabilities = np.zeros((len(X_scaled), n_bags))

for i in range(n_bags):
    pos_boot = resample(pos_indices, replace=True, n_samples=len(pos_indices))
    neg_boot = resample(unlabeled_indices, replace=True, n_samples=len(pos_indices))
    X_bag = np.vstack([X_scaled[pos_boot], X_scaled[neg_boot]])
    y_bag = np.hstack([np.ones(len(pos_boot)), np.zeros(len(neg_boot))])
    clf = RandomForestClassifier(n_estimators=50, random_state=i)
    clf.fit(X_bag, y_bag)
    bag_predictions[:,i] = clf.predict(X_scaled)
    bag_probabilities[:,i] = clf.predict_proba(X_scaled)[:,1]

pu_bagging_pred = (bag_predictions.mean(axis=1)>0.5).astype(int)
pu_bagging_proba = bag_probabilities.mean(axis=1)

# ============================================
# METHOD 3: WEIGHTED PU LEARNING
# ============================================
print("\n--- Weighted PU Learning ---")
estimated_pos_ratio = pu_bagging_proba[unlabeled_indices].mean()
pos_weight = 1.0
neg_weight = estimated_pos_ratio/(1-estimated_pos_ratio)
X_weighted = X_scaled[pos_indices.tolist()+unlabeled_indices.tolist()]
y_weighted = np.hstack([np.ones(len(pos_indices)), np.zeros(len(unlabeled_indices))])
sample_weights = np.hstack([np.ones(len(pos_indices))*pos_weight, np.ones(len(unlabeled_indices))*neg_weight])

weighted_clf = RandomForestClassifier(n_estimators=100, random_state=42)
weighted_clf.fit(X_weighted, y_weighted, sample_weight=sample_weights)
weighted_pred = weighted_clf.predict(X_scaled)

# ============================================
# EVALUATION ON UNLABELED DATA
# ============================================
def evaluate_model(name, pred):
    acc = accuracy_score(y_true[unlabeled_indices], pred[unlabeled_indices])
    f1 = f1_score(y_true[unlabeled_indices], pred[unlabeled_indices])
    prec = precision_score(y_true[unlabeled_indices], pred[unlabeled_indices])
    rec = recall_score(y_true[unlabeled_indices], pred[unlabeled_indices])
    print(f"{name} -> Accuracy: {acc:.4f}, F1: {f1:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}")

evaluate_model("Two-Step PU", two_step_pred)
evaluate_model("PU Bagging", pu_bagging_pred)
evaluate_model("Weighted PU", weighted_pred)

# ============================================
# COMPARISON TABLE
# ============================================
comparison_df = pd.DataFrame({
    'Method':['Two-Step PU','PU Bagging','Weighted PU'],
    'Accuracy':[accuracy_score(y_true[unlabeled_indices], two_step_pred[unlabeled_indices]),
                accuracy_score(y_true[unlabeled_indices], pu_bagging_pred[unlabeled_indices]),
                accuracy_score(y_true[unlabeled_indices], weighted_pred[unlabeled_indices])]
})
print("\nComparison Table:")
print(comparison_df)

# ============================================
# SAVE RESULTS
# ============================================
np.save(os.path.join(output_dir,'two_step_pred.npy'), two_step_pred)
np.save(os.path.join(output_dir,'pu_bagging_pred.npy'), pu_bagging_pred)
np.save(os.path.join(output_dir,'weighted_pred.npy'), weighted_pred)

print("\nPU Learning complete. Predictions saved in:", output_dir)