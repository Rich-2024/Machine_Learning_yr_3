# ============================================
# DIABETES RISK PREDICTION - ASSIGNMENT VERSION
# ============================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import shap
import warnings

warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# --------------------------
# 1. LOAD DATA
# --------------------------
df = pd.read_csv('../diabetes_data_upload.csv')

# Convert categorical features to numeric
df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})
df['class'] = df['class'].map({'Positive':1, 'Negative':0})

symptom_cols = ['Polyuria','Polydipsia','sudden weight loss','weakness','Polyphagia',
                'Genital thrush','visual blurring','Itching','Irritability',
                'delayed healing','partial paresis','muscle stiffness','Alopecia','Obesity']

for col in symptom_cols:
    df[col] = df[col].map({'Yes':1, 'No':0})

X = df[symptom_cols].values
y = df['class'].values

# --------------------------
# 2. PREPROCESSING
# --------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Positive cases: {sum(y)}, Negative cases: {len(y)-sum(y)}")

# --------------------------
# 3. TRAIN-TEST SPLIT
# --------------------------
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# --------------------------
# 4. SUPERVISED MODELS
# --------------------------

# --- Logistic Regression ---
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train, y_train)
y_pred_lr = logreg.predict(X_test)

# --- Random Forest ---
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

# --------------------------
# 5. SEMI-SUPERVISED MODELS
# --------------------------
# Create semi-supervised labels (20% labeled)
def create_semi_supervised_labels(y, labeled_ratio=0.2, random_state=42):
    np.random.seed(random_state)
    y_semi = y.copy()
    n_labeled = int(len(y) * labeled_ratio)
    labeled_idx = np.random.choice(len(y), n_labeled, replace=False)
    unlabeled_idx = np.array([i for i in range(len(y)) if i not in labeled_idx])
    y_semi[unlabeled_idx] = -1
    return y_semi, labeled_idx, unlabeled_idx

y_semi, labeled_idx, unlabeled_idx = create_semi_supervised_labels(y_train, 0.2)

# --- Label Propagation ---
lp = LabelPropagation(kernel='knn', n_neighbors=5, max_iter=1000)
lp.fit(X_train, y_semi)
y_pred_lp = lp.predict(X_test)

# --- Label Spreading ---
ls = LabelSpreading(kernel='knn', n_neighbors=5, alpha=0.5, max_iter=1000)
ls.fit(X_train, y_semi)
y_pred_ls = ls.predict(X_test)

# --------------------------
# 6. EVALUATION FUNCTION
# --------------------------
def evaluate_model(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    print(f"\n{model_name} Evaluation:")
    print(f"Accuracy: {acc:.3f}, F1: {f1:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}")
    return acc, f1, prec, rec

results = {}
results['Logistic Regression'] = evaluate_model(y_test, y_pred_lr, 'Logistic Regression')
results['Random Forest'] = evaluate_model(y_test, y_pred_rf, 'Random Forest')
results['Label Propagation'] = evaluate_model(y_test, y_pred_lp, 'Label Propagation')
results['Label Spreading'] = evaluate_model(y_test, y_pred_ls, 'Label Spreading')

# --------------------------
# 7. FEATURE IMPORTANCE / EXPLAINABILITY (RF + SHAP)
# --------------------------
explainer = shap.Explainer(rf, X_train)
shap_values = explainer(X_test)

# Plot top features
shap.summary_plot(shap_values, features=X_test, feature_names=symptom_cols, show=True)

# --------------------------
# 8. COMPARISON TABLE
# --------------------------
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m][0] for m in results],
    'F1-Score': [results[m][1] for m in results],
    'Precision': [results[m][2] for m in results],
    'Recall': [results[m][3] for m in results]
})
print("\nComparison Table:\n", comparison_df)

# --------------------------
# 9. OPTIONAL: PCA VISUALIZATION
# --------------------------
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure(figsize=(10,6))
plt.scatter(X_pca[:,0], X_pca[:,1], c=y, cmap='coolwarm', alpha=0.7)
plt.title("Diabetes Dataset PCA 2D Visualization")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.colorbar(label='Diabetes Risk')
plt.show()