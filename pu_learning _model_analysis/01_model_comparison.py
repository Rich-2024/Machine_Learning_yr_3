import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

print("\n==============================")
print("EXPLAINABLE AI ANALYSIS")
print("==============================")

# -----------------------------------
# CONFUSION MATRIX (Random Forest)
# -----------------------------------

cm = confusion_matrix(y_test, rf_predictions)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative","Positive"],
    yticklabels=["Negative","Positive"]
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

# -----------------------------------
# FEATURE IMPORTANCE
# -----------------------------------

print("\nRandom Forest Feature Importance")

importances = rf.feature_importances_

importance_df = pd.DataFrame({
    "Feature": symptom_cols,
    "Importance": importances
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
)

print(importance_df)

plt.figure(figsize=(10,6))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)

plt.title("Feature Importance for Diabetes Prediction")

plt.show()

# -----------------------------------
# SHAP EXPLAINABLE AI
# -----------------------------------

print("\nRunning SHAP Analysis...")

explainer = shap.TreeExplainer(rf)

shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=symptom_cols
)

# Bar importance plot
shap.summary_plot(
    shap_values,
    X_test,
    feature_names=symptom_cols,
    plot_type="bar"
)

# -----------------------------------
# INDIVIDUAL PREDICTION EXPLANATION
# -----------------------------------

print("\nExample Patient Explanation")

sample_index = 0

shap.force_plot(
    explainer.expected_value,
    shap_values[sample_index],
    X_test[sample_index],
    feature_names=symptom_cols,
    matplotlib=True
)