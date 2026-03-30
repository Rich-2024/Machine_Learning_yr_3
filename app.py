import os
import joblib
import numpy as np
import gradio as gr

# --------------------------
# Load model and scaler
# --------------------------
model_dir = "models"
rf = joblib.load(os.path.join(model_dir, "random_forest.joblib"))
scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))

# Symptom columns (must match training order)
symptom_cols = [
    "Polyuria",
    "Polydipsia",
    "sudden weight loss",
    "weakness",
    "Polyphagia",
    "Genital thrush",
    "visual blurring",
    "Itching",
    "Irritability",
    "delayed healing",
    "partial paresis",
    "muscle stiffness",
    "Alopecia",
    "Obesity",
]

# --------------------------
# Recommendation logic
# --------------------------
def get_recommendation(probability):
    if probability < 20:
        return "Very low risk. Maintain a healthy lifestyle and continue regular medical checkups."
    elif probability < 50:
        return "Moderate risk. Monitor symptoms closely and consider consulting a healthcare professional."
    else:
        return "High risk. Strongly recommend visiting a healthcare professional for further evaluation and testing."

# --------------------------
# Prediction function
# --------------------------
def predict_diabetes(*args):
    mapped = [1 if value == "Yes" else 0 for value in args]
    patient_data = np.array([mapped])
    patient_scaled = scaler.transform(patient_data)

    prediction = rf.predict(patient_scaled)[0]
    probability = rf.predict_proba(patient_scaled)[0][1] * 100
    recommendation = get_recommendation(probability)

    class_label = "Positive" if prediction == 1 else "Negative"

    if probability < 20:
        risk_badge = "Low Risk"
        risk_class = "low-risk"
    elif probability < 50:
        risk_badge = "Moderate Risk"
        risk_class = "moderate-risk"
    else:
        risk_badge = "High Risk"
        risk_class = "high-risk"

    prediction_class = "prediction-positive" if prediction == 1 else "prediction-negative"

    result = f"""
    <div class="result-card">
        <h3>📊 Prediction Result</h3>

        <div class="result-row">
            <span class="result-label">Prediction</span>
            <span class="prediction-badge {prediction_class}">{class_label}</span>
        </div>

        <div class="result-row">
            <span class="result-label">Risk Level</span>
            <span class="risk-badge {risk_class}">{risk_badge}</span>
        </div>

        <div class="result-row probability-row">
            <span class="result-label">Risk Probability</span>
            <span class="probability-value">{probability:.1f}%</span>
        </div>

        <div class="recommendation-box">
            <div class="recommendation-title">Recommendation</div>
            <div class="recommendation-text">{recommendation}</div>
        </div>
    </div>
    """
    return result

# --------------------------
# Clear function
# --------------------------
def clear_form():
    return ["No"] * len(symptom_cols) + [""]

# --------------------------
# Custom CSS
# --------------------------
custom_css = """..."""  # Keep all your existing CSS here

# --------------------------
# Create inputs
# --------------------------
inputs = [
    gr.Radio(
        choices=["No", "Yes"],
        value="No",
        label=col,
        elem_classes="symptom-card"
    )
    for col in symptom_cols
]

output = gr.HTML(label="Result", elem_id="result-box")

# --------------------------
# Build interface
# --------------------------
with gr.Blocks(css=custom_css, title="Diabetes Risk Predictor") as demo:
    with gr.Column(elem_classes="app-shell"):
        gr.Markdown(
            """
            <div class="hero-card">
                <div class="hero-title">🩺 Diabetes Risk Assessment</div>
                <p class="hero-subtitle">
                    Clinical-style screening interface powered by a Random Forest model for rapid symptom-based diabetes risk estimation.
                </p>
            </div>
            """
        )

        gr.Markdown(
            """
            <div class="instruction-box">
                <strong>📋 Instructions for medical staff or screening users:</strong><br>
                For each symptom below, choose <strong>Yes</strong> if the patient currently presents the symptom,
                or <strong>No</strong> if the symptom is absent.
                The system will estimate diabetes risk probability and provide a recommendation.
            </div>
            """
        )

        gr.Markdown('<div class="section-title">Patient Symptom Checklist</div>')

        half = len(inputs) // 2

        with gr.Row():
            with gr.Column():
                for inp in inputs[:half]:
                    inp.render()

            with gr.Column():
                for inp in inputs[half:]:
                    inp.render()

        with gr.Row():
            predict_btn = gr.Button("Predict Risk", elem_id="predict-btn")
            clear_btn = gr.Button("Clear Form", elem_id="clear-btn")

        output.render()

        gr.Markdown(
            """
            <div class="footer-note">
                This tool is intended for screening support only and does not replace professional medical diagnosis.
            </div>
            """
        )

        predict_btn.click(
            fn=predict_diabetes,
            inputs=inputs,
            outputs=output
        )

        clear_btn.click(
            fn=clear_form,
            inputs=[],
            outputs=inputs + [output]
        )

# --------------------------
# Launch app (Render-ready)
# --------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    demo.launch(server_name="0.0.0.0", server_port=port, show_error=True)