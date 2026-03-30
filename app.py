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
    # Convert Yes/No to 1/0
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
custom_css = """
/* Global */
html, body, .gradio-container {
    background: #f4f8fc !important;
    color: #123047 !important;
    font-family: 'Segoe UI', Roboto, Arial, sans-serif !important;
}

body {
    background-color: #f4f8fc !important;
    color: #123047 !important;
}

.gradio-container {
    max-width: 1180px !important;
    margin: 0 auto !important;
    padding-top: 24px !important;
    padding-bottom: 24px !important;
}

/* Main shell */
.app-shell {
    background: #ffffff !important;
    border: 1px solid #dbe7f3 !important;
    border-radius: 24px !important;
    box-shadow: 0 12px 40px rgba(31, 72, 115, 0.08) !important;
    padding: 28px !important;
}

/* Header */
.hero-card {
    background: linear-gradient(135deg, #f8fcff 0%, #eef6fd 100%) !important;
    border: 1px solid #d8e8f7 !important;
    border-radius: 20px !important;
    padding: 24px !important;
    margin-bottom: 20px !important;
}

.hero-title {
    font-size: 2rem !important;
    font-weight: 700 !important;
    color: #0f4c81 !important;
    margin-bottom: 8px !important;
}

.hero-subtitle {
    font-size: 1rem !important;
    color: #476781 !important;
    margin-bottom: 0 !important;
}

/* Instructions */
.instruction-box {
    background: #f2f8fe !important;
    border-left: 5px solid #1976d2 !important;
    padding: 16px 18px !important;
    border-radius: 16px !important;
    margin-bottom: 22px !important;
    color: #1f3b57 !important;
    line-height: 1.6 !important;
}

.section-title {
    font-size: 1.15rem !important;
    font-weight: 700 !important;
    color: #0f4c81 !important;
    margin: 12px 0 16px 0 !important;
}

/* Symptom cards */
.symptom-card {
    background: linear-gradient(135deg, #1e88e5 0%, #1565c0 100%) !important;
    border: 1px solid #1565c0 !important;
    border-radius: 18px !important;
    padding: 14px 16px !important;
    box-shadow: 0 6px 18px rgba(25, 118, 210, 0.18) !important;
    margin-bottom: 12px !important;
}

/* Make everything in symptom cards white */
.symptom-card,
.symptom-card *,
.symptom-card label,
.symptom-card span,
.symptom-card p,
.symptom-card div {
    color: #ffffff !important;
}

/* Radio area */
.symptom-card fieldset {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}

.symptom-card input[type="radio"] {
    accent-color: #ffffff !important;
}

/* Fix internal wrappers across Gradio versions */
.symptom-card .wrap,
.symptom-card .form,
.symptom-card .block,
.symptom-card .gr-box,
.symptom-card .gr-form,
.symptom-card .gr-input,
.symptom-card .gradio-radio {
    background: transparent !important;
    color: #ffffff !important;
}

/* Buttons */
button {
    border-radius: 999px !important;
    font-weight: 700 !important;
    padding: 0.85rem 1.5rem !important;
    border: none !important;
    box-shadow: none !important;
    transition: 0.2s ease !important;
}

button:hover {
    transform: translateY(-1px);
}

/* Button colors */
#predict-btn {
    background: #1976d2 !important;
    color: #ffffff !important;
}

#clear-btn {
    background: #eaf3fb !important;
    color: #123047 !important;
    border: 1px solid #cfe0ef !important;
}

/* Result output */
#result-box {
    margin-top: 14px !important;
}

#result-box,
#result-box * {
    opacity: 1 !important;
}

/* Strong result card */
.result-card {
    background: #ffffff !important;
    border: 2px solid #90caf9 !important;
    border-radius: 20px !important;
    padding: 22px !important;
    box-shadow: 0 10px 24px rgba(25, 118, 210, 0.12) !important;
    color: #0f172a !important;
}

.result-card h3 {
    margin-top: 0 !important;
    margin-bottom: 18px !important;
    font-size: 1.4rem !important;
    font-weight: 800 !important;
    color: #0b3d91 !important;
}

/* Result rows */
.result-row {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    gap: 16px !important;
    padding: 12px 0 !important;
    border-bottom: 1px solid #e3eef9 !important;
}

.result-label {
    font-size: 1.02rem !important;
    font-weight: 700 !important;
    color: #1e293b !important;
}

/* Prediction badge */
.prediction-badge {
    display: inline-block !important;
    padding: 8px 16px !important;
    border-radius: 999px !important;
    font-size: 0.95rem !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    letter-spacing: 0.2px !important;
}

.prediction-positive {
    background: #d32f2f !important;
}

.prediction-negative {
    background: #2e7d32 !important;
}

/* Risk badge */
.risk-badge {
    display: inline-block !important;
    padding: 8px 16px !important;
    border-radius: 999px !important;
    font-size: 0.95rem !important;
    font-weight: 800 !important;
}

/* Risk colors */
.low-risk {
    background: #e8f5e9 !important;
    color: #1b5e20 !important;
    border: 1px solid #a5d6a7 !important;
}

.moderate-risk {
    background: #fff8e1 !important;
    color: #8a5a00 !important;
    border: 1px solid #ffd54f !important;
}

.high-risk {
    background: #ffebee !important;
    color: #b71c1c !important;
    border: 1px solid #ef9a9a !important;
}

/* Probability */
.probability-row {
    margin-bottom: 6px !important;
}

.probability-value {
    font-size: 1.25rem !important;
    font-weight: 900 !important;
    color: #0b3d91 !important;
}

/* Recommendation */
.recommendation-box {
    margin-top: 18px !important;
    background: #f7fbff !important;
    border-left: 5px solid #1976d2 !important;
    border-radius: 14px !important;
    padding: 16px !important;
}

.recommendation-title {
    font-size: 1rem !important;
    font-weight: 800 !important;
    color: #0b3d91 !important;
    margin-bottom: 8px !important;
}

.recommendation-text {
    font-size: 1rem !important;
    line-height: 1.6 !important;
    color: #1e293b !important;
    font-weight: 600 !important;
}

/* Footer */
.footer-note {
    margin-top: 18px;
    color: #5b7690;
    font-size: 0.92rem;
    text-align: center;
}

/* Avoid faint markdown styles */
.prose, .markdown, div[data-testid="markdown"] {
    background: transparent !important;
    color: #123047 !important;
    opacity: 1 !important;
}
"""

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
# Launch app
# --------------------------
if __name__ == "__main__":
    demo.launch()
