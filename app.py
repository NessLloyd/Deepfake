import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import os

st.set_page_config(page_title="Deepfake Detection", layout="wide")

# Advanced CSS and animations
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600&display=swap" rel="stylesheet">
<style>
/* General Layout */
html, body, .stApp {
    font-family: 'Inter', sans-serif;
    background-color: #f9f9fb;
    color: #111;
}
.block-container {
    padding-top: 6rem;
}

/* Top Navbar */
.navbar {
    background-color: white;
    padding: 15px 30px;
    position: fixed;
    top: 0;
    width: 100%;
    z-index: 999;
    box-shadow: 0 2px 6px rgba(0,0,0,0.1);
    font-weight: 600;
    font-size: 18px;
}

/* Hero Section */
.hero {
    background: linear-gradient(135deg, #ffffff, #f1f3f7);
    border-radius: 16px;
    padding: 80px 40px 60px 40px;
    text-align: center;
    margin-bottom: 60px;
    animation: fadeIn 1.2s ease-out both;
}
.hero h1 {
    font-size: 3rem;
    font-weight: 700;
    margin-bottom: 12px;
    color: #1a1a1a;
}
.hero p {
    font-size: 1.2rem;
    color: #555;
    max-width: 720px;
    margin: auto;
}

/* File Upload Box */
.stFileUploader {
    border: 2px dashed #ccc;
    padding: 20px;
    border-radius: 12px;
    background-color: white;
    transition: all 0.3s ease;
}
.stFileUploader:hover {
    background-color: #f0f0f0;
}

/* Buttons */
.stButton > button {
    background-color: #2e3a59;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
    transition: all 0.3s ease;
}
.stButton > button:hover {
    background-color: #1c2333;
    transform: scale(1.02);
}

/* Prediction Box */
.prediction-box {
    background-color: #ffffff;
    border: 1px solid #ddd;
    border-left: 8px solid #4CAF50;
    padding: 24px;
    border-radius: 10px;
    margin-top: 20px;
    animation: fadeIn 1s ease-out both;
}
.prediction-fake {
    border-left-color: #e53935 !important;
}
.prediction-real {
    border-left-color: #43a047 !important;
}

/* Fade-in animations */
.fade-in-section {
    opacity: 0;
    transform: translateY(20px);
    animation: fadeInUp 1s ease forwards;
}
@keyframes fadeInUp {
    to {
        opacity: 1;
        transform: translateY(0);
    }
}
@keyframes fadeIn {
    from {
        opacity: 0;
        transform: translateY(40px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Footer */
.footer {
    text-align: center;
    font-size: 14px;
    color: #777;
    padding: 30px 0;
    margin-top: 80px;
}
.footer a {
    color: #2e3a59;
    text-decoration: none;
}
</style>
""", unsafe_allow_html=True)

# Navbar
st.markdown("<div class='navbar'>Deepfake Detection Platform</div>", unsafe_allow_html=True)

# Hero
st.markdown("""
<div class='hero'>
    <h1>AI vs Reality: Deepfake Detection</h1>
    <p>Upload a facial image and our trained AI model will determine whether the image is authentic or artificially generated.</p>
</div>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_best_model():
    return load_model("best_model.keras")

model = load_best_model()

# Upload section
st.markdown("<div class='fade-in-section'>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])
st.markdown("</div>", unsafe_allow_html=True)

# Prediction logic
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    resized = image.resize((224, 224))
    array = np.expand_dims(preprocess_input(np.array(resized)), axis=0)

    with st.spinner("Analyzing image..."):
        pred = model.predict(array)[0][0]
        label = "Real" if pred > 0.5 else "Fake"
        confidence = pred if pred > 0.5 else 1 - pred

    box_class = "prediction-box prediction-real" if label == "Real" else "prediction-box prediction-fake"

    st.markdown(f"""
    <div class="{box_class}">
        <h3>Prediction: {label}</h3>
        <p>Confidence Score: <strong>{confidence:.2%}</strong></p>
    </div>
    """, unsafe_allow_html=True)
# Experimental Results Section
st.markdown("<hr style='margin-top:4rem; margin-bottom:3rem;'>", unsafe_allow_html=True)
st.markdown("""
<div class="fade-in-section" style="text-align: center;">
    <h2 style="font-size:2.2rem; margin-bottom:10px;">Model Evaluation Summary</h2>
    <p style="color: #555; max-width: 700px; margin: auto;">
        Our deepfake detection model was trained using the EfficientNetB3 architecture. 
        Through extensive experimentation with image augmentation, regularization, and optimizer tuning, we achieved strong predictive performance.
    </p>
</div>
""", unsafe_allow_html=True)

# Side-by-side metrics and curves
col1, col2 = st.columns(2)

with col1:
    st.markdown("""
    <div class="fade-in-section" style="padding: 20px;">
        <h4>Training Configuration</h4>
        <ul style="line-height: 1.8;">
            <li><strong>Architecture:</strong> EfficientNetB3</li>
            <li><strong>Input Size:</strong> 224 × 224</li>
            <li><strong>Batch Size:</strong> 32</li>
            <li><strong>Optimizer:</strong> Adam</li>
            <li><strong>Dropout Rates:</strong> 0.4, 0.3</li>
            <li><strong>Learning Rate:</strong> 0.001</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="fade-in-section" style="padding: 20px;">
        <h4>Final Metrics</h4>
        <ul style="line-height: 1.8;">
            <li><strong>Validation Accuracy:</strong> 83.04%</li>
            <li><strong>ROC AUC Score:</strong> 0.91</li>
            <li><strong>Average Precision:</strong> 0.91</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Accuracy & Loss curves
st.markdown("<div class='fade-in-section' style='margin-top:40px;'>", unsafe_allow_html=True)
chart_col1, chart_col2 = st.columns(2)
with chart_col1:
    st.image("accuracy_curve.png", caption="Validation Accuracy", use_container_width=True)
with chart_col2:
    st.image("loss_curve.png", caption="Training Loss", use_container_width=True)
st.markdown("</div>", unsafe_allow_html=True)

# Footer
st.markdown("---", unsafe_allow_html=True)
st.markdown("""
<style>
.footer {
    position: relative;
    bottom: 0;
    width: 100%;
    text-align: center;
    color: gray;
    font-size: 14px;
    padding: 20px 0 10px 0;
    margin-top: 50px;
}
.footer a {
    color: #3f51b5;
    text-decoration: none;
}
</style>
<div class="footer">
    <p>📅 Completed on: <strong>May 19, 2025</strong></p>
    <p>👥 Created by: Vanessa Lloyd, Vireak Sroeung, George Battikha, Zachary Heffernan, Luke Andriopolous</p>
    <p>🔗 <a href="https://ids-ips-blockchain.streamlit.app/" target="_blank">Live Demo</a> | <a href="https://github.com/NessLloyd/Deepfake" target="_blank">GitHub Repo</a></p>
</div>
""", unsafe_allow_html=True)
