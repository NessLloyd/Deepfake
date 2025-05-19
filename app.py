import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import os

# Set page config
st.set_page_config(page_title="Deepfake Detection", layout="wide")

# Load model
@st.cache_resource
def load_best_model():
    return load_model("best_model.keras")

model = load_best_model()

# Layout: Title
st.title("AI or Real - Deepfake Detection")
st.markdown("Upload a **face image**, and we’ll tell you if it's a deepfake or not.")

# File upload
uploaded_file = st.file_uploader("📄 Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    col1, col2 = st.columns(2)

    image = Image.open(uploaded_file).convert("RGB")
    col1.image(image, caption="Uploaded Image", use_container_width=True)

    resized = image.resize((224, 224))
    array = np.expand_dims(preprocess_input(np.array(resized)), axis=0)

    pred = model.predict(array)[0][0]
    label = "Real" if pred > 0.5 else "Fake"
    confidence = pred if pred > 0.5 else 1 - pred

    col2.markdown(f"###  **Prediction:** `{label}`")
    col2.markdown(f"**Confidence:** `{confidence:.2%}`")

import streamlit.components.v1 as components

# Section title
st.markdown("---")
st.markdown("<h2 style='text-align:center;'>Prediction Results</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#b06666;'>Sample prediction results produced by our deepfake detection model</p>", unsafe_allow_html=True)

# Define gallery image URLs
image_urls = [
    f"https://raw.githubusercontent.com/NessLloyd/Deepfake/main/gallery/gallery-{i}.png"
    for i in range(1, 13)
]

# Create horizontal scrolling HTML
gallery_html = """
<div style="display: flex; overflow-x: auto; scroll-behavior: smooth; gap: 20px; padding: 20px; background: #fafafa;">
"""
for url in image_urls:
    gallery_html += f"""
    <div style="flex: 0 0 auto;">
        <img src="{url}" style="height:300px; border-radius:10px;" />
    </div>
    """
gallery_html += "</div>"

# Render it using components.html
components.html(gallery_html, height=340)


# Section: Model Info
st.markdown("---")
st.markdown("#### 🧬 Model Information")
st.markdown("""
- **Architecture**: EfficientNetB0  
- **Training Accuracy**: 83%  
- **Validation AUC**: 0.91  
- **Dataset**: Custom Celeb-DF subset
""")
