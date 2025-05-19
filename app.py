import streamlit as st

# 🧠 Set page config FIRST — before anything else
st.set_page_config(page_title="Deepfake Detection", layout="centered")

import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import os

# Load model
@st.cache_resource
def load_best_model():
    return load_model("best_model.keras")

model = load_best_model()

# UI Title
st.title("AI or Real - Deepfake Detection")
st.markdown("Upload a **face image**, and we’ll tell you if it's a deepfake or not.")


# Layout: Upload + Prediction
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
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
    
    # Folder containing images
    image_folder = "gallery"
    image_files = sorted([img for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))])
    
    # Streamlit image slider
    index = st.slider("Prediction Results", 0, len(image_files)-1, 0)
    image_path = os.path.join(image_folder, image_files[index])
    image = Image.open(image_path)
    st.image(image, caption=f"Image: {image_files[index]}", use_container_width=True)

    # Layout: Model Info
    st.markdown("---")
    st.markdown("#### 🧬 Model Information")
    st.markdown("""
    - **Architecture**: EfficientNetB0
    - **Training Accuracy**: 83%
    - **Validation AUC**: 0.91
    - **Dataset**: Custom Celeb-DF subset
    """)


    st.markdown(f"### 🧠 Prediction: `{label}`")
    st.markdown(f"**Confidence:** `{confidence:.2%}`")
