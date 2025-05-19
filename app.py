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

# Gallery Section Title
st.markdown("---")
st.markdown("<h2 style='text-align:center;'>Prediction Results</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#b06666;'>Sample prediction results produced by our deepfake detection model</p>", unsafe_allow_html=True)

# Load gallery images
image_folder = "gallery"
image_files = sorted([
    os.path.join(image_folder, f)
    for f in os.listdir(image_folder)
    if f.endswith((".png", ".jpg", ".jpeg"))
])

# Horizontal scrollable block of images
scroll_block = """
<div style="display: flex; overflow-x: auto; scroll-behavior: smooth; gap: 20px; padding: 20px;">
"""

for path in image_files:
    try:
        scroll_block += f"""
            <div style="flex: 0 0 auto;">
                <img src="https://raw.githubusercontent.com/NessLloyd/Deepfake/main/{path}" style="height:300px; border-radius:10px;" />
            </div>
        """
    except Exception as e:
        st.error(f"Failed to load image: {path} — {e}")

scroll_block += "</div>"

st.markdown(scroll_block, unsafe_allow_html=True)

# Model info
st.markdown("---")
st.markdown("#### 🧬 Model Information")
st.markdown("""
- **Architecture**: EfficientNetB0
- **Training Accuracy**: 83%
- **Validation AUC**: 0.91
- **Dataset**: Custom Celeb-DF subset
""")
