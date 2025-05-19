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
gallery_folder = "gallery"
gallery_images = sorted([
    f for f in os.listdir(gallery_folder)
    if f.endswith((".png", ".jpg", ".jpeg"))
])

# Create HTML for carousel
slides_html = ""
for image_file in gallery_images:
    image_url = f"https://raw.githubusercontent.com/NessLloyd/Deepfake/main/{gallery_folder}/{image_file}"
    slides_html += f"""
    <div class='carousel-item'>
        <img src='{image_url}' />
    </div>
    """

carousel_code = f"""
<style>
.carousel-wrapper {{
    position: relative;
    overflow: hidden;
    width: 100%;
    height: 360px;
    background: #f5f5f5;
    padding: 20px 0;
}}
.carousel-track {{
    display: flex;
    gap: 40px;
    animation: scroll-carousel 60s linear infinite;
}}
.carousel-item {{
    flex: 0 0 auto;
    width: 320px;
    height: 320px;
    transition: transform 0.6s;
    position: relative;
}}
.carousel-item img {{
    width: 100%;
    height: 100%;
    object-fit: cover;
    border-radius: 12px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.2);
}}
.carousel-track .carousel-item:nth-child(7n+4) img {{
    transform: scale(1.15);
    z-index: 2;
    border: 2px solid #444;
}}
@keyframes scroll-carousel {{
  0% {{ transform: translateX(0); }}
  12.5% {{ transform: translateX(-360px); }}
  25% {{ transform: translateX(-720px); }}
  37.5% {{ transform: translateX(-1080px); }}
  50% {{ transform: translateX(-1440px); }}
  62.5% {{ transform: translateX(-1800px); }}
  75% {{ transform: translateX(-2160px); }}
  87.5% {{ transform: translateX(-2520px); }}
  100% {{ transform: translateX(-2880px); }}
}}
</style>
<div class='carousel-wrapper'>
  <div class='carousel-track'>
    {slides_html}
    {slides_html} <!-- duplicate for looping -->
  </div>
</div>
"""

st.components.v1.html(carousel_code, height=400, scrolling=False)

# Model info
st.markdown("---")
st.markdown("####  Model Information")
st.markdown("""
- **Architecture**: EfficientNetB0
- **Training Accuracy**: 83%
- **Validation AUC**: 0.91
- **Dataset**: Custom Celeb-DF subset
""")
