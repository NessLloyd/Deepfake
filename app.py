import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import os
import streamlit.components.v1 as components

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

# Define image URLs
gallery_urls = [
    f"https://raw.githubusercontent.com/NessLloyd/Deepfake/main/gallery/gallery-{i}.png" for i in range(1, 13)
]

# HTML with animation and center-scale effect
gallery_html = f"""
<style>
.carousel-wrapper {{
  width: 100%;
  overflow: hidden;
  background: #fafafa;
  padding: 20px 0;
  position: relative;
}}

.carousel-container {{
  display: flex;
  animation: scroll 60s linear infinite;
  gap: 30px;
  padding-left: 100%;
}}

.carousel-container:hover {{
  animation-play-state: paused;
}}

.carousel-container img {{
  height: 250px;
  border-radius: 12px;
  transition: transform 0.3s;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}}

.carousel-container div {{
  flex: 0 0 auto;
  display: flex;
  align-items: center;
  justify-content: center;
}}

.carousel-container div:nth-child(6) img {{
  transform: scale(1.1);
  border: 3px solid #b06666;
}}

@keyframes scroll {{
  0% {{ transform: translateX(0); }}
  100% {{ transform: translateX(-100%); }}
}}

.dots {{
  text-align: center;
  margin-top: 10px;
}}
.dots span {{
  display: inline-block;
  height: 12px;
  width: 12px;
  background: #ccc;
  border-radius: 50%;
  margin: 0 5px;
  animation: blink 60s linear infinite;
}}
.dots span:nth-child(6) {{ background: #b06666; }}
@keyframes blink {{
  0%, 100% {{ opacity: 1; }}
  50% {{ opacity: 0.5; }}
}}
</style>
<div class="carousel-wrapper">
  <div class="carousel-container">
    {''.join([f'<div><img src="{url}" /></div>' for url in gallery_urls * 2])}
  </div>
  <div class="dots">{''.join(['<span></span>' for _ in gallery_urls])}</div>
</div>
"""

components.html(gallery_html, height=350, scrolling=False)

# Model info
st.markdown("---")
st.markdown("#### 🧬 Model Information")
st.markdown("""
- **Architecture**: EfficientNetB0
- **Training Accuracy**: 83%
- **Validation AUC**: 0.91
- **Dataset**: Custom Celeb-DF subset
""")
