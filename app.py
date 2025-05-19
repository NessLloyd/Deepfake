import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import os

# Set page config
st.set_page_config(page_title="Deepfake Detection", layout="wide")

# Inject light theme and CSS for cleaner appearance
st.markdown("""
    <style>
    body {
        background-color: white;
        color: black;
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stFileUploader, .stButton>button {
        background-color: #f5f5f5;
        color: black;
    }
    .title-style {
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        color: #222;
        margin-bottom: 0.2em;
    }
    .subheader-style {
        text-align: center;
        font-size: 1.1em;
        color: #555;
        margin-bottom: 2em;
    }
    </style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_best_model():
    return load_model("best_model.keras")

model = load_best_model()

# Header Section
st.markdown("<div class='title-style'>AI or Real - Deepfake Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subheader-style'>Upload a face image, and we’ll tell you if it's a deepfake or not.</div>", unsafe_allow_html=True)


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

# Create HTML for Swiper carousel
slides_html = ""
for image_file in gallery_images:
    image_url = f"https://raw.githubusercontent.com/NessLloyd/Deepfake/main/{gallery_folder}/{image_file}"
    slides_html += f"<div class='swiper-slide'><img src='{image_url}'/></div>"

carousel_code = f"""
<link
  rel="stylesheet"
  href="https://cdn.jsdelivr.net/npm/swiper@11/swiper-bundle.min.css"
/>
<script src="https://cdn.jsdelivr.net/npm/swiper@11/swiper-bundle.min.js"></script>

<style>
.swiper-container {{
  width: 100%;
  padding: 40px 0;
}}
.swiper-slide {{
  width: 280px;
  height: 360px;
  border-radius: 12px;
  overflow: hidden;
  transition: transform 0.5s ease;
}}
.swiper-slide img {{
  width: 100%;
  height: 100%;
  object-fit: cover;
  border-radius: 12px;
}}
.swiper-slide-active {{
  transform: scale(1.1);
  box-shadow: 0 8px 20px rgba(0, 0, 0, 0.3);
}}
.swiper-pagination-bullet {{
  background: #bbb;
  opacity: 1;
}}
.swiper-pagination-bullet-active {{
  background: #3f51b5;
}}
</style>

<div class="swiper-container">
  <div class="swiper-wrapper">
    {slides_html}
  </div>
  <div class="swiper-pagination"></div>
</div>

<script>
  const swiper = new Swiper('.swiper-container', {{
    slidesPerView: 6,
    spaceBetween: 20,
    loop: true,
    centeredSlides: true,
    autoplay: {{
      delay: 2500,
      disableOnInteraction: false,
    }},
    pagination: {{
      el: '.swiper-pagination',
      clickable: true,
    }},
  }});
</script>
"""

st.components.v1.html(carousel_code, height=460, scrolling=False)

# Experimental Results Section
st.markdown("---")
st.markdown("<h2 style='text-align:center;'>🔬 Experimental Results</h2>", unsafe_allow_html=True)
st.markdown("""
<p style='text-align:center; max-width:900px; margin:auto;'>
We have performed extensive training and hyperparameter tuning, such as comparing different EfficientNet models, number of convolution layers, weights, data augmentations, dropout rates, and regularizers. In the end, the following settings gave us the best results:
</p>
""", unsafe_allow_html=True)

# Hyperparameter Summary
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    - ✅ Input Size: **224 × 224**
    - ✅ Batch Size: **32**
    - ✅ Optimizer: **Adam**
    """)

with col2:
    st.markdown("""
    - ✅ Learning Rate: **0.001**
    - ✅ Dropout Rates: **0.4** and **0.3**
    """)

# Accuracy and Loss Charts
col1, col2 = st.columns(2)
with col1:
    st.image("accuracy_curve.png", caption="Accuracy Curve", use_container_width=True)
with col2:
    st.image("loss_curve.png", caption="Loss Curve", use_container_width=True)

# Final Metrics
st.markdown("### 📊 Final Performance Metrics")
st.markdown("""
- ✅ Final Validation Accuracy: **83.04%**
- ✅ ROC AUC Score: **0.91**
- ✅ Average Precision: **0.91**
""")

# Footer
st.markdown("---", unsafe_allow_html=True)
st.markdown(
    """
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
    </style>
    <div class="footer">
        <p>📅 Completed on: <strong>May 19, 2025</strong></p>
        <p>👥 Created by: Vanessa Lloyd, Vireak Sroeung, George Battikha, Zachary Heffernan, Luke Andripolous</p>
    </div>
    """,
    unsafe_allow_html=True
)


