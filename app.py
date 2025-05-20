import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import base64
import io 
import os 

# Set page configuration
st.set_page_config(page_title="Deepfake Detection", layout="wide")

# Inject custom CSS
st.markdown("""
<style>
/* Global */
html, body, .stApp {
    background-color: #f5f7fa;
    color: #111;
    font-family: 'Segoe UI', sans-serif;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    padding-left: 3vw;
    padding-right: 3vw;
    max-width: 100vw;
    box-sizing: border-box;
}

/* Title */
.title-style {
    font-size: 2.8rem;
    font-weight: 700;
    text-align: center;
    color: #222;
    margin-bottom: 1rem;
    opacity: 0;
    animation: fadeIn 1s ease-out forwards;
}

/* Upload Section */
.stFileUploader {
    border: 2px dashed #999 !important;
    background-color: #fff;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}

/* Prediction Box */
.prediction-box {
    background-color: #ffe5e5;
    border-left: 8px solid #cc0000;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 1.5rem auto;
    max-width: 600px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.05);
    animation: fadeInUp 0.8s ease-out forwards;
    opacity: 0;
}

/* Prediction Title */
.prediction-title {
    font-size: 1.8rem;
    font-weight: bold;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: #cc0000;
}

/* Confidence Text */
.confidence-text {
    font-size: 1rem;
    color: #444;
    margin-top: 0.4rem;
}

/* Image */
img {
    border-radius: 10px;
    max-width: 200px;
    margin-top: 1rem;
    box-shadow: 0 3px 10px rgba(0,0,0,0.1);
}

/* Animations */
@keyframes fadeIn {
    to { opacity: 1; }
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(30px); }
    to { opacity: 1; transform: translateY(0); }
}
.hero {
    text-align: center;
    margin: 30px 0 40px 0;
}

.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    color: #222;
    margin-bottom: 10px;
}

.hero p {
    font-size: 1.2rem;
    color: #555;
}

.footer {
    text-align: center;
    color: gray;
    font-size: 14px;
    padding: 30px 0 10px 0;
    margin-top: 50px;
}
.footer a {
    color: #3f51b5;
    text-decoration: none;
}

.stats-section {
    background-color: #f0f2f5;
    padding: 60px 30px;
    border-radius: 12px;
    margin-top: 40px;
    text-align: center;
}

.stats-section h2 {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.stats-section .subtext {
    font-size: 1.1rem;
    color: #666;
    margin-bottom: 30px;
}

.stats-grid {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
    gap: 30px;
    max-width: 1100px;
    margin: 0 auto;
}

.stat-box {
    background-color: white;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    padding: 30px 20px;
    width: 200px;
    transition: transform 0.3s ease;
}
.stat-box:hover {
    transform: translateY(-5px);
}

.stat-number {
    font-size: 1.8rem;
    font-weight: 700;
    color: #222;
    margin-bottom: 10px;
}

.stat-label {
    font-size: 1rem;
    color: #555;
}


.prediction-container {
    display: flex;
    justify-content: center;
    align-items: center;
    gap: 40px;
    flex-wrap: wrap;
    margin-top: 30px;
}

.pred-image img {
    border-radius: 12px;
    max-width: 260px;
    max-height: 360px;
    object-fit: contain;
    box-shadow: 0 3px 12px rgba(0,0,0,0.1);
}

.pred-box {
    flex: 1;
    min-width: 280px;
}

</style>
""", unsafe_allow_html=True)
import streamlit.components.v1 as components

components.html("""
<script>
const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add("fade-in-visible");
        }
    });
});
window.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.fade-section').forEach(el => observer.observe(el));
});
</script>
""", height=0)

# Hero Section
st.markdown("""
<div class='hero'>
    <h1>AI or Real?</h1>
    <p>Upload a face image and let our AI detect whether it's fake or real.</p>
</div>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_best_model():
    return load_model("best_model.keras")

model = load_best_model()


# Upload
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    resized = image.resize((224, 224))
    array = np.expand_dims(preprocess_input(np.array(resized)), axis=0)

    # Predict
    with st.spinner("Analyzing the image..."):
        pred = model.predict(array)[0][0]
        is_fake = pred < 0.5
        label = "Fake" if is_fake else "Real"
        confidence = (1 - pred) if is_fake else pred
        confidence_pct = round(confidence * 100, 2)
        icon = "❌" if is_fake else "✅"

    # Display image + prediction in custom flex layout
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()

    st.markdown(f"""
    <div class="prediction-container">
        <div class="pred-image">
            <img src="data:image/png;base64,{img_str}" />
        </div>
        <div class="pred-box">
            <div class="prediction-box">
                <div class="prediction-title">
                    <span>{icon} Prediction: {label}</span>
                </div>
                <div class="confidence-text">Confidence: <strong>{confidence_pct}%</strong></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)



# Gallery section
st.markdown("---")
st.markdown("<h2 style='text-align:center;'>Prediction Results</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#b06666;'>Sample predictions generated by the model</p>", unsafe_allow_html=True)

gallery_folder = "gallery"
gallery_images = sorted([
    f for f in os.listdir(gallery_folder)
    if f.endswith((".png", ".jpg", ".jpeg"))
])

slides_html = ""
for image_file in gallery_images:
    image_url = f"https://raw.githubusercontent.com/NessLloyd/Deepfake/main/{gallery_folder}/{image_file}"
    slides_html += f"<div class='swiper-slide'><img src='{image_url}'/></div>"

carousel_code = f"""
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swiper@11/swiper-bundle.min.css"/>
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
st.markdown("""
<div class="stats-section">
    <h2>Experimental Results</h2>
    <p class="subtext">Final model configuration from testing various EfficientNet settings.</p>
    <div class="stats-grid">
        <div class="stat-box">
            <div class="stat-number">224×224</div>
            <div class="stat-label">Input Size</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">32</div>
            <div class="stat-label">Batch Size</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">0.001</div>
            <div class="stat-label">Learning Rate</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">Adam</div>
            <div class="stat-label">Optimizer</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">0.4 & 0.3</div>
            <div class="stat-label">Dropout</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">83.04%</div>
            <div class="stat-label">Val Accuracy</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">0.91</div>
            <div class="stat-label">ROC AUC</div>
        </div>
        <div class="stat-box">
            <div class="stat-number">0.91</div>
            <div class="stat-label">Avg Precision</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# Chart Section (keep this below the stats)
st.markdown("<h3 style='text-align:center; margin-top:40px;'>Training Performance</h3>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    st.image("accuracy_curve.png", caption="Accuracy Curve", use_container_width=True)
with col2:
    st.image("loss_curve.png", caption="Loss Curve", use_container_width=True)

# Footer
st.markdown("""
<div class="footer">
    <p>Completed on: <strong>May 19, 2025</strong></p>
    <p>Created by: Vanessa Lloyd, Vireak Sroeung, George Battikha, Zachary Heffernan, Luke Andriopolous</p>
    <p>
       <a href="https://github.com/NessLloyd/Deepfake" target="_blank">GitHub Repo</a></p>
</div>
""", unsafe_allow_html=True)
