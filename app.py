import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
import os
import base64

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

# Gallery
st.markdown("---")
st.markdown("<h2 style='text-align: center;'>Prediction Results</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Sample prediction results produced by our deepfake detection model</p>", unsafe_allow_html=True)

image_folder = "gallery"
image_files = sorted([img for img in os.listdir(image_folder) if img.endswith((".png", ".jpg", ".jpeg"))])

# Encode gallery images to base64
carousel_items = ""
for img_file in image_files:
    full_path = os.path.join(image_folder, img_file)
    with open(full_path, "rb") as file:
        encoded = base64.b64encode(file.read()).decode()
        carousel_items += f"<div class='gallery-slide'><img src='data:image/png;base64,{encoded}'></div>"

carousel_html = f"""
<style>
.gallery-wrapper {{
  overflow: hidden;
  width: 100%;
  height: 300px;
  background: #f5f5f5;
  padding: 10px 0;
  display: flex;
  align-items: center;
  justify-content: center;
}}
.gallery-container {{
  display: flex;
  gap: 20px;
  animation: scroll-x 40s linear infinite;
}}
.gallery-slide img {{
  height: 280px;
  border-radius: 10px;
  object-fit: cover;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
}}

@keyframes scroll-x {{
  0% {{ transform: translateX(0); }}
  100% {{ transform: translateX(-50%); }}
}}
</style>

<div class="gallery-wrapper">
  <div class="gallery-container">
    {carousel_items}
    {carousel_items} <!-- Duplicate for infinite loop -->
  </div>
</div>
"""

st.components.v1.html(carousel_html, height=350, scrolling=False)

# Model info
st.markdown("---")
st.markdown("#### 🧬 Model Information")
st.markdown("""
- **Architecture**: EfficientNetB0
- **Training Accuracy**: 83%
- **Validation AUC**: 0.91
- **Dataset**: Custom Celeb-DF subset
""")
