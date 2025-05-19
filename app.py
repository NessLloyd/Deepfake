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
    from streamlit.components.v1 import html

    st.markdown("## 🎯 Prediction Results", unsafe_allow_html=True)
    st.markdown("#### <span style='color:#ba6b6c;'>Sample prediction results produced by our deepfake detection model</span>", unsafe_allow_html=True)
    
    # Load image file paths from gallery
    image_folder = "gallery"
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))])
    
    # Generate HTML content
        carousel_images = "".join(
            f"<div class='slide'><img src='gallery/{img}' alt='{img}' /></div>" for img in image_files
        )
        
        carousel_html = f"""
        <style>
        .carousel {{
          display: flex;
          overflow: hidden;
          width: 100%;
          height: 540px;
          position: relative;
          margin-top: 20px;
        }}
        
        .slide {{
          min-width: 100%;
          transition: transform 1s ease-in-out;
        }}
        
        .carousel-container {{
          display: flex;
          animation: scroll 30s linear infinite;
        }}
        
        .carousel img {{
          width: 100%;
          object-fit: contain;
          border-radius: 12px;
        }}
        
        @keyframes scroll {{
          0% {{ transform: translateX(0%); }}
          20% {{ transform: translateX(0%); }}
          25% {{ transform: translateX(-100%); }}
          45% {{ transform: translateX(-100%); }}
          50% {{ transform: translateX(-200%); }}
          70% {{ transform: translateX(-200%); }}
          75% {{ transform: translateX(-300%); }}
          95% {{ transform: translateX(-300%); }}
          100% {{ transform: translateX(-400%); }}
        }}
        </style>
        
        <div class="carousel">
          <div class="carousel-container">
            {carousel_images}
          </div>
        </div>
        """
        
        html(carousel_html, height=540)



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
