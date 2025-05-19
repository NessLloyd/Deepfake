import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input

# Load model
@st.cache_resource
def load_best_model():
    return load_model("best_model.keras")

model = load_best_model()

# Title and instructions
st.set_page_config(page_title="Fake or Face", layout="centered")
st.title("🔍 Fake or Face")
st.subheader("Upload a face image to detect if it's a deepfake.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    resized = image.resize((224, 224))
    array = np.array(resized)
    array = preprocess_input(array)
    array = np.expand_dims(array, axis=0)

    # Predict
    pred = model.predict(array)[0][0]
    label = "Real" if pred > 0.5 else "Fake"
    confidence = pred if pred > 0.5 else 1 - pred

    st.markdown(f"### 🧠 Prediction: `{label}`")
    st.markdown(f"**Confidence:** `{confidence:.2%}`")