import streamlit as st
from PIL import Image
import cv2

# 👇 import NEW functions
from predict import load_model_from_hf, predict_image

# ==========================
# LOAD MODEL ONLY ONCE
# ==========================

@st.cache_resource
def load_model():
    model = load_model_from_hf()
    return model

model = load_model()

# ==========================
# PAGE SETUP
# ==========================

st.set_page_config(page_title="Garbage Classification AI", layout="centered")

st.title("♻️ Garbage Classification AI")

tab1, tab2, tab3 = st.tabs([
    "📂 Upload Image",
    "📷 Camera Capture",
    "🎥 Real-Time Video"
])

# -------------------------
# TAB 1 — Upload Image
# -------------------------
with tab1:

    uploaded_file = st.file_uploader(
        "Upload Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(image, caption="Uploaded Image")

        label, confidence = predict_image(image, model)

        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {confidence:.4f}")

# -------------------------
# TAB 2 — Camera Capture
# -------------------------
with tab2:

    camera_image = st.camera_input("Take a photo")

    if camera_image is not None:

        image = Image.open(camera_image).convert("RGB")

        st.image(image, caption="Captured Image")

        label, confidence = predict_image(image, model)

        st.success(f"Prediction: {label}")
        st.info(f"Confidence: {confidence:.4f}")

# ==========================
# FOOTER
# ==========================


st.markdown("---")
st.markdown(
    """
    <div style="text-align:center;
                font-size:14px;
                color:gray;">
        ♻️ Garbage Classification AI — Implemented by <b>SU YEE</b>
    </div>
    """,
    unsafe_allow_html=True
)
