import streamlit as st

# ✅ MUST be the first Streamlit command
st.set_page_config(
    page_title="Garbage Classification AI",
    layout="centered",
    page_icon="♻️"
)

from PIL import Image
import cv2

# 👇 import model utilities
from predict import load_model_from_hf, predict_image

# ==========================
# LOAD MODEL ONLY ONCE
# ==========================

@st.cache_resource
def load_model():
    return load_model_from_hf()

model = load_model()

# ==========================
# HEADER
# ==========================

st.title("♻️ Garbage Classification AI")
st.caption("AI-powered garbage classification using deep learning")

# ==========================
# INSTRUCTIONS (VISIBLE)
# ==========================

st.markdown("### 📘 How to Use")

st.markdown("""
This AI model classifies garbage into **6 categories**:

- 🧴 **Plastic**
- 🔩 **Metal**
- 🍾 **Glass**
- 📦 **Cardboard**
- 📄 **Paper**
- 🗑️ **Trash**

**Tips for best accuracy:**
- Use clear, well-lit images
- Avoid blurry or low-quality photos
- Center the garbage object
- One object per image works best
""")

st.markdown("---")

# ==========================
# TABS
# ==========================

tab1, tab2 = st.tabs([
    "📂 Upload Image",
    "📷 Camera Capture",
])

# -------------------------
# TAB 1 — Upload Image
# -------------------------
with tab1:
    uploaded_file = st.file_uploader(
        "Upload an image of garbage",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        label, confidence = predict_image(image, model)

        st.success(f"🧠 Prediction: **{label}**")
        st.info(f"📊 Confidence: **{confidence:.4f}**")

# -------------------------
# TAB 2 — Camera Capture
# -------------------------
with tab2:
    camera_image = st.camera_input("Take a photo of garbage")

    if camera_image is not None:
        image = Image.open(camera_image).convert("RGB")
        st.image(image, caption="Captured Image", use_container_width=True)

        label, confidence = predict_image(image, model)

        st.success(f"🧠 Prediction: **{label}**")
        st.info(f"📊 Confidence: **{confidence:.4f}**")

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
