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

# -------------------------
# TAB 3 — REAL-TIME VIDEO
# -------------------------
with tab3:

    st.write("Live webcam classification")

    class VideoProcessor(VideoProcessorBase):

        def recv(self, frame):

            img = frame.to_ndarray(format="bgr24")

            pil_img = Image.fromarray(
                cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            )

            label, confidence = predict_image(pil_img, model)

            cv2.putText(
                img,
                f"{label} ({confidence:.2f})",
                (10,30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

            return frame.from_ndarray(img, format="bgr24")

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
