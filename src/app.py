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

    webrtc_streamer(
        key="real-time",
        video_processor_factory=VideoProcessor
    )

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
