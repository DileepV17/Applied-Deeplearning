import os
import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Inference UI", layout="centered")
st.title("Image classification (Streamlit → FastAPI)")

API_URL = os.getenv("API_URL")  # e.g. https://clip-backend-xxxxx.europe-west3.run.app
TOPK = int(os.getenv("TOPK", "1"))

if not API_URL:
    st.error("Missing API_URL env var. Set it to your backend Cloud Run URL.")
    st.stop()

uploaded = st.file_uploader("Upload image", type=["png", "jpg", "jpeg", "webp"])

if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Input image", use_column_width=True)

    if st.button("Predict"):
        with st.spinner("Calling backend..."):
            files = {"file": (uploaded.name, uploaded.getvalue(), uploaded.type)}
            r = requests.post(f"{API_URL}/predict", files=files, timeout=120)

        if r.status_code != 200:
            st.error(f"Backend error {r.status_code}: {r.text}")
            st.stop()

        data = r.json()
        st.subheader("Top predictions")
        for item in data.get("topk", [])[:TOPK]:
            st.write(f"**{item['label']}** — {item['prob']*100:.2f}%")