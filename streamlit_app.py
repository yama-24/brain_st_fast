# streamlit_app.py

import streamlit as st
import requests
from PIL import Image

# FASTAPI_URL = "http://localhost:8000/predict/" # local
FASTAPI_URL = "https://brain-st-fast.onrender.com/predict/" # Render

st.title("脳画像分類アプリ")

uploaded_file = st.file_uploader("画像をアップロードしてください", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロードされた画像", use_column_width=True)

    # 予測の実行
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(FASTAPI_URL, files=files)

    # デバッグ用
    print("Status code:", response.status_code)
    print("Response content:\n", response.text)

    predicted_class_idx = response.json()["predicted_class_idx"]

    # 予測結果の表示
    st.write(f"予測されたクラスのインデックスは {predicted_class_idx} です")
