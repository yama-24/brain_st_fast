# streamlit_app.py

import streamlit as st
import requests
from PIL import Image

# FASTAPI_URL = "http://localhost:8000/predict/" # local
FASTAPI_URL = "https://brain-st-fast.onrender.com/predict/" # Render

# 推論したラベルから脳の病名を返す関数
def getName(label):
    if label==0:
        return '神経膠腫（glioma）'
    elif label==1:
        return '髄膜腫（meningioma）'
    elif label==2:
        return '腫瘍なし（notumor）'
    elif label==3:
        return '下垂体腫瘍（pituitary）'

# サイドバー（入力画面）
st.sidebar.title("脳画像から腫瘍を分類")
uploaded_file = st.sidebar.file_uploader("脳画像をアップロードしてください", type=["jpg", "jpeg", "png"])

# 画面表示
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロードされた画像", use_column_width="auto")

    # 予測の実行
    files = {"file": uploaded_file.getvalue()}
    response = requests.post(FASTAPI_URL, files=files)

    # デバッグ用
    # print("Status code:", response.status_code)
    # print("Response content:\n", response.text)

    predicted_class_idx = response.json()["predicted_class_idx"]

    # 予測結果の表示
    st.write(f"予測される腫瘍は {getName(predicted_class_idx)} です")