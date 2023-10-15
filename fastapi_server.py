# fastapi_server.py

from fastapi import FastAPI, UploadFile, File
import torch
from PIL import Image
import io
from model import transform, Net # model.py から前処理とネットワークの定義を読み込み

# FastAPI のインスタンス化
app = FastAPI()

# ネットワークの準備
net = Net().cpu().eval()
# 学習済みモデルの重み（brain.pt）を読み込み
net.load_state_dict(torch.load('brain.pt', map_location=torch.device('cpu')))

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert('RGB')

    #　データの前処理
    img = transform(image)
    img =img.unsqueeze(0) # 1次元増やす

    #　推論
    predicted_class_idx = int(torch.argmax(net(img), dim=1).cpu().item())
    
    return {"predicted_class_idx": predicted_class_idx}