import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import shutil
from uuid import uuid4
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from model import load_model, predict
from grad_cam import generate_grad_cam

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
model = load_model()

UPLOAD_FOLDER = Path("static/images/original")
GRAD_CAM_FOLDER = Path("static/images/grad_cam")
UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
GRAD_CAM_FOLDER.mkdir(parents=True, exist_ok=True)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(('.jpg', '.jpeg', '.png')):
            raise HTTPException(status_code=400, detail="Invalid file format. Only JPG, JPEG, and PNG are supported.")

        file_id = str(uuid4())
        file_path = UPLOAD_FOLDER / f"{file_id}.png"
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        print(f"File saved to: {file_path}")  # Debug print

        prediction, probability = predict(model, file_path)
        print(f"Prediction: {prediction}, Probability: {probability}")  # Debug print

        grad_cam_path = GRAD_CAM_FOLDER / f"{file_id}_grad_cam.png"
        generate_grad_cam(model, file_path, grad_cam_path)

        return JSONResponse(content={
            "success": True,
            "original_image_url": f"/static/images/original/{file_id}.png",
            "grad_cam_image_url": f"/static/images/grad_cam/{file_id}_grad_cam.png",
            "prediction": prediction,
            "probability": probability
        })
    except Exception as e:
        print(str(e))
        raise HTTPException(status_code=500, detail=str(e))
