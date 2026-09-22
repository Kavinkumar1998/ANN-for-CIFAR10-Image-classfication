from __future__ import annotations

import io
from typing import Any

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from PIL import Image

from src.models.predict import Predictor

app = FastAPI(title="CIFAR-10 Image Classifier", version="1.0.0")

try:
    predictor = Predictor()
except FileNotFoundError:
    predictor = None


@app.get("/", response_class=HTMLResponse)
async def home() -> str:
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0" />
        <title>CIFAR-10 Classifier</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                background: #f3f6fb;
                margin: 0;
                padding: 40px 20px;
                display: flex;
                justify-content: center;
            }
            .card {
                background: white;
                max-width: 600px;
                width: 100%;
                border-radius: 16px;
                padding: 30px;
                box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
            }
            h1 { margin-top: 0; }
            input[type="file"] { margin: 16px 0; }
            button {
                background: #2563eb;
                color: white;
                border: none;
                padding: 10px 18px;
                border-radius: 10px;
                cursor: pointer;
            }
            .result {
                margin-top: 20px;
                font-size: 1.2rem;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="card">
            <h1>CIFAR-10 Image Classifier</h1>
            <p>Upload an image to predict the class.</p>
            <input id="fileInput" type="file" accept="image/*" />
            <button id="predictBtn">Predict</button>
            <div id="result" class="result">No image selected</div>

            <script>
                const predictBtn = document.getElementById('predictBtn');
                const fileInput = document.getElementById('fileInput');
                const result = document.getElementById('result');

                predictBtn.addEventListener('click', async () => {
                    const file = fileInput.files[0];
                    if (!file) {
                        result.textContent = 'Please choose an image first.';
                        return;
                    }

                    const formData = new FormData();
                    formData.append('image', file);

                    result.textContent = 'Predicting...';

                    try {
                        const response = await fetch('/predict', {
                            method: 'POST',
                            body: formData,
                        });

                        const data = await response.json();
                        if (!response.ok) {
                            throw new Error(data.detail || 'Prediction failed');
                        }

                        result.textContent = 'Prediction: ' + data.prediction;
                    } catch (error) {
                        result.textContent = 'Error: ' + error.message;
                    }
                });
            </script>
        </div>
    </body>
    </html>
    """


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)) -> dict[str, Any]:
    if predictor is None:
        raise HTTPException(500, detail="Model is not available. Please train it first.")

    if not image.content_type or "image" not in image.content_type:
        raise HTTPException(400, detail="Please upload an image file.")

    try:
        contents = await image.read()
        image_pil = Image.open(io.BytesIO(contents)).convert("RGB")
        image_pil = image_pil.resize((32, 32))
        image_array = np.asarray(image_pil, dtype=np.float32)
        prediction = predictor.predict(image_array)
        return {"prediction": prediction}
    except Exception as exc:
        raise HTTPException(400, detail=f"Invalid image: {str(exc)}") from exc


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

