import pandas as pd
import pickle
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os
import joblib

model_path = "artifacts/best_model.pkl"
model = joblib.load(model_path)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

class PredictionRequest(BaseModel):
    features: List[float]


@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        feature_names = [
            "potential_issue", "perf_6_month_avg", "perf_12_month_avg", 
            "local_bo_qty", "ppap_risk", "lead_time", "deck_risk"
        ]
        data = np.array(request.features).reshape(1, -1)
        data_df = pd.DataFrame(data, columns=feature_names)

        prediction = model.predict(data_df)

        prediction_label = "Went on Backorder" if prediction[0] == 1 else "Did not go on Backorder"

        return {"prediction": prediction_label}

    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


@app.get("/", response_class=HTMLResponse)
async def get_home():
    with open("templates/index.html", "r") as file:
        return file.read()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
