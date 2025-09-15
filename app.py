from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import warnings

# Suppress warning about feature names
warnings.filterwarnings("ignore", message="X does not have valid feature names")

app = FastAPI()
model = joblib.load("model.pkl")

# Define input schema
class WeatherFeature(BaseModel):
    Temperature: float
    Humidity: float
    Wind_Speed: float
    Cloud_Cover: float
    Pressure: float

@app.post("/predict")
async def predict(features: WeatherFeature):
    try:
        
        features_list = [
            features.Temperature,
            features.Humidity,
            features.Wind_Speed,
            features.Cloud_Cover,
            features.Pressure
        ]
        print("Features list:", features_list)

        prediction = model.predict([features_list])[0]
        print("Prediction result:", prediction)

        return {"prediction": int(prediction)}

    except Exception as e:
        print("Error during prediction:", e)
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/hello")
async def hello():
    return {"message": "Hello World!"}
