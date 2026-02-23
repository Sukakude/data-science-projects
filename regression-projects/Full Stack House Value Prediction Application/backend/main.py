from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from app.model.house_model import House
import joblib
import pathlib
import pandas as pd
import numpy as np
from datetime import datetime
import json

api = FastAPI()

@api.get("/")
def model_info():
    if pathlib.Path('model.pkl').exists():
        json_data = jsonable_encoder({
            "status":"Healthy",
            "version":"v1.0",
            "accuracy": 74,
            "last_train_date": datetime(2026, 2, 14)
        })
        return JSONResponse(content=json_data)
    else:
        json_data = jsonable_encoder({
            "status":"Unhealthy",
            "version":"v1.0",
            "accuracy": 0,
            "last_train_date": datetime(2026, 2, 14)
        })
        return JSONResponse(content=json_data)

@api.post("/predict/")
async def predict_house_value(house: House):
    try:
        # LOAD THE MODEL TO PREDICT
        model = joblib.load('model.pkl')

        # LOAD THE SCALER OBJECT
        scaler = joblib.load('scaler.pkl')

        # LOAD THE ENCODER OBJECT
        encoder = joblib.load('encoder.pkl')

        # GET THE INDEPENDENT VARIABLES TO USE FOR PREDICTION      
        X_pred = pd.DataFrame(
            data=[house.model_dump()]
        )

        # ENCODE THE CATEGORICAL DATA
        X_pred['ocean_proximity'] = encoder.transform(
            X_pred[['ocean_proximity']]
        )

        # PREPARE DATA FOR THE MODEL
        X_scaled = scaler.transform(X_pred)

        # GENERATE PREDICTIONS
        pred = model.predict(X_scaled)

        json_data = jsonable_encoder({
            "message": "Success", 
            "house_value": round(float(pred[0]), 2)
        })

        return JSONResponse(content=json_data)
    except Exception as e:
        print(f'Error: {e}')
        raise e
        json_data = jsonable_encoder({"message": f"{e}", "data" : None})
        return JSONResponse(content=json_data) 

