
# user_id = "6bqUd7Vh93YM0GUF6efGoCbsjBv2"

import uvicorn
from os import getenv
from pydantic import BaseModel
from typing import List
from fastapi import FastAPI, HTTPException, Path
from starlette.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX
import firebase_admin
from firebase_admin import credentials, firestore

# Load the saved model
loaded_model = joblib.load("sarima_model.pkl")

# Initialize Firebase Admin SDK
cred = credentials.Certificate("firebase_credentials.json")
firebase_admin.initialize_app(cred)
db = firestore.client()

# Define the collection name
collection_name = "Payments"

# Create the FastAPI instance
app = FastAPI()

# Define CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

class UserInput(BaseModel):
    user_id: str

class PredictionResult(BaseModel):
    forecast: List[float]

@app.get("/")
def read_root():
    return {"message": "Welcome to the Expense Forecasting API!"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/predict/{user_id}", response_model=PredictionResult)
async def predict(user_id: str = Path(..., title="The ID of the user to predict expenses for")):
    try:
        # Fetch input data from Firebase
        emp_ref = db.collection(collection_name).where('UserId', '==', user_id)
        docsm = emp_ref.stream()

        df = 0
        for doc in docsm:
            # Access document data using the document reference
            doc_data = doc.to_dict()
            daily_expense = doc_data.get('Daily Expense', {})
            for date, amount in daily_expense.items():
                df += 1

        forecast_steps = df
        
        # Generate forecasts for the next `forecast_steps` days
        forecast = loaded_model.forecast(steps=forecast_steps)
        forecast_data = forecast.tolist()
        
        return {"forecast": forecast_data}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(getenv("PORT", 8000))
    uvicorn.run("main:app", host="127.0.0.1", port=port, reload=True)




