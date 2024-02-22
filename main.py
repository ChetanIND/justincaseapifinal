
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

credentials_firebase = {
    "type": "service_account",
    "project_id": "hackmatrix-ba8fb",
    "private_key_id": "c3bfdfb17f35146f36f52a0e105058a5c4f43c69",
    "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvAIBADANBgkqhkiG9w0BAQEFAASCBKYwggSiAgEAAoIBAQCrfl9CewDW42pC\nr9+J26ND/HpNy4/F5B1umJu/o+8di+N7k+q1GN/4gYF+cfR1bRnO2VWM9j3Fc0WS\nZ35C6SGV2xdyjjAco4ioopV6C5qYcZwAyZP/p0iRjb6LHgDMCnPdCpGSF8XFATQx\njJ1vG9vu7fFbrrVqVgHzTnUp6bxCsMIvi2fDGZnrRdPCC+ufow4KJgbTQtDqWKyD\nDdrTQzbE2SjRUjOnLDTxABZfDcGtqM5AjuQ68w9Cguu/aPDKi9gY22ITNreX29g5\n2WZfUrHYOw25F6V6XrFIOuDf2/3jEXvy5cIgzqGj+NQ87SaQwm/XyEsv7hpTZMW6\nwqnQjGZdAgMBAAECggEABtwbuF+ONYpMPlWlpfCMs9P+Gm2JlztcOAfPtxc6Id7u\nHwvYKB1SHDEl+mZZhnbyQNVtuCFDcEn/nu8X3FpR9xoX8oOghgy+kyxJuWOMcAuf\nV2K40lGhM/1NAWiWVJMYdl+NWiAsT4iQS2kaBQ9CuWh4Lpgq9pFxK4fYZPBOEQbL\nQcPybGENy38+Nf9DrgfjUS1y8bbEI5XBozJ8RVC1jzX8Lef0njtJhe2mZjNQMvFp\nHc1FoTFAMjGi8+apO266ja4YYjjzQwSsemLD0pvllyQHO+xOy3XKhPNF+Tg3DRs3\nAXYoutETR4FwHAPTolzo3P6fgjyBYvqDyKfW3tlvkQKBgQDVVxVvTpJPrnIInKRL\nB9F7KLPmRNZV+JuA0WiGfej2OPrzHz9KI4GSiA6DBeBpDbZduEy7jNmTxlGS4SN3\nar09R4HM3pTZLetYTfa0SxjMTAMmbzKBmyKC/M7Vv3P17+bmTsXLSMrjZIRYH+8e\nRwEfpKTbvOFeArZJGG8jyEowZQKBgQDNySh2sKMBuXqdeCtChf177PFoRG4nnHLJ\nEWIyBwk2J20vmlSuBlY8iScO2LFJXaEWhfl2UYkDnsdUUtvyKcJmPXA7erccLt11\nH/XQwTLlGsgaEtBqjB+FeeB0WXd7lvR5KGGRrTZeDP7ZbR+o3JLbUTeZv1DvV0hp\nrC1jGj7ymQKBgCGUz57xsz4vq2uHnKTi2iqUwZyhgUuPEos4a0egUidP2NCkPoYh\nCKhUGlStfCGNMwOVmx56kVUdhoGkRrzpZFhdBSWGc8+r1rvTqd2/ZGvkGyrVnhGg\npdIQkU48ELjJxoLCK4hQMP+SNvLYM/+EFb0xYXHlTWRK8P6YhgYP5P2xAoGAQdu0\n9XdGU9D2atsAjUOwgi6se8AauNaa7bqAgJ471nb7vJZZr3AbvTfvphK3elFasoih\n87nYba4tANGbzn6K1omnF4IIhB6DhW57DxolnajajW2kAdViaSc+LD5NvOHsz7Ga\nuDKFCciC7za7QSGGZmYxsyTFVDPM2vTdea/2oVECgYBVfhStujfj/ZJyVz2WHLm1\nmAeMrBLShi7iaAXHwkMdzrkq4tGnLwkMAWbzTur/9QC6exG+bm92TVqUmyK9qlkY\nGusPu94aqemac6uIEf/4z26y3sgPwFJ34fuihsBGeTOwfjCfZWzNuXtMj/f/QO4d\nPwXtGHhRIFQOr7TzXD19Ig==\n-----END PRIVATE KEY-----\n",
    "client_email": "firebase-adminsdk-aew9a@hackmatrix-ba8fb.iam.gserviceaccount.com",
    "client_id": "113847620523291284487",
    "auth_uri": "https://accounts.google.com/o/oauth2/auth",
    "token_uri": "https://oauth2.googleapis.com/token",
    "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
    "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-aew9a%40hackmatrix-ba8fb.iam.gserviceaccount.com",
    "universe_domain": "googleapis.com"
}

# Initialize Firebase Admin SDK
cred = credentials.Certificate()
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

'''
if __name__ == "__main__":
    port = int(getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)'''




