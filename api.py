import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import requests
import uvicorn


class InputData(BaseModel):
    Available_Extra_Rooms_in_Hospital: int
    Department: str
    Ward_Facility_Code: str
    doctor_name: str
    staff_available: int
    Age: str
    gender: str
    Type_of_Admission: str
    Severity_of_Illness: str
    health_conditions: str
    Visitors_with_Patient: int
    Insurance: str
    Admission_Deposit: float

app = FastAPI()

@app.on_event("startup")
def load_model():
    global model
    model_url = "https://github.com/KevKibe/Hospital-LOS-Prediction/releases/download/Model01/LOSmodel.pkl"
    response = requests.get(model_url)
    response.raise_for_status()
    model = pickle.loads(response.content)

@app.post('/predict')
async def make_prediction(input_data: InputData):
    input_dict = input_data.dict()

    input_df = pd.DataFrame([input_dict])

    categorical_features = ['Department', 'Ward_Facility_Code', 'doctor_name', 'Age', 'gender',
                            'Type_of_Admission', 'Severity_of_Illness', 'health_conditions', 'Insurance']
    encoded_features = pd.get_dummies(input_df[categorical_features], drop_first=True)

    processed_features = pd.concat([encoded_features, input_df.drop(categorical_features, axis=1)], axis=1)

    predictions = model.predict(processed_features)

    rounded_predictions = [int(round(prediction)) for prediction in predictions]

    return {"predictions": predictions.tolist()}

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000)
