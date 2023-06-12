import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import requests


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
    model_url = "https://github.com/username/repo/releases/latest/download/LOSmodel.pkl"
    response = requests.get(model_url)
    response.raise_for_status()
    model = pickle.loads(response.content)

@app.post('/predict')
async def make_prediction(input_data: InputData):
    # Convert the input data to a dictionary
    input_dict = input_data.dict()

    # Create a DataFrame from the input dictionary
    input_df = pd.DataFrame([input_dict])

    # Perform one-hot encoding on the categorical features
    categorical_features = ['Department', 'Ward_Facility_Code', 'doctor_name', 'Age', 'gender',
                            'Type_of_Admission', 'Severity_of_Illness', 'health_conditions', 'Insurance']
    encoded_features = pd.get_dummies(input_df[categorical_features], drop_first=True)

    processed_features = pd.concat([encoded_features, input_df.drop(categorical_features, axis=1)], axis=1)

    predictions = model.predict(processed_features)

    return {"predictions": predictions.tolist()}

