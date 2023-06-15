import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import uvicorn
import numpy as np

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
def load_files():
    global model
    global enc
    global features
    enc = pickle.load(open('encoder.pkl','rb'))
    features = pickle.load(open('features.pkl','rb'))                  
    model=pickle.load(open('modelv3.pickle','rb'))


@app.post('/predict')
async def make_prediction(input_data: InputData):
    input_dict = input_data.dict()

    columns_to_encode = ['Age', 'gender', 'Type_of_Admission', 'Severity_of_Illness', 'health_conditions', 'Insurance',
                         'Ward_Facility_Code', 'doctor_name', 'Department']
    to_encode = [input_dict[col] for col in columns_to_encode]
    encoded_features = list(enc.transform(np.array(to_encode).reshape(1,-1))[0])

    # Input array
    to_predict = [input_dict[feature] for feature in features if feature not in columns_to_encode]
    to_predict += encoded_features

    prediction = model.predict(np.array(to_predict).reshape(1,-1))

    return {"predictions": prediction[0]}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
