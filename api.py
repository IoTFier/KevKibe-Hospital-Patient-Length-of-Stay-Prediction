import pickle
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
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
    model_path = r"C:\Users\kibe\Desktop\Projects\Hospital-LOS-Prediction\LOSModel7.pkl"
    with open(model_path,"rb") as f:
        model=pickle.load(f)


@app.post('/predict')
async def make_prediction(input_data: InputData):

    input_dict = input_data.dict()

    input_df = pd.DataFrame([input_dict])

    categorical_features =['Age', 'gender', 'Type_of_Admission', 'Severity_of_Illness', 'health_conditions', 'Insurance',
                           'Ward_Facility_Code', 'doctor_name', 'Department']
    encoded_features = pd.get_dummies(input_df[categorical_features], drop_first=True)

    expected_columns = ['Available_Extra_Rooms_in_Hospital', 'staff_available',
       'Visitors_with_Patient', 'Admission_Deposit', 'Department_anesthesia',
       'Department_gynecology', 'Department_radiotherapy',
       'Department_surgery', 'Ward_Facility_Code_B', 'Ward_Facility_Code_C',
       'Ward_Facility_Code_D', 'Ward_Facility_Code_E', 'Ward_Facility_Code_F',
       'doctor_name_Dr John', 'doctor_name_Dr Mark', 'doctor_name_Dr Nathan',
       'doctor_name_Dr Olivia', 'doctor_name_Dr Sam', 'doctor_name_Dr Sarah',
       'doctor_name_Dr Simon', 'doctor_name_Dr Sophia', 'Age_11-20',
       'Age_21-30', 'Age_31-40', 'Age_41-50', 'Age_51-60', 'Age_61-70',
       'Age_71-80', 'Age_81-90', 'Age_91-100', 'gender_Male', 'gender_Other',
       'Type_of_Admission_Trauma', 'Type_of_Admission_Urgent',
       'Severity_of_Illness_Minor', 'Severity_of_Illness_Moderate',
       'health_conditions_Diabetes', 'health_conditions_Heart disease',
       'health_conditions_High Blood Pressure', 'health_conditions_None',
       'health_conditions_Other', 'Insurance_Yes'
                        ]
    for column in expected_columns:
        if column not in encoded_features.columns:
            encoded_features[column] = 0
    column_order = ['Available_Extra_Rooms_in_Hospital', 'staff_available',
       'Visitors_with_Patient', 'Admission_Deposit', 'Department_anesthesia',
       'Department_gynecology', 'Department_radiotherapy',
       'Department_surgery', 'Ward_Facility_Code_B', 'Ward_Facility_Code_C',
       'Ward_Facility_Code_D', 'Ward_Facility_Code_E', 'Ward_Facility_Code_F',
       'doctor_name_Dr John', 'doctor_name_Dr Mark', 'doctor_name_Dr Nathan',
       'doctor_name_Dr Olivia', 'doctor_name_Dr Sam', 'doctor_name_Dr Sarah',
       'doctor_name_Dr Simon', 'doctor_name_Dr Sophia', 'Age_11-20',
       'Age_21-30', 'Age_31-40', 'Age_41-50', 'Age_51-60', 'Age_61-70',
       'Age_71-80', 'Age_81-90', 'Age_91-100', 'gender_Male', 'gender_Other',
       'Type_of_Admission_Trauma', 'Type_of_Admission_Urgent',
       'Severity_of_Illness_Minor', 'Severity_of_Illness_Moderate',
       'health_conditions_Diabetes', 'health_conditions_Heart disease',
       'health_conditions_High Blood Pressure', 'health_conditions_None',
       'health_conditions_Other', 'Insurance_Yes'
                        ]
    
    processed_features = pd.concat([encoded_features, input_df.drop(categorical_features, axis=1)], axis=1)[column_order]
    predictions = model.predict(processed_features)

    rounded_predictions = [int(round(prediction)) for prediction in predictions]

    return {"predictions": rounded_predictions}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
