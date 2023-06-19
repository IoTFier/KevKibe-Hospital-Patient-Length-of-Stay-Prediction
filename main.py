import streamlit as st
import pandas as pd
import requests
import base64

def preprocess_data(data):
    column_mapping = {
        'Available Extra Rooms in Hospital': 'Available_Extra_Rooms_in_Hospital',
        'Visitors with Patient': 'Visitors_with_Patient',
        'Stay (in days)': 'Stay_in_Days',
        'Type of Admission': 'Type_of_Admission',
        'Severity of Illness': 'Severity_of_Illness'
    }
    data = data.rename(columns=column_mapping)
    categorical_columns = ['Age', 'gender', 'Type_of_Admission', 'Severity_of_Illness', 'health_conditions', 'Insurance',
                           'Ward_Facility_Code', 'doctor_name', 'Department']
    data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

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
        if column not in data.columns:
            data[column] = 0

    data = data[expected_columns]

    return data

def make_prediction(model, data):
    preprocessed_data = preprocess_data(data)

    # Send a POST request to the API
    response = requests.post('http://localhost:5000/predict', json=preprocessed_data.to_dict())
    predictions = response.json()['predictions']

    data['predicted_los'] = predictions

    return data



st.title('Hospital LOS Prediction')
uploaded_file = st.file_uploader("Upload a file", type=['xlsx'])

if uploaded_file is not None:
    data = pd.read_excel(uploaded_file)
    predicted_data = make_prediction(model, data)

    st.subheader('Modified Excel File')
    st.write(predicted_data)

    # Download the modified Excel file
    csv = predicted_data.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="modified_data.csv">Download Modified Excel File</a>'
    st.markdown(href, unsafe_allow_html=True)
