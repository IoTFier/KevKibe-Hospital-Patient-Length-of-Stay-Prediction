from flask import Flask, request, jsonify
import pandas as pd
import pickle
import numpy as np
import json
import requests

app = Flask(__name__)

enc = pickle.load(open('encoder.pkl','rb'))
features = pickle.load(open('features.pkl','rb'))                  
model=pickle.load(open('modelv3.pickle','rb'))


@app.route('/predict', methods=['POST'])
def predict():
    input_data = request.get_json()

    #input df
    df= pd.DataFrame(input_data, index=[0])

    #matching input data to data that was fit to model features
    columns_to_encode= ['Age', 'gender', 'Type_of_Admission', 'Severity_of_Illness', 'health_conditions', 'Insurance',
                         'Ward_Facility_Code', 'doctor_name', 'Department']
    #locating the categorical features
    df = df.loc[:, columns_to_encode]
    #creating a list for those categorical features in the input data
    to_encode = [input_data[col] for col in columns_to_encode]
    #using the imported encoder to encode the features
    encoded_features = list(enc.transform(np.array(to_encode).reshape(1,-1))[0])

    to_predict = [input_data[feature] for feature in features if feature not in columns_to_encode]
    input_data = to_predict + encoded_features
    input_columns = [feature for feature in features if feature not in columns_to_encode] + list(enc.get_feature_names_out(columns_to_encode))
    df_encoded = pd.DataFrame([input_data], columns=input_columns)

    #prediction
    prediction = model.predict(df_encoded)
    return jsonify({'prediction': prediction.tolist()})







if __name__ == "__main__":
    app.run()
