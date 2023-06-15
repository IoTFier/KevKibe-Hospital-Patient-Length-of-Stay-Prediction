import requests
to_predict_dict = {"Available_Extra_Rooms_in_Hospital": 8,
                   "Department": "anesthesia",
                   "Ward_Facility_Code": "B",
                   "doctor_name": "Dr Olivia",
                   "staff_available": 7,
                   "Age": "21-30",
                   "gender": "Male",
                   "Type_of_Admission": "Urgent",
                   "Severity_of_Illness": "Minor",
                   "health_conditions": "Diabetes",
                   "Visitors_with_Patient": 4,
                   "Insurance": "Yes",
                   "Admission_Deposit": 4800.0
                   }

url = 'http://127.0.0.1:8000/predict'
r = requests.post(url,json=to_predict_dict); r.json()
print(r.text)