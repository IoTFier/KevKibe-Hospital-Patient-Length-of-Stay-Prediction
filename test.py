import requests
import json

input_data = {"Available_Extra_Rooms_in_Hospital": 8,
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

payload = json.dumps(input_data)

headers = {'Content-Type': 'application/json'}

response= requests.post('https://hospital-los-pred-f5m2fxxbbq-uw.a.run.app/predict', data=payload, headers=headers)
print(response.text)

print(response.json())

