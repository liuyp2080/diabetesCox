from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
import joblib
import json
import uvicorn
import os
import pandas as pd
import numpy as np
import gunicorn
#from dotenv import load_dotenv

# load_dotenv()
# root_path = os.getenv("SCRIPT_NAME", "")
app = FastAPI()

class model_input(BaseModel):
    fasting_plasma_glucose: float
    family_history_of_diabetes: int
    weight: float
    cholesterol: float
    blood_urea_nitrogen: float  
    drinking_status: int
    age: int
    
diabetes_model = joblib.load("rsf_best.lzma")

@app.post('/diabetes_pred_time')
def diabetes_pred_time(input_parameters: model_input):
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    patient = pd.DataFrame(input_dictionary,index=[0]) 
    times= np.arange(3,5,1/6)
    chf_funcs=diabetes_model.predict_cumulative_hazard_function(patient)
    for fn in chf_funcs:
        if fn(times[-1])<1:
            return '该患者在{}年内预测不会患糖尿病'.format(round(times[-1],2))
            
        else:
            for time in times:
                if fn(time)>1:
                    time_value=time
                    break
                return '该患者将在{}年时患糖尿病'.format(time_value)
                

@app.post('/diabetes_pred_proba')
def diabetes_pred_proba(input_parameters:model_input,time: int):
    input_data=input_parameters.json()
    input_dictionary=json.loads(input_data)
    patient=pd.DataFrame(input_dictionary,index=[0])
    chf_funcs=diabetes_model.predict_cumulative_hazard_function(patient)
    for fn in chf_funcs:
        result=fn(time)
        result=round(result,3)*100
        return '该患者在{}年内患糖尿病的概率是{}%'.format(time,result)
        
    

if __name__ == '__main__':
    gunicorn -w 2 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8000 main:app
    
    
