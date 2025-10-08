import pandas as pd
import numpy as np
import random

# Generate synthetic clinical dataset for Diabetes Risk Prediction
N = 2000
districts = ['dehradun','haridwar','nainital','pauri','tehri','almora','pithoragarh','chamoli','rudraprayag','bageshwar','champawat','udhamsinghnagar','uttarkashi']

def make_patient(i):
    age = np.random.randint(18,80)
    sex = random.choice(['M','F'])
    district = random.choice(districts)
    bmi = round(np.random.normal(25,4),1)
    fasting_glucose = round(np.random.normal(110,30),1)
    hba1c = round(np.random.normal(6.0,1.2),2)
    systolic_bp = int(np.random.normal(125,15))
    smoker = random.choice([0,1])
    family_history = random.choice([0,1])
    physical_activity = random.choice(['low','moderate','high'])

    # Risk logic
    risk_score = 0
    if age>45: risk_score+=1
    if bmi>=30: risk_score+=2
    if hba1c>=6.5: risk_score+=3
    if fasting_glucose>=126: risk_score+=3
    if family_history==1: risk_score+=1
    label = 1 if risk_score>=4 else 0

    return [f'P{i:05d}',age,sex,district,bmi,fasting_glucose,hba1c,
            systolic_bp,smoker,family_history,physical_activity,label]

cols = ['patient_id','age','sex','district','bmi','fasting_glucose',
        'hba1c','systolic_bp','smoker','family_history','physical_activity',
        'high_risk_diabetes']

df = pd.DataFrame([make_patient(i) for i in range(N)], columns=cols)
df.to_csv('synthetic_clinical_dataset_diabetes.csv', index=False)
print("✅ Saved synthetic_clinical_dataset_diabetes.csv with", len(df), "records")
