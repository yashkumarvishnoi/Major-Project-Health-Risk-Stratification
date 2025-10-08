
import geopandas as gpd
import pandas as pd
import numpy as np

# Load district codes from GeoJSON
uttarakhand_map = gpd.read_file('UTTARAKHAND_DISTRICTS.geojson')
district_codes = uttarakhand_map['dtcode11'].astype(int).tolist()
NUM_DISTRICTS = len(district_codes)
NUM_PATIENTS = 50000

print("Generating synthetic clinical data for Tuberculosis in Uttarakhand...")

np.random.seed(202)
clinical_data = {
    'Patient_ID': range(1001, 1001 + NUM_PATIENTS),
    'dtcode11': np.random.choice(district_codes, NUM_PATIENTS),
    'Age': np.random.randint(10, 85, NUM_PATIENTS),
    'Sex': np.random.choice(['Male', 'Female'], NUM_PATIENTS, p=[0.55, 0.45]),
    'BMI': np.clip(np.random.normal(21, 5, NUM_PATIENTS), 13, 40),
    'Smoker': np.random.choice([0, 1], NUM_PATIENTS, p=[0.75, 0.25]),
    'HIV_Positive': np.random.choice([0, 1], NUM_PATIENTS, p=[0.97, 0.03]),
    'Family_History_TB': np.random.choice([0, 1], NUM_PATIENTS, p=[0.92, 0.08]),
    'Living_in_Crowded_Area': np.random.choice([0, 1], NUM_PATIENTS, p=[0.65, 0.35])
}

clinical_df = pd.DataFrame(clinical_data)

# Compute a TB risk score based on clinical and behavioral factors
risk_score = (
    0.05 * (clinical_df['Age'] - 40) +
    -0.12 * (clinical_df['BMI'] - 21) +  # lower BMI increases risk
    1.2 * clinical_df['Smoker'] +
    1.5 * clinical_df['HIV_Positive'] +
    0.9 * clinical_df['Family_History_TB'] +
    1.3 * clinical_df['Living_in_Crowded_Area'] +
    np.random.normal(0, 0.5, NUM_PATIENTS) - 2.0
)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Apply sigmoid to get TB probability, then assign Has_TB
tb_probability = sigmoid(risk_score)
clinical_df['Has_TB'] = (np.random.rand(NUM_PATIENTS) < tb_probability).astype(int)

clinical_df.to_csv('synthetic_tb_clinical_dataset.csv', index=False)

print("\n--- Synthetic Tuberculosis Clinical Data Generation Complete ---")
print(clinical_df.head())
