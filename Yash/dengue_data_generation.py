import pandas as pd
import numpy as np
import geopandas as gpd

# --- Configuration ---
try:
    uttarakhand_map = gpd.read_file('UTTARAKHAND_DISTRICTS.geojson')
    district_codes = uttarakhand_map['dtcode11'].astype(int).tolist()
    NUM_DISTRICTS = len(district_codes)
    print(f"Loaded {NUM_DISTRICTS} district codes from GeoJSON file.")
except Exception as e:
    print(f"Error: Could not load 'UTTARAKHAND_DISTRICTS.geojson'. Make sure it's in the folder. Details: {e}")
    exit()

NUM_PATIENTS = 50000

print("Generating synthetic clinical data for a DENGUE outbreak...")

# --- 1. Create the DataFrame with clinical features ---
np.random.seed(101)
clinical_data = {
    'Patient_ID': range(1001, 1001 + NUM_PATIENTS),
    'dtcode11': np.random.choice(district_codes, NUM_PATIENTS),
    'Age': np.random.randint(18, 85, NUM_PATIENTS),
    'Sex': np.random.choice(['Male', 'Female'], NUM_PATIENTS, p=[0.52, 0.48]),
    'BMI': np.clip(np.random.normal(24, 5, NUM_PATIENTS), 15, 45),
    'Has_Hypertension': np.random.choice([0, 1], NUM_PATIENTS, p=[0.8, 0.2]),
    'Has_High_Cholesterol': np.random.choice([0, 1], NUM_PATIENTS, p=[0.7, 0.3]),
    'Family_History_Diabetes': np.random.choice([0, 1], NUM_PATIENTS, p=[0.9, 0.1])
}
clinical_df = pd.DataFrame(clinical_data)

# --- 2. Add Seasonality Attribute ---
day_of_year = np.random.randint(1, 366, size=NUM_PATIENTS)
conditions = [
    (day_of_year >= 151) & (day_of_year <= 243),  # Monsoon: June 1 - Aug 31
    (day_of_year >= 244) & (day_of_year <= 334),  # Post-Monsoon: Sep 1 - Nov 30
]
seasons = ['Monsoon', 'Post-Monsoon']
clinical_df['Season'] = np.select(conditions, seasons, default='Other')

# --- 3. Generate the Target Variable (Has_Dengue) with Seasonal Influence ---
risk_score = (
    0.01 * (clinical_df['Age'] - 40) +
    0.1 * (clinical_df['BMI'] - 22) +
    np.random.normal(0, 0.5, NUM_PATIENTS) - 5.0
)

# Add a strong seasonal multiplier
seasonal_multiplier = clinical_df['Season'].map({'Monsoon': 4.0, 'Post-Monsoon': 3.5}).fillna(0)
risk_score += seasonal_multiplier

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

disease_probability = sigmoid(risk_score)
clinical_df['Has_Dengue'] = (np.random.rand(NUM_PATIENTS) < disease_probability).astype(int)

# --- 4. Save to CSV ---
clinical_df.to_csv('synthetic_clinical_dataset_dengue.csv', index=False)

print("\n--- Synthetic DENGUE Clinical Data Generation Complete ---")
print("First 5 rows of the new clinical dataset with 'Season':")
print(clinical_df.head())
print("\nDistribution of DENGUE cases by season (shows the outbreak pattern):")
print(clinical_df.groupby('Season')['Has_Dengue'].value_counts(normalize=True))
print(f"\nDataset saved to 'synthetic_clinical_dataset_dengue.csv'")