
import geopandas as gpd
import pandas as pd
import numpy as np

# Load district names and codes from GeoJSON
uttarakhand_map = gpd.read_file('UTTARAKHAND_DISTRICTS.geojson')
district_names = uttarakhand_map['dtname'].str.strip().tolist()
district_codes = uttarakhand_map['dtcode11'].astype(int).tolist()
NUM_DISTRICTS = len(district_codes)

print("Generating synthetic SDOH data for Uttarakhand...")

sdoh_data = {
    'dtcode11': district_codes,
    'dtname': district_names
}
sdoh_df = pd.DataFrame(sdoh_data)

np.random.seed(42)
disadvantage_score = np.random.uniform(0.1, 1.0, NUM_DISTRICTS)
sdoh_df['disadvantage_score'] = disadvantage_score

# Generate SDOH variables
sdoh_df['Pct_Pop_Below_Poverty'] = np.clip(10 + 30 * disadvantage_score + np.random.normal(0, 3, NUM_DISTRICTS), 5, 50)
sdoh_df['Pct_Scheduled_Caste'] = np.clip(5 + 40 * disadvantage_score + np.random.normal(0, 4, NUM_DISTRICTS), 5, 60)
sdoh_df['Avg_Household_Income'] = np.clip(50000 - 40000 * disadvantage_score + np.random.normal(0, 5000, NUM_DISTRICTS), 8000, 60000)
sdoh_df['Pct_Illiterate'] = np.clip(3 + 15 * disadvantage_score + np.random.normal(0, 2, NUM_DISTRICTS), 2, 25)
sdoh_df['Literacy_Rate'] = np.clip(90 - 40 * disadvantage_score + np.random.normal(0, 5, NUM_DISTRICTS), 40, 95)
sdoh_df['Pct_HH_No_Toilet'] = np.clip(10 + 40 * disadvantage_score + np.random.normal(0, 5, NUM_DISTRICTS), 10, 60)
sdoh_df['Pct_HH_Electricity'] = np.clip(98 - 60 * disadvantage_score + np.random.normal(0, 5, NUM_DISTRICTS), 30, 99)
sdoh_df['Pct_HH_Clean_Cooking_Fuel'] = np.clip(95 - 70 * disadvantage_score + np.random.normal(0, 5, NUM_DISTRICTS), 20, 99)
sdoh_df['Air_Quality_Index_Avg'] = np.clip(50 + 150 * disadvantage_score + np.random.normal(0, 10, NUM_DISTRICTS), 40, 250)
sdoh_df['Health_Insurance_Coverage'] = np.clip(90 - 50 * disadvantage_score + np.random.normal(0, 5, NUM_DISTRICTS), 30, 98)
sdoh_df['Primary_Health_Centers_Per_100k'] = np.clip(5 - 4 * disadvantage_score + np.random.normal(0, 0.5, NUM_DISTRICTS), 0.5, 6)

sdoh_df = sdoh_df.drop(columns=['disadvantage_score'])
sdoh_df.to_csv('synthetic_sdoh_dataset.csv', index=False)

print("\n--- Synthetic SDOH Data Generation Complete ---")
print(sdoh_df.head())