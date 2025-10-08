import geopandas as gpd
import pandas as pd
import numpy as np

uttarakhand_map = gpd.read_file('UTTARAKHAND_DISTRICTS.geojson')
district_names = uttarakhand_map['dtname'].str.strip().tolist()
district_codes = uttarakhand_map['dtcode11'].astype(int).tolist()
NUM_DISTRICTS = len(district_codes)

print("Generating synthetic SDOH data for Tuberculosis in Uttarakhand...")

sdoh_data = {
    'dtcode11': district_codes,
    'dtname': district_names
}
sdoh_df = pd.DataFrame(sdoh_data)

np.random.seed(303)
disadvantage_score = np.random.uniform(0.1, 1.0, NUM_DISTRICTS)
sdoh_df['disadvantage_score'] = disadvantage_score
sdoh_df['Pct_Pop_Below_Poverty'] = np.clip(10 + 35 * disadvantage_score + np.random.normal(0, 4, NUM_DISTRICTS), 5, 55)
sdoh_df['Avg_Household_Income'] = np.clip(48000 - 38000 * disadvantage_score + np.random.normal(0, 6000, NUM_DISTRICTS), 8000, 60000)
sdoh_df['Pct_Urban_Pop'] = np.clip(20 + 50 * (1 - disadvantage_score) + np.random.normal(0, 5, NUM_DISTRICTS), 10, 80)
sdoh_df['Literacy_Rate'] = np.clip(85 - 35 * disadvantage_score + np.random.normal(0, 5, NUM_DISTRICTS), 40, 95)
sdoh_df['Pct_HH_No_Toilet'] = np.clip(10 + 45 * disadvantage_score + np.random.normal(0, 5, NUM_DISTRICTS), 10, 65)
sdoh_df['Pct_HH_Overcrowded'] = np.clip(5 + 35 * disadvantage_score + np.random.normal(0, 4, NUM_DISTRICTS), 5, 50)
sdoh_df['Pct_HH_Electricity'] = np.clip(96 - 60 * disadvantage_score + np.random.normal(0, 5, NUM_DISTRICTS), 30, 99)
sdoh_df['Pct_HH_Clean_Cooking_Fuel'] = np.clip(93 - 65 * disadvantage_score + np.random.normal(0, 6, NUM_DISTRICTS), 15, 99)
sdoh_df['Air_Quality_Index_Avg'] = np.clip(60 + 140 * disadvantage_score + np.random.normal(0, 10, NUM_DISTRICTS), 50, 250)
sdoh_df['Health_Insurance_Coverage'] = np.clip(85 - 55 * disadvantage_score + np.random.normal(0, 5, NUM_DISTRICTS), 20, 95)
sdoh_df['Primary_Health_Centers_Per_100k'] = np.clip(6 - 5 * disadvantage_score + np.random.normal(0, 0.5, NUM_DISTRICTS), 0.5, 6)
sdoh_df = sdoh_df.drop(columns=['disadvantage_score'])
sdoh_df.to_csv('synthetic_tb_sdoh_dataset.csv', index=False)
print("\n--- Synthetic Tuberculosis SDOH Data Generation Complete ---")
print(sdoh_df.head())

