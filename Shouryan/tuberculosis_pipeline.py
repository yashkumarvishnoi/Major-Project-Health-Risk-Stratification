import pandas as pd
import geopandas as gpd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb

# --- Step 1: Load Synthetic Datasets ---
clinical_df = pd.read_csv('synthetic_tb_clinical_dataset.csv')
sdoh_df = pd.read_csv('synthetic_tb_sdoh_dataset.csv')

# --- Step 2: Merge Clinical and SDOH Data ---
merged_df = clinical_df.merge(sdoh_df, on='dtcode11', how='left')
print("Merged dataset shape:", merged_df.shape)
print(merged_df.head())

# --- Step 3: Define Target and Features ---
y = merged_df['Has_TB']

# Drop ID and text columns
X = merged_df.drop(columns=['Has_TB', 'Patient_ID', 'dtname', 'dtcode11'])

# One-hot encode categorical variables
X = pd.get_dummies(X, columns=['Sex'], drop_first=True)

# --- Step 4: Train-Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Step 5: LightGBM Model ---
lgbm = lgb.LGBMClassifier(random_state=42, class_weight='balanced')
lgbm.fit(X_train, y_train)

# --- Step 6: Evaluate Model ---
y_pred = lgbm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Tuberculosis Classification")
plt.show()

# --- Step 7: Feature Importances ---
importances = lgbm.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8,6))
plt.barh(feature_names, importances)
plt.xlabel("Feature Importance")
plt.title("LightGBM Feature Importances - TB Model")
plt.show()

# --- Step 8: District-Level TB Risk Mapping ---
uttarakhand_map = gpd.read_file('UTTARAKHAND_DISTRICTS.geojson')
uttarakhand_map['dtcode11'] = uttarakhand_map['dtcode11'].astype(int)

# Compute district-level TB prevalence
district_risk = merged_df.groupby('dtcode11')['Has_TB'].mean().reset_index()
district_risk['dtcode11'] = district_risk['dtcode11'].astype(int)
district_risk.columns = ['dtcode11', 'Predicted_TB_Risk']

# Merge with GeoJSON
map_with_risk = uttarakhand_map.merge(district_risk, on='dtcode11')

# Plot choropleth
fig, ax = plt.subplots(figsize=(12, 10))
map_with_risk.plot(column='Predicted_TB_Risk', cmap='OrRd', legend=True, ax=ax, edgecolor='black')

# Add district labels
for idx, row in map_with_risk.iterrows():
    if row['geometry'].centroid.is_empty:
        continue
    plt.annotate(
        text=row['dtname'].strip(),
        xy=(row['geometry'].centroid.x, row['geometry'].centroid.y),
        ha='center', fontsize=9, color='black', weight='bold'
    )

plt.title('Predicted Tuberculosis (TB) Risk by District in Uttarakhand', fontsize=16)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.show()
