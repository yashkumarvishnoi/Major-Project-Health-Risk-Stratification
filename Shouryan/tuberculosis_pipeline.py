import pandas as pd
import geopandas as gpd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import lightgbm as lgb

clinical_df = pd.read_csv('synthetic_tb_clinical_dataset.csv')
sdoh_df = pd.read_csv('synthetic_tb_sdoh_dataset.csv')

merged_df = clinical_df.merge(sdoh_df, on='dtcode11', how='left')
print("Merged dataset shape:", merged_df.shape)
print(merged_df.head())
y = merged_df['Has_TB']
X = merged_df.drop(columns=['Has_TB', 'Patient_ID', 'dtname', 'dtcode11'])
X = pd.get_dummies(X, columns=['Sex'], drop_first=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
lgbm = lgb.LGBMClassifier(random_state=42, class_weight='balanced')
lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Reds')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Tuberculosis Classification")
plt.show()

importances = lgbm.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8,6))
plt.barh(feature_names, importances)
plt.xlabel("Feature Importance")
plt.title("LightGBM Feature Importances - TB Model")
plt.show()

uttarakhand_map = gpd.read_file('UTTARAKHAND_DISTRICTS.geojson')
uttarakhand_map['dtcode11'] = uttarakhand_map['dtcode11'].astype(int)

district_risk = merged_df.groupby('dtcode11')['Has_TB'].mean().reset_index()
district_risk['dtcode11'] = district_risk['dtcode11'].astype(int)
district_risk.columns = ['dtcode11', 'Predicted_TB_Risk']

map_with_risk = uttarakhand_map.merge(district_risk, on='dtcode11')

fig, ax = plt.subplots(figsize=(12, 10))
map_with_risk.plot(column='Predicted_TB_Risk', cmap='OrRd', legend=True, ax=ax, edgecolor='black')

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
