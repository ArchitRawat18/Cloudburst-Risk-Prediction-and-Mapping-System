import os
import numpy as np
import xarray as xr
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
import joblib

# ================= ⚙️ CONFIGURATION =================
DATA_DIR = r'F:\Major Project\Project Working\Datasets\Hardened_Data'
MODEL_DIR = r'F:\Major Project\Project Working\Modelling_i2'
FEATURES = ['tp', 'z', 'q', 't', 'u', 'v', 'r', 'z_dip', 'r_spike', 'tp_trend']

def get_data(file_path):
    with xr.open_dataset(file_path) as ds:
        if 'pressure_level' in ds.dims: ds = ds.isel(pressure_level=0)
        # Compute engineered features [cite: 476, 551]
        ds = ds.assign(
            z_dip=ds['z'].diff('valid_time').fillna(0),
            r_spike=ds['r'].diff('valid_time').fillna(0),
            tp_trend=ds['tp'].rolling(valid_time=3).mean().fillna(0)
        )
        return ds[FEATURES + ['target']].to_dataframe().reset_index().dropna()

# --- 🚀 EXECUTION ---
all_samples = []
# TRICK: Include July (07) to capture the Samej-style intensity
for year in range(2015, 2022): 
    for month in ['06', '07']: 
        y_path = os.path.join(DATA_DIR, str(year))
        files = [os.path.join(y_path, f) for f in os.listdir(y_path) if f"_{month}_" in f]
        if files:
            df = get_data(files[0]) # Take one representative file
            # Over-sample the rare events to force the RF to see them
            bursts = df[df['target'] > 0]
            normals = df[df['target'] == 0].sample(frac=0.01) # Keep it lean but diverse
            all_samples.append(pd.concat([bursts, normals]))

train_df = pd.concat(all_samples)
scaler = MinMaxScaler()
X = scaler.fit_transform(train_df[FEATURES])
y = train_df['target']

# Train the "Spatial Brain" [cite: 561, 610]
rf = RandomForestClassifier(n_estimators=150, class_weight='balanced', n_jobs=-1)
rf.fit(X, y)

# SAVE EVERYTHING
joblib.dump(rf, os.path.join(MODEL_DIR, "rf_spatial_engine.pkl"))
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl")) # THIS IS THE KEY
print("✅ Hybrid Spatial Engine & Scaler Saved!")