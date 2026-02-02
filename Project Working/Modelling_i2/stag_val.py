import xarray as xr
import numpy as np
import joblib
import pandas as pd
from tensorflow.keras.models import load_model

rf = joblib.load(r'F:\Major Project\Project Working\Modelling_i2\rf_spatial_engine.pkl')
scaler = joblib.load(r'F:\Major Project\Project Working\Modelling_i2\scaler.pkl')
lstm = load_model(r'F:\Major Project\Project Working\Modelling_i2\lstm_temporal_engine.keras')

FEATURES = ['tp', 'z', 'q', 't', 'u', 'v', 'r', 'z_dip', 'r_spike', 'tp_trend']

def evaluate_hybrid_v3(file_path, hour_idx):
    with xr.open_dataset(file_path) as ds:
        if 'pressure_level' in ds.dims: ds = ds.isel(pressure_level=0)
        
        # 1. Feature Engineering [cite: 476, 551]
        ds = ds.assign(
            z_dip=ds['z'].diff('valid_time').fillna(0),
            r_spike=ds['r'].diff('valid_time').fillna(0),
            tp_trend=ds['tp'].rolling(valid_time=3).mean().fillna(0)
        )
        
        # 2. Preparation
        raw_df = ds[FEATURES].to_dataframe().reset_index().dropna()
        scaled_features = scaler.transform(raw_df[FEATURES])
        
        # 3. Spatial Slice for the specific hour
        hour_mask = raw_df['valid_time'] == ds.valid_time.values[hour_idx]
        
        # RF Soft Probability (Sum of all non-normal classes)
        rf_all_probs = rf.predict_proba(scaled_features[hour_mask])
        rf_probs = np.sum(rf_all_probs[:, 1:], axis=1) 
        
        # LSTM Temporal Data [cite: 556]
        da_lstm = ds[FEATURES].to_array().transpose('valid_time', 'latitude', 'longitude', 'variable').values
        lstm_scaled = (da_lstm - da_lstm.min()) / (da_lstm.max() - da_lstm.min() + 1e-7)
        
        # We test 4 different geographical points
        test_points = [(10, 10), (20, 20), (30, 30), (5, 35)]
        print(f"--- Hybrid Fusion (Spatial Variance Test) ---")
        
        for lat, lon in test_points:
            # RF value for this pixel
            # Find index in the flat array for this lat/lon
            idx = np.where((raw_df[hour_mask]['latitude'] == ds.latitude.values[lat]) & 
                           (raw_df[hour_mask]['longitude'] == ds.longitude.values[lon]))[0][0]
            r_p = rf_probs[idx]
            
            # LSTM value for this pixel (Crucial: Use the actual lat/lon sequence!)
            lstm_input = lstm_scaled[hour_idx-24:hour_idx, lat, lon, :].reshape(1, 24, 10)
            l_p = lstm.predict(lstm_input, verbose=0)[0][3]
            
            final_risk = (r_p * 0.6) + (l_p * 0.4)
            print(f"Coord ({lat}, {lon}) | RF Risk: {r_p:.4f} | LSTM Trend: {l_p:.4f} | TOTAL: {final_risk:.4f}")

evaluate_hybrid_v3(r'F:\Major Project\Project Working\Datasets\Hardened_Data\2024\Hardened_2024_07_21.nc', 263)