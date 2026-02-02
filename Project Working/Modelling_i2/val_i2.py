import xarray as xr
import numpy as np
import joblib
import pandas as pd
from tensorflow.keras.models import load_model

# Load Hybrid Engines
rf = joblib.load(r'F:\Major Project\Project Working\Modelling_i2\rf_spatial_engine.pkl')
scaler = joblib.load(r'F:\Major Project\Project Working\Modelling_i2\scaler.pkl')
lstm = load_model(r'F:\Major Project\Project Working\Modelling_i2\lstm_temporal_engine_i2.keras')

FEATURES = ['tp', 'z', 'q', 't', 'u', 'v', 'r', 'z_dip', 'r_spike', 'tp_trend']

def final_fusion_test(file_path, hour_idx):
    with xr.open_dataset(file_path) as ds:
        if 'pressure_level' in ds.dims: ds = ds.isel(pressure_level=0)
        
        # 1. Real-time Feature Engineering 
        ds = ds.assign(
            z_dip=ds['z'].diff('valid_time').fillna(0),
            r_spike=ds['r'].diff('valid_time').fillna(0),
            tp_trend=ds['tp'].rolling(valid_time=3).mean().fillna(0)
        )
        
        # 2. RF Path (Spatial) 
        raw_df = ds[FEATURES].to_dataframe().reset_index().dropna()
        scaled_rf = scaler.transform(raw_df[FEATURES])
        hour_mask = raw_df['valid_time'] == ds.valid_time.values[hour_idx]
        
        # Extract probabilities for the selected hour [cite: 496]
        rf_all_probs = rf.predict_proba(scaled_rf[hour_mask])
        rf_probs = np.sum(rf_all_probs[:, 1:], axis=1) 
        
        # 3. LSTM Path (Temporal) [cite: 484]
        da_lstm = ds[FEATURES].to_array().transpose('valid_time', 'latitude', 'longitude', 'variable').values
        # Safe Normalization to match training
        d_min = da_lstm.min(axis=(0,1,2), keepdims=True)
        d_max = da_lstm.max(axis=(0,1,2), keepdims=True)
        lstm_scaled = (da_lstm - d_min) / (d_max - d_min + 1e-7)
        
        test_points = [(10, 10), (20, 20), (30, 30), (5, 35)]
        print(f"--- Iteration 2 Hybrid Fusion: {ds.valid_time.values[hour_idx]} ---")
        
        for lat, lon in test_points:
            # Match flattened RF index to the grid
            p_idx = np.where((raw_df[hour_mask]['latitude'] == ds.latitude.values[lat]) & 
                           (raw_df[hour_mask]['longitude'] == ds.longitude.values[lon]))[0][0]
            r_p = rf_probs[p_idx]
            
            # LSTM risk for this specific coordinate
            l_input = lstm_scaled[hour_idx-24:hour_idx, lat, lon, :].reshape(1, 24, 10)
            l_p = lstm.predict(l_input, verbose=0)[0][3]
            
            # Weighted Decision Fusion (70% Spatial / 30% Temporal) [cite: 491, 570]
            final_risk = (r_p * 0.7) + (l_p * 0.3)
            print(f"Coord ({lat}, {lon}) | RF: {r_p:.4f} | LSTM: {l_p:.4f} | FUSED: {final_risk:.4f}")

# Run test on Samej hour
final_fusion_test(r'F:\Major Project\Project Working\Datasets\Hardened_Data\2024\Hardened_2024_08_21.nc', 263)