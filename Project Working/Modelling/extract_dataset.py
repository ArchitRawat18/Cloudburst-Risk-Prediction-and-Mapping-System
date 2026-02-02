import os
import xarray as xr
import numpy as np

# --- CONFIG ---
DATA_DIR = r'F:\Major Project\Project Working\Datasets\Hardened_Data'
SAVE_PATH = r'F:\Major Project\Project Working\Modelling\balanced_training_data.npz'
FEATURES = ['tp', 'z', 'q', 't', 'u', 'v', 'r']
LOOKBACK = 24

def extract_balanced_data():
    x_final, y_final = [], []
    print("🔍 Searching for all Extreme/Cloudburst events...")
    
    for root, dirs, files in os.walk(DATA_DIR):
        for file in files:
            if file.endswith('.nc'):
                with xr.open_dataset(os.path.join(root, file)) as ds:
                    if 'pressure_level' in ds.dims: ds = ds.isel(pressure_level=0)
                    labels = ds.target.values
                    
                    # Check if this file has ANY event (Label 2 or 3)
                    if np.any(labels > 1):
                        data = ds[FEATURES].to_array().transpose('valid_time','latitude','longitude','variable').values
                        data = (data - data.min()) / (data.max() - data.min() + 1e-7)
                        
                        # Find indices of events
                        event_indices = np.argwhere(labels >= 2)
                        for t, lat, lon in event_indices:
                            if t >= LOOKBACK:
                                x_final.append(data[t-LOOKBACK:t, lat, lon, :])
                                y_final.append(labels[t, lat, lon])
                                
                                # Add 1 Normal sample for balance
                                x_final.append(data[t-LOOKBACK:t, 0, 0, :])
                                y_final.append(0)

    np.savez(SAVE_PATH, x=np.array(x_final), y=np.array(y_final))
    print(f"✅ Saved {len(y_final)} balanced samples to {SAVE_PATH}")

extract_balanced_data()