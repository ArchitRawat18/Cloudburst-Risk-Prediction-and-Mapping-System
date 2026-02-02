import os
import numpy as np
import xarray as xr
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

# --- Config ---
MODEL_PATH = r'F:\Major Project\Project Working\Modelling\cloudburst_model.keras'
TEST_YEARS = ["2024", "2025"] # Purely unseen data
DATA_DIR = r'F:\Major Project\Project Working\Datasets\Hardened_Data'
FEATURES = ['tp', 'z', 'q', 't', 'u', 'v', 'r'] 
LOOKBACK = 24

def run_spatial_test():
    model = load_model(MODEL_PATH)
    all_y_true = []
    all_y_pred = []

    print("🚀 Running Spatial Scan on 2024-2025 Test Set...")

    for year in TEST_YEARS:
        y_path = os.path.join(DATA_DIR, year)
        files = [os.path.join(y_path, f) for f in os.listdir(y_path) if f.endswith('.nc')]
        
        for f in files:
            with xr.open_dataset(f) as ds:
                if 'pressure_level' in ds.dims: 
                    ds = ds.isel(pressure_level=0)
                
                # Preprocessing
                data = ds[FEATURES].to_array().transpose('valid_time', 'latitude', 'longitude', 'variable').values
                data = (data - data.min()) / (data.max() - data.min() + 1e-7)
                labels = ds.target.values # (Time, Lat, Lon)
                
                # Spatial Scan Loop
                for lat_idx in range(data.shape[1]):
                    for lon_idx in range(data.shape[2]):
                        # Optimization: Only test pixels that contain an event (Label > 0)
                        if np.any(labels[:, lat_idx, lon_idx] > 0):
                            for t in range(LOOKBACK, data.shape[0]):
                                # Get the 24-hour sequence for THIS specific pixel
                                seq = data[t-LOOKBACK:t, lat_idx, lon_idx, :]
                                
                                # Add to ground truth
                                all_y_true.append(labels[t, lat_idx, lon_idx])
                                
                                # Model Prediction
                                p = model.predict(seq.reshape(1, LOOKBACK, -1), verbose=0)
                                all_y_pred.append(np.argmax(p))

    # --- Print the Real Report ---
    print("\n" + "="*40)
    print("🏆 SPATIAL EVALUATION REPORT (2024-2025)")
    print("="*40)
    print(classification_report(all_y_true, all_y_pred, 
                                target_names=['Normal', 'Heavy', 'Extreme', 'Cloudburst'],
                                labels=[0, 1, 2, 3], zero_division=0))

if __name__ == "__main__":
    run_spatial_test()