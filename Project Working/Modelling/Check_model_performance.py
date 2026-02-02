import os
import numpy as np
import xarray as xr
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

# --- Config ---
MODEL_PATH = r'F:\Major Project\Project Working\Modelling\cloudburst_final_v2.keras'
TEST_YEARS = ["2024", "2025"] # Purely unseen data
DATA_DIR = r'F:\Major Project\Project Working\Datasets\Hardened_Data'
FEATURES = ['tp', 'z', 'q', 't', 'u', 'v', 'r'] 
LOOKBACK = 24

def run_fast_test():
    model = load_model(MODEL_PATH)
    all_y_true, all_y_pred = [], []

    print("🚀 Running Targeted Evaluation on Cloudburst Zones...")

    for year in TEST_YEARS:
        y_path = os.path.join(DATA_DIR, year)
        files = [os.path.join(y_path, f) for f in os.listdir(y_path) if f.endswith('.nc')]
        
        for f in files:
            with xr.open_dataset(f) as ds:
                if 'pressure_level' in ds.dims: ds = ds.isel(pressure_level=0)
                data = ds[FEATURES].to_array().transpose('valid_time','latitude','longitude','variable').values
                data = (data - data.min()) / (data.max() - data.min() + 1e-7)
                labels = ds.target.values
                
                # OPTIMIZATION: Only look at pixels where Label > 0 exists
                # np.argwhere finds exactly which (lat, lon) indices have interesting data
                interesting_indices = np.argwhere(np.any(labels > 0, axis=0))
                
                for lat_idx, lon_idx in interesting_indices:
                    # Create a batch of all time steps for this specific pixel at once
                    X_pixel = []
                    y_pixel_true = []
                    for t in range(LOOKBACK, data.shape[0]):
                        X_pixel.append(data[t-LOOKBACK:t, lat_idx, lon_idx, :])
                        y_pixel_true.append(labels[t, lat_idx, lon_idx])
                    
                    if X_pixel:
                        # Batch prediction is MUCH faster than individual calls
                        preds = model.predict(np.array(X_pixel), verbose=0)
                        all_y_pred.extend(np.argmax(preds, axis=1))
                        all_y_true.extend(y_pixel_true)
            print(f"✅ Processed: {os.path.basename(f)}")

    print("\n" + "="*40)
    print("🏆 TARGETED EVALUATION REPORT (2024-2025)")
    print("="*40)
    print(classification_report(all_y_true, all_y_pred, 
                                target_names=['Normal','Heavy','Extreme','Cloudburst'],
                                labels=[0,1,2,3], zero_division=0))

if __name__ == "__main__":
    run_fast_test()