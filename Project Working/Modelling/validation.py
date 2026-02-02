import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pandas as pd

def automated_validation_search(file_path, target_date):
    # Load your current best model
    model = load_model(r'F:\Major Project\Project Working\Modelling\cloudburst_final_v2.keras')
    
    with xr.open_dataset(file_path) as ds:
        if 'pressure_level' in ds.dims: ds = ds.isel(pressure_level=0)
        
        # 1. AUTOMATED SEARCH: Find the target hour in the file
        time_values = pd.to_datetime(ds.valid_time.values)
        target_indices = np.where(time_values.strftime('%Y-%m-%d') == target_date)[0]
        
        if len(target_indices) == 0:
            print(f"❌ Date {target_date} not found in this file.")
            return

        # For the showcase, we pick the hour with the most intense rain (max target value)
        hour_idx = target_indices[np.argmax([ds.target.values[i].max() for i in target_indices])]
        print(f"📅 Automated Search found peak event at: {ds.valid_time.values[hour_idx]}")

        # 2. DATA PREPARATION
        features = ['tp', 'z', 'q', 't', 'u', 'v', 'r']
        data = ds[features].to_array().transpose('valid_time','latitude','longitude','variable').values
        data = (data - data.min()) / (data.max() - data.min() + 1e-7)
        
        lat_size, lon_size = ds.sizes['latitude'], ds.sizes['longitude']
        all_pixels = []
        for lat in range(lat_size):
            for lon in range(lon_size):
                all_pixels.append(data[hour_idx-24:hour_idx, lat, lon, :])
        
        # 3. PREDICTION
        preds = model.predict(np.array(all_pixels), batch_size=64, verbose=0)
        prob_map = preds[:, 3].reshape(lat_size, lon_size)

        # 4. HIGHLIGHT LOGIC (Your visibility fix)
        max_idx = np.unravel_index(np.argmax(prob_map, axis=None), prob_map.shape)
        max_lat, max_lon = ds.latitude.values[max_idx[0]], ds.longitude.values[max_idx[1]]
        
        clean_prob_map = prob_map.copy()
        clean_prob_map[clean_prob_map < 0.021] = np.nan 

        # 5. SIDE-BY-SIDE SHOWCASE
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        extent = [ds.longitude.min(), ds.longitude.max(), ds.latitude.min(), ds.latitude.max()]
        
        # Left: Actual Ground Truth
        im1 = ax1.imshow(ds.target.values[hour_idx], cmap='Blues', extent=extent, origin='upper')
        ax1.set_title(f"ACTUAL EVENTS (Ground Truth)\n{ds.valid_time.values[hour_idx]}", fontsize=12)
        plt.colorbar(im1, ax=ax1, label='Label (3=Cloudburst)')

        # Right: Model Prediction + COORDINATE HIGHLIGHT
        im2 = ax2.imshow(clean_prob_map, cmap='YlOrRd', extent=extent, origin='upper')
        # The Blue Circle to ensure you can see the dot
        ax2.scatter(max_lon, max_lat, s=400, facecolors='none', edgecolors='blue', linewidths=3, label='Model Detected Peak')
        # The Coordinate Label
        ax2.text(max_lon, max_lat + 0.3, f"Risk Peak: {max_lat:.2f}N, {max_lon:.2f}E", 
                 color='blue', fontweight='bold', ha='center', bbox=dict(facecolor='white', alpha=0.8))
        
        ax2.set_title("MODEL PREDICTION (Risk Map with Coordinate Highlight)", fontsize=12)
        plt.colorbar(im2, ax=ax2, label='Probability Score')
        ax2.legend()

        plt.tight_layout()
        plt.show()

# --- RUN IT FOR YOUR EVALUATION ---
FILE = r'F:\Major Project\Project Working\Datasets\Hardened_Data\2024\Hardened_2024_07_21.nc'
automated_validation_search(FILE, "2024-07-25")