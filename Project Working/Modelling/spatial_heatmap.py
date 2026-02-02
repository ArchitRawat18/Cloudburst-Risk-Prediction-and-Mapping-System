import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
from tensorflow.keras.models import load_model

def showcase_samej_prediction(file_path):
    # Load your latest high-intensity model
    model = load_model(r'F:\Major Project\Project Working\Modelling\cloudburst_final_v2.keras')
    
    with xr.open_dataset(file_path) as ds:
        if 'pressure_level' in ds.dims: ds = ds.isel(pressure_level=0)
        
        # 1. Prepare Data
        features = ['tp', 'z', 'q', 't', 'u', 'v', 'r']
        data = ds[features].to_array().transpose('valid_time','latitude','longitude','variable').values
        data = (data - data.min()) / (data.max() - data.min() + 1e-7)
        
        # 2. Timing: Samej Event Peak
        LOOKBACK = 24
        hour_idx = 740 
        
        if hour_idx >= ds.sizes['valid_time']:
            hour_idx = ds.sizes['valid_time'] - 1
            print(f"⚠️ Hour index too high. Adjusting to last hour: {hour_idx}")

        if hour_idx < LOOKBACK:
            raise ValueError(f"hour_idx must be at least {LOOKBACK}. Current index: {hour_idx}")

        print(f"📅 Analyzing Timestamp: {ds.valid_time.values[hour_idx]}")
        lat_size, lon_size = ds.sizes['latitude'], ds.sizes['longitude']
        
        # 3. FAST BATCH PREDICTION
        all_pixels = []
        for lat in range(lat_size):
            for lon in range(lon_size):
                all_pixels.append(data[hour_idx-LOOKBACK:hour_idx, lat, lon, :])
        
        print(f"🗺️ Analyzing {len(all_pixels)} spatial points simultaneously...")
        raw_preds = model.predict(np.array(all_pixels), batch_size=64, verbose=1)
        
        # Extract Cloudburst Probability (Class 3)
        prediction_map = raw_preds[:, 3].reshape(lat_size, lon_size)

        # 4. SHOWCASE PLOTTING
        plt.figure(figsize=(12, 7))
        extent = [ds.longitude.min(), ds.longitude.max(), ds.latitude.min(), ds.latitude.max()]
        
        # FIND THE MAXIMUM RISK POINT
        # We find this before applying the NaN mask
        max_idx = np.unravel_index(np.argmax(prediction_map, axis=None), prediction_map.shape)
        max_lat = ds.latitude.values[max_idx[0]]
        max_lon = ds.longitude.values[max_idx[1]]
        
        # Apply mask for visual clarity
        clean_map = prediction_map.copy()
        clean_map[clean_map < 0.021] = np.nan 

        plt.imshow(clean_map, cmap='YlOrRd', extent=extent, origin='upper')
        plt.colorbar(label='Model Probability of Cloudburst')
        
        # DRAW THE HIGHLIGHT CIRCLE
        # s=500 makes the circle large and unmistakable
        plt.scatter(max_lon, max_lat, s=500, facecolors='none', edgecolors='blue', 
                    linewidths=3, label='Detected High Risk Zone')
        
        # Add a text label next to the circle
        plt.text(max_lon + 0.2, max_lat, f"Risk Peak: {max_lat:.2f}N, {max_lon:.2f}E", 
                 color='blue', fontweight='bold', bbox=dict(facecolor='white', alpha=0.7))

        plt.title(f"Himalayan Cloudburst Risk Detection\nDate: 2024-07-31 | Time: {ds.valid_time.values[hour_idx]}", fontsize=14)
        plt.xlabel("Longitude (°E)", fontsize=12)
        plt.ylabel("Latitude (°N)", fontsize=12)
        plt.grid(alpha=0.3)
        plt.legend(loc='upper right')
        
        plt.show()
        
        print(f"🎯 Detection found at: {max_lat:.4f}°N, {max_lon:.4f}°E")

# --- RUN IT ---
FILE = r'F:\Major Project\Project Working\Datasets\Hardened_Data\2024\Hardened_2024_07_21.nc'
showcase_samej_prediction(FILE)