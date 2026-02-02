import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

def practical_validation(file_path, hour_idx):
    model = load_model(r'F:\Major Project\Project Working\Modelling\cloudburst_final_v2.keras')
    
    with xr.open_dataset(file_path) as ds:
        if 'pressure_level' in ds.dims: ds = ds.isel(pressure_level=0)
        
        # 1. Get Actual Label (Ground Truth)
        actual_labels = ds.target.values[hour_idx]
        
        # 2. Get Model Prediction
        features = ['tp', 'z', 'q', 't', 'u', 'v', 'r']
        data = ds[features].to_array().transpose('valid_time','latitude','longitude','variable').values
        data = (data - data.min()) / (data.max() - data.min() + 1e-7)
        
        # Batch predict for the whole grid
        all_pixels = []
        for lat in range(ds.sizes['latitude']):
            for lon in range(ds.sizes['longitude']):
                all_pixels.append(data[hour_idx-24:hour_idx, lat, lon, :])
        
        preds = model.predict(np.array(all_pixels), batch_size=64, verbose=0)
        prob_map = preds[:, 3].reshape(ds.sizes['latitude'], ds.sizes['longitude'])

        # 3. SIDE-BY-SIDE PLOT
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Actual Data
        im1 = ax1.imshow(actual_labels, cmap='Blues', origin='upper')
        ax1.set_title(f"ACTUAL EVENTS (Ground Truth)\n{ds.valid_time.values[hour_idx]}")
        plt.colorbar(im1, ax=ax1, label='Label (0=Normal, 3=Burst)')

        # Model Prediction
        im2 = ax2.imshow(prob_map, cmap='YlOrRd', origin='upper')
        ax2.set_title("MODEL PREDICTION (Risk Map)")
        plt.colorbar(im2, ax=ax2, label='Probability Score')

        plt.tight_layout()
        plt.show()

# Example: Samej Event Validation
practical_validation(r'F:\Major Project\Project Working\Datasets\Hardened_Data\2024\Hardened_2024_07_21.nc', 740)