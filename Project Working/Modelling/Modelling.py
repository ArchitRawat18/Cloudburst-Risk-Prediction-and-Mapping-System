import os
import numpy as np
import xarray as xr
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ================= ⚙️ CONFIGURATION =================
DATA_DIR = r'F:\Major Project\Project Working\Datasets\Hardened_Data'
MODEL_DIR = r'F:\Major Project\Project Working\Modelling'
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "cloudburst_model.keras")
LOG_PATH = os.path.join(MODEL_DIR, "training_history.csv")

# Feature list based on your NetCDF variables
FEATURES = ['tp', 'z', 'q', 't', 'u', 'v', 'r'] 
LOOKBACK = 24  
BATCH_SIZE = 64 # Conservative for 4GB VRAM
# ====================================================

class CloudburstGenerator(tf.keras.utils.Sequence):
    """Memory-efficient generator with corrected axes and Keras 3 support"""
    def __init__(self, years, batch_size=64, **kwargs):
        # Fix for UserWarning
        super().__init__(**kwargs) 
        self.files = []
        for year in years:
            y_path = os.path.join(DATA_DIR, str(year))
            if os.path.exists(y_path):
                self.files.extend([os.path.join(y_path, f) for f in os.listdir(y_path) if f.endswith('.nc')])
        self.batch_size = batch_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            with xr.open_dataset(self.files[idx]) as ds:
                # Handle pressure_level by selecting the first one (index 0)
                # This ensures we don't have a 5th dimension crashing the axes
                if 'pressure_level' in ds.dims:
                    ds = ds.isel(pressure_level=0)
                
                # Convert to array and use ellipsis (...) to handle remaining dims safely
                da = ds[FEATURES].to_array()
                
                # Explicitly transpose naming all required dimensions
                data = da.transpose('valid_time', 'latitude', 'longitude', 'variable').values
                
                # Normalization (0 to 1)
                data = (data - data.min()) / (data.max() - data.min() + 1e-7)
                
                labels = ds.target.values 
                
                X, y = [], []
                for t in range(LOOKBACK, data.shape[0]):
                    # Input shape: (Time, Features) for the center pixel
                    X.append(data[t-LOOKBACK:t, 0, 0, :]) 
                    y.append(labels[t, 0, 0])
                
                return np.array(X), np.array(y)
        except Exception as e:
            print(f"❌ Fixed Axis Error in {os.path.basename(self.files[idx])}: {e}")
            return np.zeros((1, LOOKBACK, len(FEATURES))), np.zeros((1,))

# --- 🧠 Model Architecture ---
def create_model(input_shape):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        # CuDNN Optimized LSTM for RTX 3050
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.3),
        layers.LSTM(64),
        layers.Dense(32, activation='relu'),
        layers.Dense(4, activation='softmax') # Labels 0, 1, 2, 3
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# --- ⏯️ Execution Logic (Pause/Resume) ---
train_gen = CloudburstGenerator(range(2010, 2022))
val_gen = CloudburstGenerator([2022, 2023])

if os.path.exists(CHECKPOINT_PATH):
    print(f"🔄 Resuming Training from {CHECKPOINT_PATH}")
    model = models.load_model(CHECKPOINT_PATH)
else:
    print("🆕 Initializing New Model...")
    model = create_model((LOOKBACK, len(FEATURES)))

# Callbacks for safety
model_callbacks = [
    callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_best_only=False, verbose=1),
    callbacks.CSVLogger(LOG_PATH, append=True),
    callbacks.EarlyStopping(patience=5, restore_best_weights=True)
]



print("🚀 Starting Training Loop...")
model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=model_callbacks
)