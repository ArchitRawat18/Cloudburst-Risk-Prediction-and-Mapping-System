import os
import numpy as np
import xarray as xr
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ================= ⚙️ CONFIGURATION =================
DATA_DIR = r'F:\Major Project\Project Working\Datasets\Hardened_Data'
MODEL_DIR = r'F:\Major Project\Project Working\Modelling_i2'
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "lstm_temporal_engine_i2.keras")

FEATURES = ['tp', 'z', 'q', 't', 'u', 'v', 'r', 'z_dip', 'r_spike', 'tp_trend']
LOOKBACK = 24  
SAMPLES_PER_FILE = 5 
# ====================================================

class SpatialTemporalGenerator(tf.keras.utils.Sequence):
    def __init__(self, years, batch_size=32):
        super().__init__()
        self.files = []
        for year in years:
            path = os.path.join(DATA_DIR, str(year))
            if os.path.exists(path):
                self.files.extend([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.nc')])
        self.batch_size = batch_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        try:
            with xr.open_dataset(self.files[idx]) as ds:
                if 'pressure_level' in ds.dims: ds = ds.isel(pressure_level=0)
                
                # [cite_start]1. ROBUST FEATURE ENGINEERING [cite: 140, 476, 552]
                z_dip = ds['z'].diff('valid_time').fillna(0)
                r_spike = ds['r'].diff('valid_time').fillna(0)
                tp_trend = ds['tp'].rolling(valid_time=3).mean().fillna(0)
                ds = ds.assign(z_dip=z_dip, r_spike=r_spike, tp_trend=tp_trend)
                
                # 2. EXTRACT & CLEAN DATA
                data = ds[FEATURES].to_array().transpose('valid_time', 'latitude', 'longitude', 'variable').values
                # Kill any NaNs or Infs immediately
                data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
                
                # 3. ROBUST NORMALIZATION
                d_min = data.min(axis=(0,1,2), keepdims=True)
                d_max = data.max(axis=(0,1,2), keepdims=True)
                denom = (d_max - d_min) + 1e-7
                data = (data - d_min) / denom
                
                labels = np.nan_to_num(ds.target.values, nan=0).astype(int)
                X, y = [], []

                for _ in range(SAMPLES_PER_FILE):
                    lat = np.random.randint(0, data.shape[1])
                    lon = np.random.randint(0, data.shape[2])
                    
                    for t in range(LOOKBACK, data.shape[0]):
                        sample = data[t-LOOKBACK:t, lat, lon, :]
                        # Final check: if this specific sample contains any NaN, skip it
                        if not np.isnan(sample).any():
                            X.append(sample)
                            y.append(labels[t, lat, lon])
                
                return np.array(X), np.array(y)
        except Exception as e:
            return np.zeros((1, LOOKBACK, len(FEATURES))), np.zeros((1,))

# --- 🧠 STABLE ARCHITECTURE ---
def create_lstm_i2_final():
    model = models.Sequential([
        layers.Input(shape=(LOOKBACK, len(FEATURES))),
        layers.LSTM(128, return_sequences=True, kernel_initializer='glorot_uniform'),
        layers.Dropout(0.3),
        layers.LSTM(64),
        layers.Dense(64, activation='relu'),
        layers.Dense(4, activation='softmax')
    ])
    # Tighter clipping and ultra-stable learning rate
    opt = tf.keras.optimizers.Adam(learning_rate=0.00005, clipnorm=0.1) 
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

train_gen = SpatialTemporalGenerator(range(2010, 2022))
model = create_lstm_i2_final()
CLASS_WEIGHTS = {0: 1.0, 1: 3.0, 2: 10.0, 3: 50.0} # Balanced weights for stability

print("🛡️ Starting Indestructible LSTM Training (Iteration 2)...")
model.fit(
    train_gen, 
    epochs=40, 
    class_weight=CLASS_WEIGHTS,
    callbacks=[
        callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_best_only=False, verbose=1),
        callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    ]
)