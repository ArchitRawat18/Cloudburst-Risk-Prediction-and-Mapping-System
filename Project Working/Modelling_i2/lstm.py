import os
import numpy as np
import xarray as xr
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

# ================= ⚙️ CONFIGURATION =================
DATA_DIR = r'F:\Major Project\Project Working\Datasets\Hardened_Data'
MODEL_DIR = r'F:\Major Project\Project Working\Modelling_i2'
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "lstm_temporal_engine.keras")

# 10 Features: 7 Original + 3 Engineered
FEATURES = ['tp', 'z', 'q', 't', 'u', 'v', 'r', 'z_dip', 'r_spike', 'tp_trend']
LOOKBACK = 24  
# ====================================================

class HybridLSTMGenerator(tf.keras.utils.Sequence):
    def __init__(self, years, batch_size=32):
        self.files = []
        for year in years:
            path = os.path.join(DATA_DIR, str(year))
            if os.path.exists(path):
                self.files.extend([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.nc')])
        self.batch_size = batch_size

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with xr.open_dataset(self.files[idx]) as ds:
            if 'pressure_level' in ds.dims: ds = ds.isel(pressure_level=0)
            
            # Engineered Features on the fly 
            z_dip = ds['z'].diff('valid_time').fillna(0)
            r_spike = ds['r'].diff('valid_time').fillna(0)
            tp_trend = ds['tp'].rolling(valid_time=3).mean().fillna(0)
            ds = ds.assign(z_dip=z_dip, r_spike=r_spike, tp_trend=tp_trend)
            
            # Global Scaling (0-1) [cite: 1, 5]
            data = ds[FEATURES].to_array().transpose('valid_time', 'latitude', 'longitude', 'variable').values
            data = (data - data.min()) / (data.max() - data.min() + 1e-7)
            
            labels = ds.target.values
            X, y = [], []
            # We sample a random coordinate for each file to ensure spatial variety 
            lat, lon = np.random.randint(0, data.shape[1]), np.random.randint(0, data.shape[2])
            
            for t in range(LOOKBACK, data.shape[0]):
                X.append(data[t-LOOKBACK:t, lat, lon, :])
                y.append(labels[t, lat, lon])
                
            return np.array(X), np.array(y)

# --- 🧠 LSTM Architecture --- 
def create_lstm():
    model = models.Sequential([
        layers.Input(shape=(LOOKBACK, len(FEATURES))),
        layers.LSTM(128, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(64),
        layers.Dense(32, activation='relu'),
        layers.Dense(4, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Train on the same 2010-2021 window 
train_gen = HybridLSTMGenerator(range(2010, 2022))
model = create_lstm()

print("🚀 Starting LSTM Temporal Engine Training...")
model.fit(train_gen, epochs=30, callbacks=[callbacks.ModelCheckpoint(CHECKPOINT_PATH, save_best_only=True)])