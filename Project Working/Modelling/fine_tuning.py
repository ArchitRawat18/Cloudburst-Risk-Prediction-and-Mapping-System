import os
import numpy as np
import xarray as xr
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, callbacks

# ================= ⚙️ CONFIGURATION =================
DATA_DIR = r'F:\Major Project\Project Working\Datasets\Hardened_Data'
MODEL_DIR = r'F:\Major Project\Project Working\Modelling'
# We load the existing 50-epoch model
BASE_MODEL_PATH = os.path.join(MODEL_DIR, "cloudburst_model.keras")
# We save the new weights separately to compare
FINE_TUNED_PATH = os.path.join(MODEL_DIR, "cloudburst_finetuned.keras")

FEATURES = ['tp', 'z', 'q', 't', 'u', 'v', 'r'] 
LOOKBACK = 24
BATCH_SIZE = 64
# ====================================================

class CloudburstGenerator(tf.keras.utils.Sequence):
    """Re-defining the generator so this script is self-contained"""
    def __init__(self, years, batch_size=64, **kwargs):
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
        # 1. Load the original batch
        original_x, original_y = super().__getitem__(idx)
        
        # 2. Find all 'Extreme' and 'Cloudburst' indices in the train set
        # For this final run, we will simply filter the batch to ensure 
        # that if a file HAS a cloudburst, we prioritize those sequences.
        if 3 in original_y or 2 in original_y:
            interesting_idx = np.where((original_y == 3) | (original_y == 2))[0]
            normal_idx = np.where(original_y == 0)[0][:len(interesting_idx)]
            
            final_idx = np.concatenate([interesting_idx, normal_idx])
            return original_x[final_idx], original_y[final_idx]
        
        return original_x, original_y

# --- 🚀 Execution Logic ---
if __name__ == "__main__":
    # 1. Initialize Generators
    train_gen = CloudburstGenerator(range(2010, 2022))
    val_gen = CloudburstGenerator([2022, 2023])

    # 2. Load Model
    if os.path.exists(BASE_MODEL_PATH):
        print(f"🔄 Loading base model from {BASE_MODEL_PATH}...")
        model = load_model(BASE_MODEL_PATH)
    else:
        print("❌ Error: Base model not found! Check your path.")
        exit()

    # 3. Apply Extreme Class Weights
    # We prioritize Label 3 by 10,000x to overcome 0.001% imbalance
    class_weights = {0: 1.0, 1: 50.0, 2: 500.0, 3: 10000.0}

    # 4. Re-compile with Lower Learning Rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # 5. Fine-Tuning Callbacks
    fine_tune_callbacks = [
        callbacks.ModelCheckpoint(FINE_TUNED_PATH, save_best_only=True, monitor='val_loss'),
        callbacks.CSVLogger(os.path.join(MODEL_DIR, "finetuning_history.csv"), append=True),
        callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]

    

    print("⚖️ Starting 20-Epoch Fine-Tuning with Heavy Class Weights...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=20,
        class_weight=class_weights,
        callbacks=fine_tune_callbacks
    )
    print(f"✅ Fine-tuning complete. Saved to: {FINE_TUNED_PATH}")