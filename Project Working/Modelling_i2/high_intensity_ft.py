import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the gold data
data = np.load(r'F:\Major Project\Project Working\Modelling\balanced_training_data.npz')
X, y = data['x'], data['y']

# Load the model
model = load_model(r'F:\Major Project\Project Working\Modelling\cloudburst_model.keras')

# Massive weights to ensure it finally "cares"
class_weights = {0: 1.0, 1: 2.0, 2: 10.0, 3: 50.0}

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), 
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("🚀 Starting High-Intensity Fine-Tuning on Balanced Data...")
model.fit(X, y, epochs=30, batch_size=32, class_weight=class_weights)
model.save(r'F:\Major Project\Project Working\Modelling\cloudburst_final_v2.keras')