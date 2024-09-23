import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# Load the dataset from the Excel file
data = pd.read_excel('nv-hpl.xlsx')  # Update with your file path

# Select relevant columns and hardcode the configuration
data['os_version'] = 'v1'
data['vm_sku'] = 'A'

# Prepare input features (parameters + configuration)
param_features = ['Ns', 'NB']
config_features = ['os_version', 'vm_sku']
X_params = data[param_features]
X_config = pd.get_dummies(data[config_features])  # One-hot encode configurations
X = pd.concat([X_params, X_config], axis=1)

# Target feature (assumes 'score' is already computed in your dataset)
y = data['Score']  # Make sure 'score' is present in your dataset

# Scale the input parameters (important for neural networks)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to numpy arrays for TensorFlow
X_train_np = np.array(X_train)
y_train_np = np.array(y_train)

# Define the neural network model using Keras
def create_model(input_shape):
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)
    ])
    return model

# Use MirroredStrategy for multi-GPU training
strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    # Create and compile the model inside the strategy scope
    model = create_model(X_train.shape[1])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model using the distributed strategy
history = model.fit(X_train_np, y_train_np, epochs=200, batch_size=32, verbose=1)

# Save the trained model to a file
model.save('score_predictor_tf.h5')

