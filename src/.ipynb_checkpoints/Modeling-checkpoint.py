import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import pandas as pd
from preprocessing import load_data, scale_features

# Load and preprocess data
data = load_data("../data/covid.csv")
features = ["total_cases", "total_deaths", "total_tests", "population"]

data, scaler = scale_features(data, features)
target = "new_cases"

data_values = data[features].values
target_values = data[target].values.reshape(-1, 1)

# Prepare sequences for LSTM
seq_length = 10
X, y = [], []

for i in range(len(data_values) - seq_length):
    X.append(data_values[i:i+seq_length])
    y.append(target_values[i+seq_length])

X, y = np.array(X), np.array(y)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define LSTM model
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, len(features))),
    Dropout(0.2),
    LSTM(50, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"Mean Absolute Error: {mae}")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")

# Save model and scaler
model.save("../models/covid_lstm_model.h5")
joblib.dump(scaler, "../models/scaler.pkl")
