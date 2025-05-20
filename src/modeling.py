import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_absolute_error, mean_squared_error
from math import sqrt
import joblib

from preprocessing import load_data, scale_features

def create_sequences(data_values, target_values, seq_length):
    X, y = [], []
    for i in range(len(data_values) - seq_length):
        X.append(data_values[i:i+seq_length])
        y.append(target_values[i+seq_length])
    return np.array(X), np.array(y)

def train_model():
    data = load_data("datasets\covid.csv")
    features = ["total_cases", "total_deaths", "total_tests", "population"]
    target = "new_cases"

    data_scaled, scaler = scale_features(data, features)
    data_values = data_scaled[features].values
    target_values = data_scaled[target].values.reshape(-1, 1)

    seq_length = 10
    X, y = create_sequences(data_values, target_values, seq_length)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    np.save("datasets/X_test.npy", X_test)
    np.save("datasets/y_test.npy", y_test)

    model = Sequential([
        LSTM(50, activation='relu', return_sequences=True, input_shape=(seq_length, len(features))),
        Dropout(0.2),
        LSTM(50, activation='relu'),
        Dropout(0.2),
        Dense(1)
    ])

    model.compile(optimizer='adam', loss='mse')

    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = sqrt(mse)

    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, RMSE: {rmse:.4f}")

    model.save("../models/covid_lstm_model.h5")
    joblib.dump(scaler, "../models/scaler.pkl")

if __name__ == "__main__":
    train_model()
