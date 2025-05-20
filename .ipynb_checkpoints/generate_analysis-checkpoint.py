import matplotlib.pyplot as plt
import numpy as np
import joblib
import tensorflow as tf

# Load model and scaler
model = tf.keras.models.load_model("../models/covid_lstm_model.h5")
scaler = joblib.load("../models/scaler.pkl")

# Load test data
X_test, y_test = np.load("../data/X_test.npy"), np.load("../data/y_test.npy")

# Make predictions
y_pred = model.predict(X_test)

# Inverse scale predictions
y_test_unscaled = scaler.inverse_transform(y_test)
y_pred_unscaled = scaler.inverse_transform(y_pred)

# Plot results
plt.figure(figsize=(10, 5))
plt.plot(y_test_unscaled, label='Actual Cases', color='blue')
plt.plot(y_pred_unscaled, label='Predicted Cases', color='red', linestyle='dashed')
plt.xlabel("Days")
plt.ylabel("COVID-19 Cases")
plt.legend()
plt.title("COVID-19 Case Predictions vs Actual")
plt.savefig("../reports/analysis.png")
plt.show()

# Log results
with open("../reports/results.log", "w") as log_file:
    log_file.write(f"Mean Absolute Error: {np.mean(np.abs(y_test_unscaled - y_pred_unscaled))}\n")
    log_file.write(f"Root Mean Squared Error: {np.sqrt(np.mean((y_test_unscaled - y_pred_unscaled) ** 2))}\n")
    log_file.write("Prediction analysis completed.")
