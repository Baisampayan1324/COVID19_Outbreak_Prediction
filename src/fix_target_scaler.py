# # fix_scaler_from_test.py
# import numpy as np
# import pickle
# import os
# from sklearn.preprocessing import MinMaxScaler

# # Load target test data
# y_test = np.load("data/y_test.npy").reshape(-1, 1)

# # Fit scaler on y_test
# target_scaler = MinMaxScaler()
# target_scaler.fit(y_test)

# # Save to models directory
# os.makedirs("models", exist_ok=True)
# with open("models/scaler.pkl", "wb") as f:
#     pickle.dump(target_scaler, f)

# print("✅ Scaler fitted using y_test and saved to 'models/scaler.pkl'.")

# fix_scaler_from_test.py
import numpy as np
import pickle
import os
from sklearn.preprocessing import MinMaxScaler

# Constants
TEST_PATH = "datasets/y_test.npy"
SCALER_PATH = "models/scaler.pkl"

# Ensure test data exists
if not os.path.exists(TEST_PATH):
    raise FileNotFoundError(f"❌ Test data not found at {TEST_PATH}")

# Load target test data
y_test = np.load(TEST_PATH).reshape(-1, 1)

# Fit scaler on y_test
target_scaler = MinMaxScaler()
target_scaler.fit(y_test)

# Save scaler
os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
with open(SCALER_PATH, "wb") as f:
    pickle.dump(target_scaler, f)

print(f"✅ Scaler fitted on y_test and saved to '{SCALER_PATH}'.")
