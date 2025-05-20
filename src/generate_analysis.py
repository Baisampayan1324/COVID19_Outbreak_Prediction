import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K

def mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

def generate_analysis(X_test, y_test, 
                      model_path='models/covid_lstm_model.h5', 
                      scaler_path='models/scaler.pkl',
                      reports_dir='reports'):

    os.makedirs(reports_dir, exist_ok=True)

    # Load model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = load_model(model_path, custom_objects={'mse': mse})

    # Load scaler
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    with open(scaler_path, 'rb') as f:
        target_scaler = pickle.load(f)

    # Predict
    y_pred = model.predict(X_test)

    # Force shapes to 2D: (n_samples, 1)
    y_pred = np.reshape(y_pred, (-1, 1))
    y_test = np.reshape(y_test, (-1, 1))

    # Safety checks
    scaler_feature_count = target_scaler.scale_.shape[0]
    if y_pred.shape[1] != scaler_feature_count:
        raise ValueError(
            f"Shape mismatch: scaler expects {scaler_feature_count} feature(s), "
            f"but prediction has shape {y_pred.shape}"
        )

    # Inverse transform
    try:
        y_pred_inv = target_scaler.inverse_transform(y_pred).flatten()
        y_test_inv = target_scaler.inverse_transform(y_test).flatten()
    except Exception as e:
        raise RuntimeError(f"Failed inverse transform: {str(e)}")

    # Metrics
    mae = mean_absolute_error(y_test_inv, y_pred_inv)
    mse_val = mean_squared_error(y_test_inv, y_pred_inv)
    rmse = np.sqrt(mse_val)
    r2 = r2_score(y_test_inv, y_pred_inv)

    # Log text
    metrics_text = (
        "Model Evaluation Metrics:\n"
        f"  MAE:   {mae:.4f}\n"
        f"  MSE:   {mse_val:.4f}\n"
        f"  RMSE:  {rmse:.4f}\n"
        f"  R²:    {r2:.4f}\n"
    )
    print(metrics_text)

    with open(os.path.join(reports_dir, 'evaluation_metrics.txt'), 'w') as f:
        f.write(metrics_text)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_inv, label='Actual', linewidth=2)
    plt.plot(y_pred_inv, label='Predicted', linewidth=2, alpha=0.7)
    plt.title('COVID-19: Actual vs Predicted Cases')
    plt.xlabel('Sample Index')
    plt.ylabel('New Cases')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_path = os.path.join(reports_dir, 'prediction_vs_actual.png')
    plt.savefig(plot_path)
    plt.show()
    print(f"Saved plot to: {plot_path}")

if __name__ == "__main__":
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    generate_analysis(X_test, y_test)
