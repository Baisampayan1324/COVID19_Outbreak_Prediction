import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

def load_data(filepath):
    covid = pd.read_csv("P:\COVID-19 Outbreak Prediction\datasets\covid.csv")

    # Imputation strategy
    impute_strategy = {
        'total_cases_per_million': 'mean',
        'new_cases_per_million': 'mean',
        'total_deaths_per_million': 'mean',
        'new_deaths_per_million': 'mean',
        'total_tests': 'median',
        'new_tests': 'median',
        'stringency_index': 'mean',
        'population': 'mean',
        'population_density': 'mean',
        'median_age': 'mean',
        'aged_65_older': 'mean',
        'gdp_per_capita': 'mean'
    }
    for col, strategy in impute_strategy.items():
        if col in covid.columns:
            if strategy == 'mean':
                covid[col].fillna(covid[col].mean(), inplace=True)
            elif strategy == 'median':
                covid[col].fillna(covid[col].median(), inplace=True)

    covid.fillna(method='ffill', inplace=True)
    covid["date"] = pd.to_datetime(covid["date"])

    if 'new_cases' in covid.columns:
        covid['new_cases_avg'] = covid['new_cases'].rolling(window=7).mean()
        covid['new_cases_avg'].fillna(covid['new_cases_avg'].mean(), inplace=True)

    return covid

def scale_features(df, feature_cols, scaler_output_path="../models/scaler.pkl"):
    scaler = MinMaxScaler()
    df_scaled = df.copy()
    df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])

    # Save the scaler object for later inverse_transform
    os.makedirs(os.path.dirname(scaler_output_path), exist_ok=True)
    with open(scaler_output_path, "wb") as f:
        pickle.dump(scaler, f)

    return df_scaled, scaler
