"""
Created on 17/07/2025

@author: Aryan

Filename: data_utils.py

Relative Path: dynamic-oseq-qsvm/data_utils.py
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_financial_data(csv_path="ticker/GOOGL/GOOGL_data.csv"):
    """
    Loads the user CSV.  Falls back to synthetic data if the file
    isn't found, exactly like your original demo.
    """
    try:
        df = pd.read_csv(csv_path, parse_dates=['Date']).dropna()
        features = ['ma5_scaled', 'ma20_scaled',
                    'rsi_scaled', 'volatility_scaled']
        X = df[features].values
        y = df['regime'].values.astype(int)
        y = np.where(y == 0, -1, 1)
    except Exception:
        # --- synthetic fallback (identical logic) ---
        np.random.seed(42)
        n_samples = 355
        features = ['ma5_scaled', 'ma20_scaled',
                    'rsi_scaled', 'volatility_scaled']
        X = np.random.randn(n_samples, 4)
        y = np.where(X[:, 0] + X[:, 1] > 0, 1, -1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return train_test_split(X_scaled, y, test_size=0.3, random_state=42), features
