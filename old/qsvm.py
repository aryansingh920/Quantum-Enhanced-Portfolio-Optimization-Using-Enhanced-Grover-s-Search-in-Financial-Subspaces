"""
Created on 01/07/2025

@author: Aryan

Filename: qsvm.py

Relative Path: qsvm.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# Load your data
ticker = "JPM"
df = pd.read_csv(f"ticker/{ticker}/{ticker}_data.csv", parse_dates=['Date'])
df = df.dropna()

# --- Select features and labels ---
features = ['ma5_scaled', 'ma20_scaled', 'rsi_scaled', 'volatility_scaled']
X = df[features].values
y = df['regime'].values.astype(int)

# Ensure binary class for QSVM
y = np.where(y == 0, -1, 1)

# Split and scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42)

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Feature map
feature_map = ZZFeatureMap(
    feature_dimension=X_train.shape[1], reps=3, entanglement='linear')

# Kernel function


def compute_exact_fidelity_matrix(X1, X2, feature_map):
    n1, n2 = len(X1), len(X2)
    kernel_matrix = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            sv1 = Statevector(feature_map.assign_parameters(X1[i]))
            sv2 = Statevector(feature_map.assign_parameters(X2[j]))
            fidelity = np.abs(sv1.data.conj().dot(sv2.data)) ** 2
            kernel_matrix[i, j] = fidelity
        if (i + 1) % 10 == 0:
            print(f"Computed {i + 1}/{n1} rows...")
    return kernel_matrix


# Compute kernel
train_kernel = compute_exact_fidelity_matrix(X_train, X_train, feature_map)
test_kernel = compute_exact_fidelity_matrix(X_test, X_train, feature_map)

# QSVM training and evaluation
C_values = [0.1, 1.0, 10.0]
best_accuracy = 0
best_C = 1.0

for C in C_values:
    model = SVC(kernel='precomputed', C=C)
    model.fit(train_kernel, y_train)
    acc = np.mean(model.predict(test_kernel) == y_test)
    print(f"C={C} => QSVM Accuracy: {acc * 100:.2f}%")
    if acc > best_accuracy:
        best_accuracy = acc
        best_C = C

print(f"\nBest QSVM Accuracy: {best_accuracy * 100:.2f}% at C={best_C}")

# Classical SVM for comparison
clf = SVC(kernel='rbf', C=best_C)
clf.fit(X_train, y_train)
classical_acc = np.mean(clf.predict(X_test) == y_test)
print(f"Classical RBF SVM Accuracy: {classical_acc * 100:.2f}%")
