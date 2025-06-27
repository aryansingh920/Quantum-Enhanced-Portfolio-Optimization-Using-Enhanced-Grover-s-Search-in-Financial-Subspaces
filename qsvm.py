import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from qiskit.circuit.library import ZZFeatureMap
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# Set seed manually
seed = 42
np.random.seed(seed)

# Create data
X, y = make_moons(n_samples=100, noise=0.1, random_state=seed)
y = 2 * y - 1

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=seed)

# Feature map
feature_dimension = X_train.shape[1]
feature_map = ZZFeatureMap(
    feature_dimension=feature_dimension, reps=2, entanglement='full')

# Compute kernel matrix manually


def compute_kernel_matrix(X1, X2, feature_map, backend):
    n1, n2 = len(X1), len(X2)
    kernel_matrix = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            # Create circuit for fidelity computation
            circ = QuantumCircuit(feature_map.num_qubits)
            circ.append(feature_map.bind_parameters(
                X1[i]), list(range(feature_map.num_qubits)))
            circ.append(feature_map.bind_parameters(
                X2[j]).inverse(), list(range(feature_map.num_qubits)))
            circ.measure_all()
            # Run on backend
            job = backend.run(circ, shots=1000)
            result = job.result()
            counts = result.get_counts()
            fidelity = counts.get('0' * feature_map.num_qubits, 0) / 1000
            kernel_matrix[i, j] = fidelity
    return kernel_matrix


backend = AerSimulator()

print("Computing quantum kernel matrix...")
train_kernel = compute_kernel_matrix(X_train, X_train, feature_map, backend)
test_kernel = compute_kernel_matrix(X_test, X_train, feature_map, backend)

qsvm = SVC(kernel='precomputed')
qsvm.fit(train_kernel, y_train)

y_pred = qsvm.predict(test_kernel)
acc = np.mean(y_pred == y_test)
print(f"Quantum SVM accuracy: {acc * 100:.2f}%")
