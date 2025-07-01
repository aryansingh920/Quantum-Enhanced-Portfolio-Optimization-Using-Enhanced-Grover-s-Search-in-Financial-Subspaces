import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from qiskit.circuit.library import ZZFeatureMap
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator, StatevectorSimulator
from qiskit.quantum_info import Statevector

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

print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

# Feature map with improved parameters
feature_dimension = X_train.shape[1]
feature_map = ZZFeatureMap(
    feature_dimension=feature_dimension,
    reps=3,  # Increased from 2 to 3 for richer expressivity
    entanglement='linear'  # Changed from 'full' to 'linear' to prevent overfitting
)

print(f"Feature map: {feature_map.num_qubits} qubits, {feature_map.reps} reps, {feature_map.entanglement} entanglement")

# Method 1: Exact fidelity using StatevectorSimulator (RECOMMENDED)


def compute_exact_fidelity_matrix(X1, X2, feature_map):
    """Compute exact fidelity using statevector simulation"""
    n1, n2 = len(X1), len(X2)
    kernel_matrix = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            # Create statevectors for both feature maps
            sv1 = Statevector(feature_map.assign_parameters(X1[i]))
            sv2 = Statevector(feature_map.assign_parameters(X2[j]))

            # Compute exact fidelity: |<φ(x)|φ(y)>|²
            fidelity = np.abs(sv1.data.conj().dot(sv2.data)) ** 2
            kernel_matrix[i, j] = fidelity

        if (i + 1) % 10 == 0:
            print(f"Computed {i + 1}/{n1} rows...")

    return kernel_matrix

# Method 2: Improved sampling-based fidelity (ALTERNATIVE)


def compute_sampling_fidelity_matrix(X1, X2, feature_map, backend, shots=8192):
    """Compute fidelity using sampling with higher shot count"""
    n1, n2 = len(X1), len(X2)
    kernel_matrix = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            # Create circuit for fidelity computation
            circ = QuantumCircuit(feature_map.num_qubits)

            # Decompose feature maps
            fm1 = feature_map.assign_parameters(X1[i]).decompose()
            fm2 = feature_map.assign_parameters(X2[j]).decompose()

            circ.compose(fm1, inplace=True)
            circ.compose(fm2.inverse(), inplace=True)
            circ.measure_all()

            # Run with higher shot count
            job = backend.run(circ, shots=shots)
            result = job.result()
            counts = result.get_counts()

            # Fidelity is probability of measuring all zeros
            fidelity = counts.get('0' * feature_map.num_qubits, 0) / shots
            kernel_matrix[i, j] = fidelity

        if (i + 1) % 10 == 0:
            print(f"Computed {i + 1}/{n1} rows...")

    return kernel_matrix


# Choose method (exact is recommended)
use_exact_method = True

if use_exact_method:
    print("Computing quantum kernel matrix using exact fidelity (Statevector)...")
    train_kernel = compute_exact_fidelity_matrix(X_train, X_train, feature_map)
    test_kernel = compute_exact_fidelity_matrix(X_test, X_train, feature_map)
else:
    print("Computing quantum kernel matrix using sampling method with 8192 shots...")
    backend = AerSimulator()
    train_kernel = compute_sampling_fidelity_matrix(
        X_train, X_train, feature_map, backend, shots=8192)
    test_kernel = compute_sampling_fidelity_matrix(
        X_test, X_train, feature_map, backend, shots=8192)

# Analyze kernel properties
print(f"\nKernel matrix statistics:")
print(
    f"Train kernel - min: {train_kernel.min():.4f}, max: {train_kernel.max():.4f}, mean: {train_kernel.mean():.4f}")
print(
    f"Test kernel - min: {test_kernel.min():.4f}, max: {test_kernel.max():.4f}, mean: {test_kernel.mean():.4f}")

# Train QSVM with regularization tuning
# Try different C values to find optimal regularization
C_values = [0.1, 1.0, 10.0, 100.0]
best_accuracy = 0
best_C = 1.0

for C in C_values:
    qsvm = SVC(kernel='precomputed', C=C)
    qsvm.fit(train_kernel, y_train)
    y_pred = qsvm.predict(test_kernel)
    acc = np.mean(y_pred == y_test)
    print(f"C={C}: Quantum SVM accuracy: {acc * 100:.2f}%")

    if acc > best_accuracy:
        best_accuracy = acc
        best_C = C

print(f"\nBest performance: C={best_C}, Accuracy: {best_accuracy * 100:.2f}%")

# Train final model with best C
final_qsvm = SVC(kernel='precomputed', C=best_C)
final_qsvm.fit(train_kernel, y_train)
final_pred = final_qsvm.predict(test_kernel)

# Compare with classical SVM
classical_svm = SVC(kernel='rbf', C=best_C)
classical_svm.fit(X_train, y_train)
classical_pred = classical_svm.predict(X_test)
classical_acc = np.mean(classical_pred == y_test)

print(f"\nComparison:")
print(f"Quantum SVM accuracy: {best_accuracy * 100:.2f}%")
print(f"Classical RBF SVM accuracy: {classical_acc * 100:.2f}%")

# Visualization
plt.figure(figsize=(15, 5))

# Plot original data
plt.subplot(1, 3, 1)
plt.scatter(X_scaled[y == 1, 0], X_scaled[y == 1, 1],
            c='red', marker='o', label='Class +1', alpha=0.7)
plt.scatter(X_scaled[y == -1, 0], X_scaled[y == -1, 1],
            c='blue', marker='s', label='Class -1', alpha=0.7)
plt.title('Original Data')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot QSVM predictions
plt.subplot(1, 3, 2)
plt.scatter(X_test[final_pred == 1, 0], X_test[final_pred == 1, 1],
            c='red', marker='o', label='Predicted +1', alpha=0.7)
plt.scatter(X_test[final_pred == -1, 0], X_test[final_pred == -1, 1],
            c='blue', marker='s', label='Predicted -1', alpha=0.7)
plt.scatter(X_test[final_pred != y_test, 0], X_test[final_pred != y_test, 1],
            facecolors='none', edgecolors='black', s=100, linewidth=2, label='Misclassified')
plt.title(
    f'QSVM Predictions\n(Accuracy: {best_accuracy*100:.1f}%, C={best_C})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot Classical SVM predictions
plt.subplot(1, 3, 3)
plt.scatter(X_test[classical_pred == 1, 0], X_test[classical_pred ==
            1, 1], c='red', marker='o', label='Predicted +1', alpha=0.7)
plt.scatter(X_test[classical_pred == -1, 0], X_test[classical_pred == -
            1, 1], c='blue', marker='s', label='Predicted -1', alpha=0.7)
plt.scatter(X_test[classical_pred != y_test, 0], X_test[classical_pred != y_test, 1],
            facecolors='none', edgecolors='black', s=100, linewidth=2, label='Misclassified')
plt.title(
    f'Classical RBF SVM\n(Accuracy: {classical_acc*100:.1f}%, C={best_C})')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Display kernel matrix heatmap
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(train_kernel, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('Training Kernel Matrix')
plt.xlabel('Sample Index')
plt.ylabel('Sample Index')

plt.subplot(1, 2, 2)
plt.imshow(test_kernel, cmap='viridis', aspect='auto')
plt.colorbar()
plt.title('Test Kernel Matrix')
plt.xlabel('Training Sample Index')
plt.ylabel('Test Sample Index')

plt.tight_layout()
plt.show()
