import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
import itertools
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class QuantumSVM:
    """
    Custom Quantum Support Vector Machine implementation from scratch
    """

    def __init__(self, n_qubits=2, entanglement_type='linear', depth=1):
        self.n_qubits = n_qubits
        self.entanglement_type = entanglement_type
        self.depth = depth
        self.classifier = None
        self.X_train = None
        self.y_train = None
        self.kernel_matrix_train = None
        self.feature_maps = []

    def create_feature_map(self, x):
        """
        Create quantum feature map circuit
        """
        qc = QuantumCircuit(self.n_qubits)

        # Data encoding layers
        for d in range(self.depth):
            # Hadamard gates for superposition
            for i in range(self.n_qubits):
                qc.h(i)

            # Rotation gates with data encoding
            for i in range(min(len(x), self.n_qubits)):
                qc.rz(x[i] * np.pi, i)
                qc.ry(x[i] * np.pi/2, i)

            # Entanglement layer
            if self.entanglement_type == 'linear':
                for i in range(self.n_qubits - 1):
                    qc.cx(i, i + 1)
                    if i < len(x) - 1:
                        qc.rz(x[i] * x[i+1] * np.pi, i + 1)
                    qc.cx(i, i + 1)
            elif self.entanglement_type == 'circular':
                for i in range(self.n_qubits):
                    qc.cx(i, (i + 1) % self.n_qubits)
            elif self.entanglement_type == 'full':
                for i in range(self.n_qubits):
                    for j in range(i + 1, self.n_qubits):
                        qc.cx(i, j)

        return qc

    def compute_fidelity(self, x1, x2):
        """
        Compute quantum kernel entry (fidelity) between two data points
        """
        try:
            # Create feature maps
            qc1 = self.create_feature_map(x1)
            qc2 = self.create_feature_map(x2)

            # Get statevectors
            psi1 = Statevector(qc1)
            psi2 = Statevector(qc2)

            # Compute fidelity |<psi1|psi2>|^2
            fidelity = np.abs(np.conj(psi1.data) @ psi2.data) ** 2
            return fidelity
        except Exception as e:
            print(f"Error computing fidelity: {e}")
            return 0.0

    def compute_kernel_matrix(self, X1, X2=None):
        """
        Compute full quantum kernel matrix
        """
        if X2 is None:
            X2 = X1

        kernel = np.zeros((len(X1), len(X2)))

        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                kernel[i, j] = self.compute_fidelity(x1, x2)

        return kernel

    def fit(self, X, y):
        """
        Train the quantum SVM
        """
        self.X_train = X
        self.y_train = y

        # Compute quantum kernel matrix
        print("Computing quantum kernel matrix...")
        self.kernel_matrix_train = self.compute_kernel_matrix(X)

        # Train classical SVM with precomputed quantum kernel
        self.classifier = SVC(kernel='precomputed', C=1.0)
        self.classifier.fit(self.kernel_matrix_train, y)

        return self

    def predict(self, X):
        """
        Make predictions
        """
        if self.classifier is None:
            raise ValueError("Model not fitted yet!")

        # Compute kernel matrix between test and training data
        kernel_matrix_test = self.compute_kernel_matrix(X, self.X_train)

        return self.classifier.predict(kernel_matrix_test)

    def score(self, X, y):
        """
        Compute accuracy score
        """
        predictions = self.predict(X)
        return accuracy_score(y, predictions)


def visualize_bloch_sphere_representation(qsvm, X_sample, title="Quantum States on Bloch Sphere"):
    """
    Visualize quantum states on Bloch sphere (for single qubit case)
    """
    if qsvm.n_qubits != 1:
        print("Bloch sphere visualization only available for single qubit systems")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Draw Bloch sphere
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))

    ax.plot_surface(x_sphere, y_sphere, z_sphere, alpha=0.1, color='lightblue')

    # Plot quantum states
    colors = ['red', 'blue']
    for i, x in enumerate(X_sample[:10]):  # Limit to first 10 samples
        qc = qsvm.create_feature_map(x)
        psi = Statevector(qc)

        # Extract Bloch vector components
        rho = psi.data
        sx = 2 * np.real(np.conj(rho[0]) * rho[1])
        sy = 2 * np.imag(np.conj(rho[0]) * rho[1])
        sz = np.abs(rho[0])**2 - np.abs(rho[1])**2

        ax.scatter(sx, sy, sz, c=colors[i % 2], s=50, alpha=0.7)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    plt.show()


def visualize_hilbert_space_projection(qsvm, X, y, title="Hilbert Space Projection"):
    """
    Visualize high-dimensional quantum states projected to 2D/3D
    """
    print("Computing quantum states for visualization...")

    # Get quantum states
    states = []
    for x in X:
        qc = qsvm.create_feature_map(x)
        psi = Statevector(qc)
        states.append(psi.data)

    states = np.array(states)

    # PCA for dimensionality reduction
    from sklearn.decomposition import PCA

    # Use real and imaginary parts
    states_real = np.column_stack([states.real, states.imag])

    pca = PCA(n_components=3)
    states_reduced = pca.fit_transform(states_real)

    # 3D visualization
    fig = plt.figure(figsize=(12, 5))

    # 3D plot
    ax1 = fig.add_subplot(121, projection='3d')
    colors = ['red', 'blue']
    for class_val in np.unique(y):
        mask = y == class_val
        ax1.scatter(states_reduced[mask, 0],
                    states_reduced[mask, 1],
                    states_reduced[mask, 2],
                    c=colors[class_val],
                    label=f'Class {class_val}',
                    alpha=0.7)

    ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
    ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
    ax1.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.2%} var)')
    ax1.set_title('3D Hilbert Space Projection')
    ax1.legend()

    # 2D plot
    ax2 = fig.add_subplot(122)
    for class_val in np.unique(y):
        mask = y == class_val
        ax2.scatter(states_reduced[mask, 0],
                    states_reduced[mask, 1],
                    c=colors[class_val],
                    label=f'Class {class_val}',
                    alpha=0.7)

    ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} var)')
    ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} var)')
    ax2.set_title('2D Hilbert Space Projection')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def visualize_quantum_kernel_matrix(kernel_matrix, title="Quantum Kernel Matrix"):
    """
    Visualize the quantum kernel matrix
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(kernel_matrix, annot=False, cmap='viridis',
                cbar_kws={'label': 'Fidelity'})
    plt.title(title)
    plt.xlabel('Sample Index')
    plt.ylabel('Sample Index')
    plt.show()


def grid_search_qsvm(X_train, y_train, X_test, y_test):
    """
    Grid search over quantum SVM parameters
    """
    # Parameter grid
    param_grid = {
        'n_qubits': [2, 3, 4],
        'entanglement_type': ['linear', 'circular'],
        'depth': [1, 2]
    }

    results = []

    print("Starting Grid Search...")
    print("-" * 50)

    for params in itertools.product(*param_grid.values()):
        n_qubits, entanglement_type, depth = params

        print(
            f"Testing: n_qubits={n_qubits}, entanglement={entanglement_type}, depth={depth}")

        try:
            # Create and train QSVM
            qsvm = QuantumSVM(n_qubits=n_qubits,
                              entanglement_type=entanglement_type,
                              depth=depth)
            qsvm.fit(X_train, y_train)

            # Evaluate
            train_score = qsvm.score(X_train, y_train)
            test_score = qsvm.score(X_test, y_test)

            results.append({
                'n_qubits': n_qubits,
                'entanglement_type': entanglement_type,
                'depth': depth,
                'train_score': train_score,
                'test_score': test_score
            })

            print(
                f"  Train Score: {train_score:.4f}, Test Score: {test_score:.4f}")

        except Exception as e:
            print(f"  Error: {e}")

    return results


def plot_grid_search_results(results):
    """
    Plot grid search results
    """
    if not results:
        print("No results to plot")
        return

    # Convert to DataFrame for easier plotting
    import pandas as pd
    df = pd.DataFrame(results)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: Test score vs n_qubits
    ax1 = axes[0, 0]
    for ent_type in df['entanglement_type'].unique():
        data = df[df['entanglement_type'] == ent_type]
        ax1.plot(data['n_qubits'], data['test_score'],
                 marker='o', label=f'{ent_type}', linewidth=2)
    ax1.set_xlabel('Number of Qubits')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Test Accuracy vs Number of Qubits')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Test score vs depth
    ax2 = axes[0, 1]
    for ent_type in df['entanglement_type'].unique():
        data = df[df['entanglement_type'] == ent_type]
        ax2.plot(data['depth'], data['test_score'],
                 marker='s', label=f'{ent_type}', linewidth=2)
    ax2.set_xlabel('Circuit Depth')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_title('Test Accuracy vs Circuit Depth')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Train vs Test scores
    ax3 = axes[1, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(df)))
    for i, (_, row) in enumerate(df.iterrows()):
        ax3.scatter(row['train_score'], row['test_score'],
                    c=[colors[i]], s=100, alpha=0.7,
                    label=f"q{row['n_qubits']}-{row['entanglement_type'][:3]}-d{row['depth']}")
    ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax3.set_xlabel('Train Accuracy')
    ax3.set_ylabel('Test Accuracy')
    ax3.set_title('Train vs Test Accuracy')
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.grid(True, alpha=0.3)

    # Plot 4: Heatmap of configurations
    ax4 = axes[1, 1]
    pivot_data = df.pivot_table(values='test_score',
                                index='n_qubits',
                                columns='entanglement_type',
                                aggfunc='mean')
    sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', ax=ax4)
    ax4.set_title('Test Accuracy Heatmap')

    plt.tight_layout()
    plt.show()


def demonstrate_feature_maps(qsvm, X_sample):
    """
    Demonstrate different feature map circuits
    """
    print("Quantum Feature Map Circuits:")
    print("=" * 40)

    for i, x in enumerate(X_sample[:3]):
        print(f"\nSample {i+1}: {x}")
        qc = qsvm.create_feature_map(x)
        print(f"Circuit depth: {qc.depth()}")
        print(f"Circuit gates: {qc.size()}")
        print(qc.draw())


def main():
    """
    Main function demonstrating complete QSVM implementation
    """
    print("🚀 Complete Quantum SVM Implementation from Scratch")
    print("=" * 60)

    # Generate datasets
    print("\n1. Generating Datasets...")
    datasets = {
        'moons': make_moons(n_samples=100, noise=0.1, random_state=42),
        'circles': make_circles(n_samples=100, noise=0.1, random_state=42),
        'classification': make_classification(n_samples=100, n_features=2, n_redundant=0,
                                              n_informative=2, random_state=42)
    }

    # Choose dataset
    dataset_name = 'moons'
    X, y = datasets[dataset_name]

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    print(f"Dataset: {dataset_name}")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Visualize original data
    plt.figure(figsize=(12, 4))

    plt.subplot(131)
    colors = ['red', 'blue']
    for class_val in np.unique(y):
        mask = y == class_val
        plt.scatter(X[mask, 0], X[mask, 1], c=colors[class_val],
                    label=f'Class {class_val}', alpha=0.7)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Original Data')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(132)
    for class_val in np.unique(y_train):
        mask = y_train == class_val
        plt.scatter(X_train[mask, 0], X_train[mask, 1], c=colors[class_val],
                    label=f'Class {class_val}', alpha=0.7)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Training Data')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(133)
    for class_val in np.unique(y_test):
        mask = y_test == class_val
        plt.scatter(X_test[mask, 0], X_test[mask, 1], c=colors[class_val],
                    label=f'Class {class_val}', alpha=0.7)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Test Data')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 2. Grid Search
    print("\n2. Performing Grid Search...")
    results = grid_search_qsvm(X_train, y_train, X_test, y_test)

    # Plot grid search results
    plot_grid_search_results(results)

    # 3. Train best model
    print("\n3. Training Best Model...")
    if results:
        best_result = max(results, key=lambda x: x['test_score'])
        print(f"Best parameters: {best_result}")

        best_qsvm = QuantumSVM(n_qubits=best_result['n_qubits'],
                               entanglement_type=best_result['entanglement_type'],
                               depth=best_result['depth'])
    else:
        print("Using default parameters...")
        best_qsvm = QuantumSVM(n_qubits=2, entanglement_type='linear', depth=1)

    # Train the model
    best_qsvm.fit(X_train, y_train)

    # 4. Evaluate model
    print("\n4. Model Evaluation...")
    train_pred = best_qsvm.predict(X_train)
    test_pred = best_qsvm.predict(X_test)

    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)

    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, test_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # 5. Visualize quantum kernel matrix
    print("\n5. Visualizing Quantum Kernel Matrix...")
    visualize_quantum_kernel_matrix(best_qsvm.kernel_matrix_train,
                                    "Quantum Kernel Matrix (Training Data)")

    # 6. Hilbert space visualization
    print("\n6. Hilbert Space Visualization...")
    visualize_hilbert_space_projection(best_qsvm, X_train, y_train,
                                       "Quantum States in Hilbert Space")

    # 7. Feature map demonstration
    print("\n7. Quantum Feature Map Circuits...")
    demonstrate_feature_maps(best_qsvm, X_train)

    # 8. Classical vs Quantum comparison
    print("\n8. Classical vs Quantum SVM Comparison...")

    # Classical SVM
    classical_svm = SVC(kernel='rbf', C=1.0)
    classical_svm.fit(X_train, y_train)
    classical_pred = classical_svm.predict(X_test)
    classical_accuracy = accuracy_score(y_test, classical_pred)

    # Comparison plot
    plt.figure(figsize=(10, 6))

    models = ['Classical SVM', 'Quantum SVM']
    accuracies = [classical_accuracy, test_accuracy]
    colors = ['lightcoral', 'lightblue']

    bars = plt.bar(models, accuracies, color=colors,
                   alpha=0.7, edgecolor='black')
    plt.ylabel('Test Accuracy')
    plt.title('Classical vs Quantum SVM Performance')
    plt.ylim(0, 1)

    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{acc:.4f}', ha='center', va='bottom', fontweight='bold')

    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.show()

    print(f"\nClassical SVM Accuracy: {classical_accuracy:.4f}")
    print(f"Quantum SVM Accuracy: {test_accuracy:.4f}")

    print("\n🎉 Complete QSVM Analysis Finished!")
    print("=" * 60)


if __name__ == "__main__":
    main()
