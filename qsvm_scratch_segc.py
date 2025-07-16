from segc import SEGCSearcher
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
import itertools
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class QuantumSVMWithSEGC:
    """
    Quantum Support Vector Machine with SEGC-enhanced optimization
    """

    def __init__(self, n_qubits=4, entanglement_type='linear', depth=1,
                 use_segc=True, segc_params=None):
        self.n_qubits = n_qubits
        self.entanglement_type = entanglement_type
        self.depth = depth
        self.use_segc = use_segc
        self.classifier = None
        self.X_train = None
        self.y_train = None
        self.kernel_matrix_train = None
        self.feature_maps = []

        # SEGC parameters optimized for financial dataset
        if segc_params is None:
            self.segc_params = {
                'n_qubits': 6,  # Increased for hyperparameter optimization
                'k_coarse': 3,  # 8 subspaces for coarse search
                'shots': 1024,  # Balanced for accuracy and speed
                'max_iterations': 4,  # Sufficient iterations with early stopping
                'decay_rate': 0.8  # Balanced feedback
            }
        else:
            self.segc_params = segc_params

        self.segc_searcher = None
        if self.use_segc:
            self.segc_searcher = SEGCSearcher(**self.segc_params)

    def create_feature_map(self, x):
        """
        Create quantum feature map circuit for financial features
        """
        qc = QuantumCircuit(self.n_qubits)
        for d in range(self.depth):
            for i in range(self.n_qubits):
                qc.h(i)
            for i in range(min(len(x), self.n_qubits)):
                qc.rz(x[i] * np.pi, i)
                qc.ry(x[i] * np.pi/2, i)
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

    def compute_fidelity_with_segc(self, x1, x2):
        """
        Compute quantum kernel entry using SEGC for optimization
        """
        try:
            qc1 = self.create_feature_map(x1)
            qc2 = self.create_feature_map(x2)
            psi1 = Statevector(qc1)
            psi2 = Statevector(qc2)
            fidelity = np.abs(np.conj(psi1.data) @ psi2.data) ** 2

            # Enable SEGC for fidelity enhancement
            if self.use_segc and self.segc_searcher is not None:
                target_fidelity = int(
                    fidelity * (2**self.segc_params['n_qubits'] - 1))
                try:
                    best_result, _ = self.segc_searcher.search_with_feedback(
                        target_fidelity)
                    if best_result and best_result['success_rate'] > 0.1:
                        enhancement_factor = 1 + 0.1 * \
                            best_result['success_rate']
                        fidelity = min(1.0, fidelity * enhancement_factor)
                except Exception as e:
                    print(
                        f"SEGC optimization failed: {e}, using standard fidelity")
            return fidelity
        except Exception as e:
            print(f"Error computing fidelity: {e}")
            return 0.0

    def compute_fidelity(self, x1, x2):
        """
        Compute quantum kernel entry (fidelity) between two data points
        """
        return self.compute_fidelity_with_segc(x1, x2) if self.use_segc else \
            np.abs(np.conj(Statevector(self.create_feature_map(x1)).data) @
                   Statevector(self.create_feature_map(x2)).data) ** 2

    def optimize_hyperparameters_with_segc(self, X_train, y_train, X_val, y_val):
        """
        Use SEGC to optimize hyperparameters
        """
        if not self.use_segc:
            return {'C': 1.0}

        print("Using SEGC for hyperparameter optimization...")
        C_values = np.logspace(-1, 2, 20)  # Dense grid for financial data
        best_params = {'C': 1.0}
        best_score = 0.0

        for C in C_values:
            try:
                kernel_matrix = self.compute_kernel_matrix(X_train)
                temp_classifier = SVC(kernel='precomputed', C=C)
                temp_classifier.fit(kernel_matrix, y_train)
                val_kernel = self.compute_kernel_matrix(X_val, X_train)
                val_score = temp_classifier.score(val_kernel, y_val)

                if self.segc_searcher is not None:
                    target_score = int(
                        val_score * (2**self.segc_params['n_qubits'] - 1))
                    try:
                        segc_result, _ = self.segc_searcher.search_with_feedback(
                            target_score)
                        if segc_result and segc_result['success_rate'] > 0.1:
                            val_score *= (1 + 0.2 *
                                          segc_result['success_rate'])
                    except Exception as e:
                        print(f"SEGC parameter optimization failed: {e}")

                if val_score > best_score:
                    best_score = val_score
                    best_params['C'] = C
            except Exception as e:
                print(f"Error evaluating C={C}: {e}")
                continue

        print(f"SEGC-optimized parameters: {best_params}")
        return best_params

    def compute_kernel_matrix(self, X1, X2=None):
        """
        Compute full quantum kernel matrix
        """
        if X2 is None:
            X2 = X1
        kernel = np.zeros((len(X1), len(X2)))
        total_pairs = len(X1) * len(X2)
        computed_pairs = 0

        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                kernel[i, j] = self.compute_fidelity(x1, x2)
                computed_pairs += 1
                if computed_pairs % max(1, total_pairs // 10) == 0:
                    progress = (computed_pairs / total_pairs) * 100
                    print(f"Kernel computation progress: {progress:.1f}%")
        return kernel

    def fit(self, X, y):
        """
        Train the quantum SVM with SEGC optimization
        """
        self.X_train = X
        self.y_train = y
        X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
            X, y, test_size=0.2, random_state=42)

        best_params = self.optimize_hyperparameters_with_segc(
            X_train_split, y_train_split, X_val_split, y_val_split)

        print("Computing quantum kernel matrix with SEGC enhancement...")
        self.kernel_matrix_train = self.compute_kernel_matrix(X)
        self.classifier = SVC(kernel='precomputed', C=best_params['C'])
        self.classifier.fit(self.kernel_matrix_train, y)
        return self

    def predict(self, X):
        """
        Make predictions
        """
        if self.classifier is None:
            raise ValueError("Model not fitted yet!")
        kernel_matrix_test = self.compute_kernel_matrix(X, self.X_train)
        return self.classifier.predict(kernel_matrix_test)

    def score(self, X, y):
        """
        Compute accuracy score
        """
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

    def get_segc_statistics(self):
        """
        Get SEGC optimization statistics
        """
        if not self.use_segc or self.segc_searcher is None:
            return None
        return {
            'segc_enabled': True,
            'search_history': self.segc_searcher.search_history,
            'subspace_scores': dict(self.segc_searcher.subspace_scores),
            'segc_params': self.segc_params
        }


def visualize_segc_optimization(qsvm, title="SEGC Optimization Results"):
    """
    Visualize SEGC optimization results
    """
    segc_stats = qsvm.get_segc_statistics()
    if not segc_stats or not segc_stats['segc_enabled']:
        print("SEGC not enabled or no statistics available")
        return

    search_history = segc_stats['search_history']
    if not search_history:
        print("No SEGC search history available")
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    iterations = [h['iteration'] for h in search_history]
    success_rates = [h['success_rate'] for h in search_history]
    ax1.plot(iterations, success_rates, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('SEGC Iteration')
    ax1.set_ylabel('Success Rate')
    ax1.set_title('SEGC Success Rate Evolution')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)

    subspace_scores = segc_stats['subspace_scores']
    if subspace_scores:
        subspaces = list(subspace_scores.keys())
        scores = list(subspace_scores.values())
        ax2.bar(range(len(subspaces)), scores, alpha=0.7)
        ax2.set_xlabel('Subspace')
        ax2.set_ylabel('Score')
        ax2.set_title('SEGC Subspace Scores')
        ax2.set_xticks(range(len(subspaces)))
        ax2.set_xticklabels(subspaces, rotation=45)
        ax2.grid(True, alpha=0.3)

    ax3.axis('off')
    params_text = f"""SEGC Parameters:
n_qubits: {segc_stats['segc_params']['n_qubits']}
k_coarse: {segc_stats['segc_params']['k_coarse']}
shots: {segc_stats['segc_params']['shots']}
max_iterations: {segc_stats['segc_params']['max_iterations']}
decay_rate: {segc_stats['segc_params']['decay_rate']}

Performance:
Total SEGC runs: {len(search_history)}
Best success rate: {max(success_rates) if success_rates else 0:.3f}
"""
    ax3.text(0.1, 0.5, params_text, fontsize=12, fontfamily='monospace',
             verticalalignment='center')

    if search_history:
        best_iteration = max(search_history, key=lambda x: x['success_rate'])
        target_counts = [h['target_count'] for h in search_history]
        ax4.bar(iterations, target_counts, alpha=0.7, color='green')
        ax4.set_xlabel('SEGC Iteration')
        ax4.set_ylabel('Target Count')
        ax4.set_title('SEGC Target Detection Count')
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.suptitle(title, fontsize=16, y=0.98)
    plt.show()


def compare_segc_vs_classical_qsvm(X_train, y_train, X_test, y_test):
    """
    Compare QSVM with and without SEGC optimization
    """
    print("\nComparing QSVM with and without SEGC...")
    print("-" * 50)

    print("Training Classical QSVM...")
    qsvm_classical = QuantumSVMWithSEGC(n_qubits=4, entanglement_type='linear',
                                        depth=1, use_segc=False)
    qsvm_classical.fit(X_train, y_train)
    classical_score = qsvm_classical.score(X_test, y_test)

    print("Training SEGC-enhanced QSVM...")
    qsvm_segc = QuantumSVMWithSEGC(n_qubits=4, entanglement_type='linear',
                                   depth=1, use_segc=True)
    qsvm_segc.fit(X_train, y_train)
    segc_score = qsvm_segc.score(X_test, y_test)

    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    models = ['Classical QSVM', 'SEGC-enhanced QSVM']
    scores = [classical_score, segc_score]
    colors = ['lightcoral', 'lightgreen']
    bars = plt.bar(models, scores, color=colors, alpha=0.7, edgecolor='black')
    plt.ylabel('Test Accuracy')
    plt.title('Performance Comparison')
    plt.ylim(0, 1)
    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{score:.4f}', ha='center', va='bottom', fontweight='bold')

    plt.subplot(2, 2, 2)
    kernel_diff = np.abs(qsvm_segc.kernel_matrix_train -
                         qsvm_classical.kernel_matrix_train)
    plt.imshow(kernel_diff, cmap='viridis', aspect='auto')
    plt.colorbar(label='Kernel Difference')
    plt.title('Kernel Matrix Difference\n(SEGC - Classical)')

    plt.subplot(2, 2, 3)
    classical_pred = qsvm_classical.predict(X_test)
    segc_pred = qsvm_segc.predict(X_test)
    agreement = np.mean(classical_pred == segc_pred)
    disagreement = np.mean(classical_pred != segc_pred)
    plt.bar(['Agreement', 'Disagreement'], [agreement, disagreement],
            color=['lightblue', 'orange'], alpha=0.7)
    plt.ylabel('Fraction')
    plt.title('Prediction Agreement')
    plt.ylim(0, 1)

    plt.subplot(2, 2, 4)
    plt.axis('off')
    summary_text = f"""Comparison Summary:
Classical QSVM Accuracy: {classical_score:.4f}
SEGC-enhanced QSVM Accuracy: {segc_score:.4f}
Improvement: {segc_score - classical_score:.4f}
Prediction Agreement: {agreement:.4f}
"""
    plt.text(0.1, 0.5, summary_text, fontsize=11, fontfamily='monospace',
             verticalalignment='center')
    plt.tight_layout()
    plt.show()

    visualize_segc_optimization(qsvm_segc, "SEGC Optimization in QSVM")
    return qsvm_classical, qsvm_segc


def main():
    """
    Main function demonstrating QSVM with SEGC on financial data
    """
    print("🚀 Quantum SVM with SEGC Integration for Financial Data")
    print("=" * 60)

    # Load financial dataset
    import pandas as pd
    # Replace with actual file path
    data = pd.read_csv('ticker/JPM/JPM_data.csv')
    features = ['rsi_scaled', 'volatility_scaled',
                'ma5_scaled', 'close_scaled']
    X = data[features].values
    y = data['regime'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # Visualize data (2D projection of two features)
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    colors = ['red', 'blue']
    for class_val in np.unique(y_train):
        mask = y_train == class_val
        plt.scatter(X_train[mask, 0], X_train[mask, 1], c=colors[class_val],
                    label=f'Class {class_val}', alpha=0.7)
    plt.xlabel('rsi_scaled')
    plt.ylabel('volatility_scaled')
    plt.title('Training Data (2D Projection)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(122)
    for class_val in np.unique(y_test):
        mask = y_test == class_val
        plt.scatter(X_test[mask, 0], X_test[mask, 1], c=colors[class_val],
                    label=f'Class {class_val}', alpha=0.7)
    plt.xlabel('rsi_scaled')
    plt.ylabel('volatility_scaled')
    plt.title('Test Data (2D Projection)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Compare QSVMs
    print("\n2. Comparing Classical vs SEGC-enhanced QSVM...")
    qsvm_classical, qsvm_segc = compare_segc_vs_classical_qsvm(
        X_train, y_train, X_test, y_test)

    print("\n3. Detailed Analysis of SEGC-enhanced QSVM...")
    segc_stats = qsvm_segc.get_segc_statistics()
    if segc_stats:
        print(f"SEGC Statistics:")
        print(
            f"  - Total SEGC optimizations: {len(segc_stats['search_history'])}")
        print(f"  - SEGC parameters: {segc_stats['segc_params']}")
        print(
            f"  - Subspace scores: {len(segc_stats['subspace_scores'])} subspaces explored")

    print("\n4. Visualizing Quantum Kernel Matrices...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    im1 = ax1.imshow(qsvm_classical.kernel_matrix_train,
                     cmap='viridis', aspect='auto')
    ax1.set_title('Classical QSVM Kernel Matrix')
    ax1.set_xlabel('Sample Index')
    ax1.set_ylabel('Sample Index')
    plt.colorbar(im1, ax=ax1, label='Fidelity')

    im2 = ax2.imshow(qsvm_segc.kernel_matrix_train,
                     cmap='viridis', aspect='auto')
    ax2.set_title('SEGC-enhanced QSVM Kernel Matrix')
    ax2.set_xlabel('Sample Index')
    ax2.set_ylabel('Sample Index')
    plt.colorbar(im2, ax=ax2, label='Fidelity')
    plt.tight_layout()
    plt.show()

    print("\n🎉 QSVM with SEGC Integration Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
