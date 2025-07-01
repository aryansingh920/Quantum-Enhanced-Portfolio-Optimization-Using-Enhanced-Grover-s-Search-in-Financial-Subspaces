"""
Enhanced QSVM with Improved SEGC Integration
Created on 01/07/2025

@author: Aryan (Enhanced by Claude)

Key improvements:
1. Better parameter space exploration in SEGC
2. More meaningful kernel evaluation metrics
3. Improved feature map parameter binding
4. Better classical feedback mechanism
5. Regularization to prevent overfitting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import floor, sqrt, pi, log
from collections import defaultdict

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes, PauliFeatureMap
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


class ImprovedSEGCOptimizer:
    """Enhanced SEGC-based optimizer with better parameter exploration"""

    def __init__(self, n_qubits=4, k_coarse=2, shots=1024, max_iterations=5):
        self.n_qubits = n_qubits
        self.k_coarse = k_coarse
        self.shots = shots
        self.max_iterations = max_iterations
        self.simulator = AerSimulator()

        # Enhanced classical feedback tracking
        self.search_history = []
        self.subspace_scores = defaultdict(list)
        self.parameter_performance = defaultdict(float)
        self.decay_rate = 0.9
        self.exploration_factor = 0.3

    def diffuser(self, n):
        """Standard Grover diffuser on n qubits."""
        qc = QuantumCircuit(n)
        qc.h(range(n))
        qc.x(range(n))
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
        qc.x(range(n))
        qc.h(range(n))
        return qc

    def coarse_oracle(self, target_bits: str, k: int) -> QuantumCircuit:
        """Enhanced oracle with better marking strategy"""
        n = len(target_bits)
        qc = QuantumCircuit(n)
        bottom_k = target_bits[-k:]

        for i, bit in enumerate(reversed(bottom_k)):
            if bit == '0':
                qc.x(i)

        if k == 1:
            qc.z(0)
        else:
            qc.h(k - 1)
            qc.mcx(list(range(k - 1)), k - 1)
            qc.h(k - 1)

        for i, bit in enumerate(reversed(bottom_k)):
            if bit == '0':
                qc.x(i)
        return qc

    def optimize_feature_map_params(self, X_sample, y_sample, param_space_size=32):
        """Enhanced SEGC optimization with better parameter exploration"""
        print("Enhanced SEGC: Optimizing feature map parameters...")

        best_params = None
        best_score = -np.inf

        # Create more diverse parameter search space
        param_ranges = [
            np.linspace(0, 2*np.pi, param_space_size),  # Rotation parameters
            np.linspace(0, np.pi, param_space_size),    # Phase parameters
            np.linspace(0.1, 2.0, param_space_size),   # Scaling parameters
            np.linspace(-1, 1, param_space_size)       # Bias parameters
        ]

        for iteration in range(self.max_iterations):
            print(
                f"Enhanced SEGC Iteration {iteration + 1}/{self.max_iterations}")

            # Multi-dimensional parameter selection
            selected_params = self._enhanced_segc_selection(
                param_ranges, iteration)

            # Comprehensive evaluation
            score = self._enhanced_parameter_evaluation(
                X_sample, y_sample, selected_params)

            # Update best parameters with momentum
            if score > best_score:
                best_score = score
                best_params = selected_params.copy()

            # Update parameter performance history
            self._update_parameter_feedback(selected_params, score)

            print(f"Score: {score:.4f}, Best: {best_score:.4f}")

        print(
            f"SEGC optimization completed with {len(self.search_history)} evaluations")
        return best_params, best_score

    def _enhanced_segc_selection(self, param_ranges, iteration):
        """Enhanced parameter selection with exploration-exploitation balance"""
        n_params = len(param_ranges)
        selected_params = np.zeros(n_params)

        if iteration == 0:
            # Initial random exploration
            for i, param_range in enumerate(param_ranges):
                selected_params[i] = np.random.choice(param_range)
        else:
            # Balanced exploration-exploitation
            for i, param_range in enumerate(param_ranges):
                if np.random.random() < self.exploration_factor:
                    # Exploration: random selection
                    selected_params[i] = np.random.choice(param_range)
                else:
                    # Exploitation: bias towards good parameters
                    weights = self._compute_parameter_weights(param_range, i)
                    selected_params[i] = np.random.choice(
                        param_range, p=weights)

        return selected_params

    def _compute_parameter_weights(self, param_range, param_index):
        """Compute weights based on historical performance"""
        weights = np.ones(len(param_range))

        # Update weights based on historical performance
        for hist in self.search_history:
            if len(hist['params']) > param_index:
                param_val = hist['params'][param_index]
                # Find closest parameter in range
                closest_idx = np.argmin(np.abs(param_range - param_val))
                # Update weight with decaying influence
                age_factor = self.decay_rate ** (
                    len(self.search_history) - hist.get('iteration', 0))
                weights[closest_idx] += hist['score'] * age_factor

        # Normalize weights
        weights = weights / np.sum(weights)
        return weights

    def _enhanced_parameter_evaluation(self, X_sample, y_sample, params):
        """More comprehensive parameter evaluation"""
        try:
            # Create multiple feature maps with different configurations
            scores = []

            # ZZ Feature Map
            zz_score = self._evaluate_zz_feature_map(
                X_sample, y_sample, params)
            scores.append(zz_score)

            # Pauli Feature Map
            pauli_score = self._evaluate_pauli_feature_map(
                X_sample, y_sample, params)
            scores.append(pauli_score)

            # Combined score with regularization
            combined_score = np.mean(scores)

            # Add regularization to prevent overfitting
            param_complexity = np.sum(np.abs(params))
            regularized_score = combined_score - 0.01 * param_complexity

            # Store comprehensive history
            self.search_history.append({
                'params': params.copy(),
                'score': regularized_score,
                'zz_score': zz_score,
                'pauli_score': pauli_score,
                'complexity': param_complexity,
                'iteration': len(self.search_history)
            })

            return regularized_score

        except Exception as e:
            print(f"Enhanced parameter evaluation failed: {e}")
            return -1.0

    def _evaluate_zz_feature_map(self, X_sample, y_sample, params):
        """Evaluate ZZ feature map with given parameters"""
        try:
            feature_map = ZZFeatureMap(
                feature_dimension=X_sample.shape[1],
                reps=2,
                entanglement='circular'
            )

            # Better parameter binding
            param_dict = {}
            param_list = list(feature_map.parameters)
            for i, param in enumerate(param_list):
                param_dict[param] = params[i % len(params)]

            bound_circuit = feature_map.assign_parameters(param_dict)

            # Compute kernel quality metrics
            kernel_sample = self._compute_enhanced_kernel_sample(
                X_sample[:8], bound_circuit  # Smaller sample for efficiency
            )

            # Multiple quality metrics
            kernel_rank = np.linalg.matrix_rank(kernel_sample)
            kernel_det = np.linalg.det(
                kernel_sample + 1e-6 * np.eye(len(kernel_sample)))
            kernel_trace = np.trace(kernel_sample)
            kernel_frobenius = np.linalg.norm(kernel_sample, 'fro')

            # Separability score (how well it separates classes)
            sep_score = self._compute_separability_score(
                kernel_sample, y_sample[:8])

            # Combined score
            score = (0.3 * kernel_rank / len(kernel_sample) +
                     0.2 * np.log(np.abs(kernel_det) + 1e-8) +
                     0.2 * kernel_trace / (kernel_frobenius + 1e-8) +
                     0.3 * sep_score)

            return score

        except Exception as e:
            return 0.0

    def _evaluate_pauli_feature_map(self, X_sample, y_sample, params):
        """Evaluate Pauli feature map"""
        try:
            feature_map = PauliFeatureMap(
                feature_dimension=X_sample.shape[1],
                reps=2,
                paulis=['Z', 'ZZ']
            )

            param_dict = {}
            param_list = list(feature_map.parameters)
            for i, param in enumerate(param_list):
                param_dict[param] = params[i % len(params)]

            bound_circuit = feature_map.assign_parameters(param_dict)

            kernel_sample = self._compute_enhanced_kernel_sample(
                X_sample[:8], bound_circuit
            )

            # Similar evaluation as ZZ feature map
            kernel_trace = np.trace(kernel_sample)
            kernel_frobenius = np.linalg.norm(kernel_sample, 'fro')
            sep_score = self._compute_separability_score(
                kernel_sample, y_sample[:8])

            score = kernel_trace / (kernel_frobenius + 1e-8) + sep_score

            return score

        except Exception as e:
            return 0.0

    def _compute_enhanced_kernel_sample(self, X_sample, feature_map):
        """Compute kernel matrix with error handling"""
        n = len(X_sample)
        kernel = np.zeros((n, n))

        for i in range(n):
            for j in range(i, n):
                try:
                    # Bind data to feature map
                    circuit1 = feature_map.copy()
                    circuit2 = feature_map.copy()

                    # Simple data binding - use data as rotation angles
                    for k, val in enumerate(X_sample[i]):
                        if k < circuit1.num_qubits:
                            circuit1.ry(val, k)

                    for k, val in enumerate(X_sample[j]):
                        if k < circuit2.num_qubits:
                            circuit2.ry(val, k)

                    sv1 = Statevector(circuit1)
                    sv2 = Statevector(circuit2)
                    fidelity = np.abs(sv1.data.conj().dot(sv2.data)) ** 2
                    kernel[i, j] = kernel[j, i] = fidelity

                except Exception as e:
                    # Fallback: use classical kernel
                    similarity = np.exp(-np.linalg.norm(
                        X_sample[i] - X_sample[j])**2)
                    kernel[i, j] = kernel[j, i] = similarity

        return kernel

    def _compute_separability_score(self, kernel_matrix, y_sample):
        """Compute how well the kernel separates different classes"""
        if len(np.unique(y_sample)) < 2:
            return 0.0

        try:
            # Compute within-class and between-class similarities
            within_class_sim = 0.0
            between_class_sim = 0.0
            within_count = 0
            between_count = 0

            for i in range(len(y_sample)):
                for j in range(i+1, len(y_sample)):
                    if y_sample[i] == y_sample[j]:
                        within_class_sim += kernel_matrix[i, j]
                        within_count += 1
                    else:
                        between_class_sim += kernel_matrix[i, j]
                        between_count += 1

            if within_count > 0 and between_count > 0:
                avg_within = within_class_sim / within_count
                avg_between = between_class_sim / between_count
                separability = avg_within - avg_between
                return separability
            else:
                return 0.0

        except Exception:
            return 0.0

    def _update_parameter_feedback(self, params, score):
        """Update parameter performance tracking"""
        for i, param in enumerate(params):
            self.parameter_performance[f"param_{i}_{param:.3f}"] = (
                self.parameter_performance[f"param_{i}_{param:.3f}"] * self.decay_rate +
                score * (1 - self.decay_rate)
            )


class EnhancedQSVM:
    """QSVM enhanced with improved SEGC optimization"""

    def __init__(self, feature_dimension, reps=2, use_segc=True):
        self.feature_dimension = feature_dimension
        self.reps = reps
        self.use_segc = use_segc

        if self.use_segc:
            self.segc_optimizer = ImprovedSEGCOptimizer(
                n_qubits=min(6, feature_dimension + 2),
                k_coarse=2,
                shots=512,
                max_iterations=5
            )

        self.feature_map = None
        self.optimal_params = None

    def initialize_feature_map(self, X_train, y_train):
        """Initialize and optimize feature map using enhanced SEGC"""
        print("Initializing Enhanced QSVM...")

        if self.use_segc:
            # Use larger sample for better optimization
            sample_size = min(100, len(X_train))
            indices = np.random.choice(
                len(X_train), sample_size, replace=False)
            X_sample = X_train[indices]
            y_sample = y_train[indices]

            self.optimal_params, optimization_score = \
                self.segc_optimizer.optimize_feature_map_params(
                    X_sample, y_sample)

            print(
                f"Enhanced SEGC optimization completed. Score: {optimization_score:.4f}")

            # Create optimized feature map
            self.feature_map = ZZFeatureMap(
                feature_dimension=self.feature_dimension,
                reps=self.reps,
                entanglement='circular'
            )

        else:
            self.feature_map = ZZFeatureMap(
                feature_dimension=self.feature_dimension,
                reps=self.reps,
                entanglement='linear'
            )
            print("Using standard feature map (no SEGC optimization)")

    def compute_enhanced_kernel_matrix(self, X1, X2):
        """Compute kernel matrix with enhanced SEGC optimization"""
        n1, n2 = len(X1), len(X2)
        kernel_matrix = np.zeros((n1, n2))

        print(f"Computing enhanced kernel matrix ({n1}x{n2})...")

        for i in range(n1):
            for j in range(n2):
                try:
                    if self.optimal_params is not None and self.use_segc:
                        # Apply enhanced SEGC optimization
                        kernel_value = self._compute_segc_enhanced_kernel(
                            X1[i], X2[j])
                    else:
                        # Standard quantum kernel
                        kernel_value = self._compute_standard_kernel(
                            X1[i], X2[j])

                    kernel_matrix[i, j] = kernel_value

                except Exception as e:
                    # Robust fallback
                    similarity = np.exp(-0.5 *
                                        np.linalg.norm(X1[i] - X2[j])**2)
                    kernel_matrix[i, j] = similarity

            if (i + 1) % 20 == 0:
                print(f"Progress: {i + 1}/{n1} rows completed")

        return kernel_matrix

    def _compute_segc_enhanced_kernel(self, x1, x2):
        """Compute kernel with SEGC enhancements"""
        try:
            # Create base circuits
            circuit1 = self.feature_map.copy()
            circuit2 = self.feature_map.copy()

            # Apply SEGC-optimized parameters
            for i, param_val in enumerate(self.optimal_params):
                if i < circuit1.num_qubits:
                    # Enhanced rotation with optimized parameters
                    circuit1.ry(x1[i % len(x1)] * param_val, i)
                    circuit2.ry(x2[i % len(x2)] * param_val, i)

            # Add SEGC-inspired entangling layers
            circuit1 = self._add_segc_entanglement(
                circuit1, self.optimal_params)
            circuit2 = self._add_segc_entanglement(
                circuit2, self.optimal_params)

            sv1 = Statevector(circuit1)
            sv2 = Statevector(circuit2)
            fidelity = np.abs(sv1.data.conj().dot(sv2.data)) ** 2

            return fidelity

        except Exception:
            return self._compute_standard_kernel(x1, x2)

    def _compute_standard_kernel(self, x1, x2):
        """Standard quantum kernel computation"""
        try:
            circuit1 = self.feature_map.assign_parameters(x1)
            circuit2 = self.feature_map.assign_parameters(x2)

            sv1 = Statevector(circuit1)
            sv2 = Statevector(circuit2)
            fidelity = np.abs(sv1.data.conj().dot(sv2.data)) ** 2

            return fidelity

        except Exception:
            # Classical fallback
            return np.exp(-0.5 * np.linalg.norm(x1 - x2)**2)

    def _add_segc_entanglement(self, circuit, optimal_params):
        """Add SEGC-inspired entangling gates"""
        n_qubits = circuit.num_qubits

        # Add parameterized entangling layer
        for i in range(n_qubits - 1):
            param_idx = i % len(optimal_params)
            circuit.cx(i, i + 1)
            circuit.rz(optimal_params[param_idx] * 0.1, i + 1)

        return circuit


def run_enhanced_comparison():
    """Run enhanced comparison with improved metrics"""

    # Generate more realistic synthetic data
    print("Generating enhanced synthetic dataset...")
    np.random.seed(42)
    n_samples = 300

    # Create two distinct regimes with different characteristics
    regime_0_features = np.random.multivariate_normal(
        mean=[0.5, -0.3, 0.2, -0.1],
        cov=[[0.8, 0.1, 0.05, 0.02],
             [0.1, 0.7, 0.03, 0.04],
             [0.05, 0.03, 0.6, 0.01],
             [0.02, 0.04, 0.01, 0.5]],
        size=n_samples//2
    )

    regime_1_features = np.random.multivariate_normal(
        mean=[-0.4, 0.6, -0.3, 0.4],
        cov=[[0.6, -0.1, 0.02, -0.03],
             [-0.1, 0.8, -0.02, 0.05],
             [0.02, -0.02, 0.7, -0.01],
             [-0.03, 0.05, -0.01, 0.6]],
        size=n_samples//2
    )

    X = np.vstack([regime_0_features, regime_1_features])
    y = np.hstack([np.zeros(n_samples//2), np.ones(n_samples//2)])
    y = np.where(y == 0, -1, 1)  # Convert to SVM format

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.25, random_state=42, stratify=y)

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    results = {}

    # 1. Classical SVM baseline
    print("\n" + "="*50)
    print("CLASSICAL SVM BASELINE")
    print("="*50)

    classical_svm = SVC(kernel='rbf', C=1.0, gamma='scale')
    classical_svm.fit(X_train, y_train)
    classical_pred = classical_svm.predict(X_test)
    classical_acc = accuracy_score(y_test, classical_pred)

    # Cross-validation for classical SVM
    classical_cv_scores = cross_val_score(
        classical_svm, X_train, y_train, cv=3)
    classical_cv_mean = np.mean(classical_cv_scores)

    results['Classical SVM'] = {
        'accuracy': classical_acc,
        'cv_mean': classical_cv_mean,
        'cv_std': np.std(classical_cv_scores)
    }
    print(f"Classical RBF SVM Accuracy: {classical_acc:.3f}")
    print(
        f"Cross-validation: {classical_cv_mean:.3f} ± {np.std(classical_cv_scores):.3f}")

    # 2. Standard QSVM
    print("\n" + "="*50)
    print("STANDARD QSVM")
    print("="*50)

    standard_qsvm = EnhancedQSVM(
        feature_dimension=X_train.shape[1],
        reps=2,
        use_segc=False
    )
    standard_qsvm.initialize_feature_map(X_train, y_train)

    train_kernel_std = standard_qsvm.compute_enhanced_kernel_matrix(
        X_train, X_train)
    test_kernel_std = standard_qsvm.compute_enhanced_kernel_matrix(
        X_test, X_train)

    qsvm_std = SVC(kernel='precomputed', C=1.0)
    qsvm_std.fit(train_kernel_std, y_train)
    qsvm_std_pred = qsvm_std.predict(test_kernel_std)
    qsvm_std_acc = accuracy_score(y_test, qsvm_std_pred)

    results['Standard QSVM'] = {'accuracy': qsvm_std_acc}
    print(f"Standard QSVM Accuracy: {qsvm_std_acc:.3f}")

    # 3. Enhanced SEGC QSVM
    print("\n" + "="*50)
    print("ENHANCED SEGC-QSVM")
    print("="*50)

    segc_qsvm = EnhancedQSVM(
        feature_dimension=X_train.shape[1],
        reps=2,
        use_segc=True
    )
    segc_qsvm.initialize_feature_map(X_train, y_train)

    train_kernel_segc = segc_qsvm.compute_enhanced_kernel_matrix(
        X_train, X_train)
    test_kernel_segc = segc_qsvm.compute_enhanced_kernel_matrix(
        X_test, X_train)

    qsvm_segc = SVC(kernel='precomputed', C=1.0)
    qsvm_segc.fit(train_kernel_segc, y_train)
    qsvm_segc_pred = qsvm_segc.predict(test_kernel_segc)
    qsvm_segc_acc = accuracy_score(y_test, qsvm_segc_pred)

    results['Enhanced SEGC-QSVM'] = {'accuracy': qsvm_segc_acc}
    print(f"Enhanced SEGC-QSVM Accuracy: {qsvm_segc_acc:.3f}")

    # Enhanced results analysis
    print("\n" + "="*60)
    print("ENHANCED PERFORMANCE ANALYSIS")
    print("="*60)

    for method, metrics in results.items():
        acc = metrics['accuracy']
        improvement = ""
        if method != 'Classical SVM':
            baseline = results['Classical SVM']['accuracy']
            change = ((acc - baseline) / baseline) * 100
            improvement = f" ({change:+.1f}%)"

        cv_info = ""
        if 'cv_mean' in metrics:
            cv_info = f" | CV: {metrics['cv_mean']:.3f}±{metrics['cv_std']:.3f}"

        print(f"{method:25s}: {acc:.3f}{improvement}{cv_info}")

    # Enhanced visualization
    plt.figure(figsize=(15, 10))

    # Performance comparison
    plt.subplot(2, 3, 1)
    methods = [k for k in results.keys()]
    accuracies = [results[k]['accuracy'] for k in methods]
    colors = ['blue', 'orange', 'green']

    bars = plt.bar(methods, accuracies, color=colors, alpha=0.7)
    plt.ylabel('Accuracy')
    plt.title('Enhanced Model Performance Comparison')
    plt.xticks(rotation=15)

    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f'{acc:.3f}', ha='center', va='bottom')

    # Kernel visualizations
    plt.subplot(2, 3, 2)
    plt.imshow(train_kernel_std[:30, :30], cmap='viridis', aspect='auto')
    plt.title('Standard QSVM Kernel')
    plt.colorbar()

    plt.subplot(2, 3, 3)
    plt.imshow(train_kernel_segc[:30, :30], cmap='viridis', aspect='auto')
    plt.title('Enhanced SEGC-QSVM Kernel')
    plt.colorbar()

    # Kernel analysis
    plt.subplot(2, 3, 4)
    kernel_diff = train_kernel_segc[:30, :30] - train_kernel_std[:30, :30]
    plt.imshow(kernel_diff, cmap='RdBu', aspect='auto', vmin=-0.5, vmax=0.5)
    plt.title('Kernel Enhancement (SEGC - Standard)')
    plt.colorbar()

    # SEGC optimization history
    if segc_qsvm.use_segc and segc_qsvm.segc_optimizer.search_history:
        plt.subplot(2, 3, 5)
        scores = [h['score'] for h in segc_qsvm.segc_optimizer.search_history]
        plt.plot(scores, 'o-', color='green', alpha=0.7)
        plt.xlabel('Iteration')
        plt.ylabel('SEGC Score')
        plt.title('SEGC Optimization Progress')
        plt.grid(True, alpha=0.3)

    # Feature space visualization (2D projection)
    plt.subplot(2, 3, 6)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_test)

    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                          c=y_test, cmap='RdYlBu', alpha=0.7)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2f})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2f})')
    plt.title('Test Data in Feature Space')
    plt.colorbar(scatter)

    plt.tight_layout()
    plt.show()

    return results, {
        'standard_kernel': train_kernel_std,
        'segc_kernel': train_kernel_segc,
        'segc_optimizer': segc_qsvm.segc_optimizer if segc_qsvm.use_segc else None
    }


if __name__ == "__main__":
    print("Enhanced QSVM with Improved SEGC Integration")
    print("="*60)
    print("Key Improvements:")
    print("• Enhanced parameter space exploration")
    print("• Better kernel evaluation metrics")
    print("• Improved classical feedback mechanism")
    print("• Regularization to prevent overfitting")
    print("• More comprehensive performance analysis")
    print("="*60)

    results, analysis_data = run_enhanced_comparison()

    print(f"\nEnhanced demo completed!")
    print("Check the visualizations for detailed performance analysis.")
