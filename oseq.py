#!/usr/bin/env python3
"""
Optimized SEGC-enhanced QSVM with performance improvements and proper parameter identification
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import time
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')


class OptimizedSEGCSearcher:
    """Optimized SEGC with early stopping and parameter analysis"""

    def __init__(self, n_qubits=4, k_coarse=2, shots=256, max_iterations=3,
                 decay_rate=0.9, convergence_threshold=0.01):
        self.n_qubits = n_qubits
        self.k_coarse = k_coarse
        self.shots = shots
        self.max_iterations = max_iterations
        self.decay_rate = decay_rate
        self.convergence_threshold = convergence_threshold
        self.search_history = []
        self.subspace_scores = defaultdict(float)
        self.parameter_analysis = {}

    def analyze_parameters(self, target_fidelity, data_characteristics):
        """Analyze and identify optimal SEGC parameters for QSVM"""

        # Parameter analysis based on data characteristics
        data_size = data_characteristics.get('size', 100)
        data_complexity = data_characteristics.get('complexity', 'medium')

        # Adaptive parameter selection
        if data_size < 50:
            optimal_qubits = 3
            optimal_k = 2
            optimal_shots = 128
        elif data_size < 200:
            optimal_qubits = 4
            optimal_k = 2
            optimal_shots = 256
        else:
            optimal_qubits = 5
            optimal_k = 3
            optimal_shots = 512

        # Complexity-based adjustments
        if data_complexity == 'high':
            optimal_shots *= 2
            optimal_k = min(optimal_k + 1, optimal_qubits - 1)
        elif data_complexity == 'low':
            optimal_shots = max(optimal_shots // 2, 64)

        self.parameter_analysis = {
            'recommended_qubits': optimal_qubits,
            'recommended_k_coarse': optimal_k,
            'recommended_shots': optimal_shots,
            'data_size': data_size,
            'data_complexity': data_complexity,
            'target_fidelity_range': target_fidelity,
            'convergence_strategy': 'early_stopping'
        }

        return self.parameter_analysis

    def search_with_feedback(self, target_value, data_characteristics=None):
        """Optimized SEGC search with parameter identification"""

        if data_characteristics:
            params = self.analyze_parameters(
                target_value, data_characteristics)
            print(f"SEGC Parameter Analysis:")
            print(f"  Recommended qubits: {params['recommended_qubits']}")
            print(f"  Recommended k_coarse: {params['recommended_k_coarse']}")
            print(f"  Recommended shots: {params['recommended_shots']}")
            print(f"  Data complexity: {params['data_complexity']}")

        # Simulate quantum search (simplified for efficiency)
        best_result = None
        best_success_rate = 0.0

        for iteration in range(self.max_iterations):
            # Simulate quantum measurement results
            success_rate = self._simulate_quantum_measurement(
                target_value, iteration)

            # Update subspace scores with decay
            for subspace in self.subspace_scores:
                self.subspace_scores[subspace] *= self.decay_rate

            # Add current iteration results
            current_subspace = format(target_value % (
                2**self.k_coarse), f"0{self.k_coarse}b")
            self.subspace_scores[current_subspace] += success_rate

            iteration_data = {
                'iteration': iteration + 1,
                'success_rate': success_rate,
                'target_count': int(success_rate * self.shots),
                'best_subspace': current_subspace,
                'convergence_metric': abs(success_rate - best_success_rate)
            }

            self.search_history.append(iteration_data)

            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_result = iteration_data

            # Early stopping conditions
            if success_rate > 0.85:
                print(
                    f"Early stopping: High success rate achieved ({success_rate:.3f})")
                break

            if len(self.search_history) >= 2:
                convergence = abs(self.search_history[-1]['success_rate'] -
                                  self.search_history[-2]['success_rate'])
                if convergence < self.convergence_threshold:
                    print(
                        f"Early stopping: Convergence detected ({convergence:.4f})")
                    break

        return best_result, self.search_history

    def _simulate_quantum_measurement(self, target_value, iteration):
        """Simulate quantum measurement with realistic noise and convergence"""
        # Base success rate increases with iterations but with diminishing returns
        base_rate = 0.1 + 0.6 * (1 - np.exp(-iteration * 0.8))

        # Add realistic quantum noise
        noise = np.random.normal(0, 0.05)

        # Subspace enhancement effect
        target_subspace = format(target_value % (
            2**self.k_coarse), f"0{self.k_coarse}b")
        enhancement = self.subspace_scores.get(target_subspace, 0) * 0.1

        success_rate = np.clip(base_rate + noise + enhancement, 0, 1)
        return success_rate


class OptimizedQuantumSVMWithSEGC:
    """Optimized Quantum SVM with SEGC integration"""

    def __init__(self, n_qubits=4, use_segc=True, segc_params=None):
        self.n_qubits = n_qubits
        self.use_segc = use_segc
        self.classifier = None
        self.X_train = None
        self.y_train = None
        self.segc_stats = None
        self.training_time = 0

        # Optimized SEGC parameters
        if segc_params is None:
            self.segc_params = {
                'n_qubits': 4,
                'k_coarse': 2,
                'shots': 256,
                'max_iterations': 3,
                'decay_rate': 0.9,
                'convergence_threshold': 0.01
            }
        else:
            self.segc_params = segc_params

        self.segc_searcher = OptimizedSEGCSearcher(
            **self.segc_params) if use_segc else None

    def _compute_simplified_kernel(self, X1, X2=None):
        """Compute simplified quantum-inspired kernel matrix"""
        if X2 is None:
            X2 = X1

        # Use RBF kernel as quantum-inspired baseline
        gamma = 1.0 / X1.shape[1]  # Adaptive gamma

        # Compute squared Euclidean distances
        X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
        X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)
        distances = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)

        # RBF kernel
        kernel = np.exp(-gamma * distances)

        # SEGC enhancement if enabled
        if self.use_segc and self.segc_searcher:
            enhancement_factor = self._get_segc_enhancement(X1, X2)
            kernel *= enhancement_factor

        return kernel

    def _get_segc_enhancement(self, X1, X2):
        """Get SEGC-based kernel enhancement"""
        # Simplified enhancement based on data characteristics
        data_characteristics = {
            'size': len(X1),
            'complexity': 'medium' if X1.shape[1] <= 4 else 'high'
        }

        # Use mean feature values as target for SEGC
        target_value = int(np.mean(X1) * 100) % (2 **
                                                 self.segc_params['n_qubits'])

        try:
            best_result, _ = self.segc_searcher.search_with_feedback(
                target_value, data_characteristics)

            if best_result:
                enhancement = 1.0 + 0.1 * best_result['success_rate']
                return enhancement
        except Exception as e:
            print(f"SEGC enhancement failed: {e}")

        return 1.0

    def fit(self, X, y):
        """Fit the optimized QSVM with SEGC"""
        start_time = time.time()

        self.X_train = X
        self.y_train = y

        print(f"Computing kernel matrix for {len(X)} samples...")
        kernel_matrix = self._compute_simplified_kernel(X)

        # Hyperparameter optimization with SEGC
        if self.use_segc:
            C = self._optimize_hyperparameters_with_segc(X, y)
        else:
            C = 1.0

        print(f"Training SVM with C={C:.3f}...")
        self.classifier = SVC(kernel='precomputed', C=C)
        self.classifier.fit(kernel_matrix, y)

        self.training_time = time.time() - start_time

        # Store SEGC statistics
        if self.use_segc:
            self.segc_stats = {
                'parameter_analysis': self.segc_searcher.parameter_analysis,
                'search_history': self.segc_searcher.search_history,
                'subspace_scores': dict(self.segc_searcher.subspace_scores)
            }

        return self

    def _optimize_hyperparameters_with_segc(self, X, y):
        """Optimize hyperparameters using SEGC"""
        print("Optimizing hyperparameters with SEGC...")

        # Limited C values for efficiency
        C_values = [0.1, 1.0, 10.0]
        best_C = 1.0
        best_score = 0.0

        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42)

        for C in C_values:
            try:
                # Compute kernel matrices
                train_kernel = self._compute_simplified_kernel(X_train)
                val_kernel = self._compute_simplified_kernel(X_val, X_train)

                # Train and evaluate
                temp_classifier = SVC(kernel='precomputed', C=C)
                temp_classifier.fit(train_kernel, y_train)
                score = temp_classifier.score(val_kernel, y_val)

                if score > best_score:
                    best_score = score
                    best_C = C

            except Exception as e:
                print(f"Error evaluating C={C}: {e}")
                continue

        return best_C

    def predict(self, X):
        """Make predictions"""
        if self.classifier is None:
            raise ValueError("Model not fitted!")

        kernel_matrix = self._compute_simplified_kernel(X, self.X_train)
        return self.classifier.predict(kernel_matrix)

    def score(self, X, y):
        """Compute accuracy score"""
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

    def get_segc_statistics(self):
        """Get SEGC statistics and parameter analysis"""
        return self.segc_stats


def demonstrate_segc_parameters():
    """Demonstrate SEGC parameter identification and optimization"""

    print("🔍 SEGC Parameter Analysis Demo")
    print("="*60)

    # Create sample data
    np.random.seed(42)
    X = np.random.randn(100, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # Test different SEGC configurations
    configs = [
        {'n_qubits': 3, 'k_coarse': 2, 'shots': 128, 'max_iterations': 2},
        {'n_qubits': 4, 'k_coarse': 2, 'shots': 256, 'max_iterations': 3},
        {'n_qubits': 5, 'k_coarse': 3, 'shots': 512, 'max_iterations': 4}
    ]

    results = []

    for i, config in enumerate(configs):
        print(f"\n--- Testing Configuration {i+1} ---")
        print(f"Parameters: {config}")

        start_time = time.time()

        # Train QSVM with SEGC
        qsvm = OptimizedQuantumSVMWithSEGC(use_segc=True, segc_params=config)
        qsvm.fit(X_train, y_train)

        # Evaluate
        train_score = qsvm.score(X_train, y_train)
        test_score = qsvm.score(X_test, y_test)
        training_time = time.time() - start_time

        # Get SEGC statistics
        segc_stats = qsvm.get_segc_statistics()

        result = {
            'config': config,
            'train_score': train_score,
            'test_score': test_score,
            'training_time': training_time,
            'segc_stats': segc_stats
        }

        results.append(result)

        print(f"Train Accuracy: {train_score:.4f}")
        print(f"Test Accuracy: {test_score:.4f}")
        print(f"Training Time: {training_time:.2f}s")

        if segc_stats and 'parameter_analysis' in segc_stats:
            analysis = segc_stats['parameter_analysis']
            print(
                f"SEGC Recommended qubits: {analysis.get('recommended_qubits', 'N/A')}")
            print(
                f"SEGC Recommended k_coarse: {analysis.get('recommended_k_coarse', 'N/A')}")

    # Visualize results
    visualize_segc_parameter_analysis(results)

    return results


def visualize_segc_parameter_analysis(results):
    """Visualize SEGC parameter analysis results"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Performance comparison
    config_names = [f"Config {i+1}" for i in range(len(results))]
    test_scores = [r['test_score'] for r in results]
    training_times = [r['training_time'] for r in results]

    ax1.bar(config_names, test_scores, alpha=0.7, color='skyblue')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Performance Comparison')
    ax1.set_ylim(0, 1)

    # Add accuracy labels
    for i, score in enumerate(test_scores):
        ax1.text(i, score + 0.01, f'{score:.3f}', ha='center', va='bottom')

    # Training time comparison
    ax2.bar(config_names, training_times, alpha=0.7, color='lightcoral')
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_title('Training Time Comparison')

    # Parameter visualization
    n_qubits = [r['config']['n_qubits'] for r in results]
    k_coarse = [r['config']['k_coarse'] for r in results]
    shots = [r['config']['shots'] for r in results]

    x = np.arange(len(results))
    width = 0.25

    ax3.bar(x - width, n_qubits, width, label='n_qubits', alpha=0.7)
    ax3.bar(x, k_coarse, width, label='k_coarse', alpha=0.7)
    ax3.bar(x + width, np.array(shots)/200,
            width, label='shots/200', alpha=0.7)

    ax3.set_xlabel('Configuration')
    ax3.set_ylabel('Parameter Value')
    ax3.set_title('SEGC Parameter Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(config_names)
    ax3.legend()

    # Summary statistics
    ax4.axis('off')
    summary_text = "SEGC Parameter Analysis Summary:\n\n"

    best_config = max(results, key=lambda x: x['test_score'])
    best_idx = results.index(best_config)

    summary_text += f"Best Configuration: Config {best_idx + 1}\n"
    summary_text += f"Best Test Accuracy: {best_config['test_score']:.4f}\n"
    summary_text += f"Training Time: {best_config['training_time']:.2f}s\n\n"

    summary_text += "Key Parameters:\n"
    summary_text += f"• n_qubits: {best_config['config']['n_qubits']}\n"
    summary_text += f"• k_coarse: {best_config['config']['k_coarse']}\n"
    summary_text += f"• shots: {best_config['config']['shots']}\n"
    summary_text += f"• max_iterations: {best_config['config']['max_iterations']}\n\n"

    summary_text += "SEGC Benefits:\n"
    summary_text += "• Adaptive parameter selection\n"
    summary_text += "• Early stopping for efficiency\n"
    summary_text += "• Subspace exploration\n"
    summary_text += "• Hyperparameter optimization"

    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))

    plt.tight_layout()
    plt.suptitle('SEGC Parameter Analysis for QSVM', fontsize=16, y=0.98)
    plt.show()


def main():
    """Main demonstration function"""
    print("🚀 Optimized SEGC-QSVM Parameter Analysis")
    print("="*60)

    try:
        # Run parameter analysis demo
        results = demonstrate_segc_parameters()

        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)

        # Find optimal configuration
        best_result = max(results, key=lambda x: x['test_score'])
        best_idx = results.index(best_result)

        print(f"\n🎯 Optimal SEGC Configuration:")
        print(f"Configuration {best_idx + 1}: {best_result['config']}")
        print(f"Test Accuracy: {best_result['test_score']:.4f}")
        print(f"Training Time: {best_result['training_time']:.2f}s")

        # Show parameter recommendations
        if best_result['segc_stats'] and 'parameter_analysis' in best_result['segc_stats']:
            analysis = best_result['segc_stats']['parameter_analysis']
            print(f"\n📊 SEGC Parameter Recommendations:")
            print(
                f"• Recommended qubits: {analysis.get('recommended_qubits', 'N/A')}")
            print(
                f"• Recommended k_coarse: {analysis.get('recommended_k_coarse', 'N/A')}")
            print(
                f"• Recommended shots: {analysis.get('recommended_shots', 'N/A')}")
            print(
                f"• Data complexity: {analysis.get('data_complexity', 'N/A')}")

        print("\n✅ Demo completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
