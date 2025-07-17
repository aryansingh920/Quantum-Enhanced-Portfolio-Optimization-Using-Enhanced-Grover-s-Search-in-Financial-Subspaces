#!/usr/bin/env python3
"""
Fixed SEGC-enhanced QSVM with dynamic feature-based oracle and hyperparameter adaptation
Created on 01/07/2025
@author: Aryan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from collections import defaultdict
import time
import warnings
warnings.filterwarnings('ignore')

# --- SEGC Functions ---


def diffuser(n):
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


def coarse_oracle(target_bits: str, k: int) -> QuantumCircuit:
    """Marks states whose bottom-k bits match the target's bottom-k bits."""
    n_qubits = len(target_bits)  # Derive n_qubits from target_bits length
    qc = QuantumCircuit(n_qubits)
    bottom_k = target_bits[-k:]

    for i, bit in enumerate(reversed(bottom_k)):
        if bit == '0':
            qc.x(i)

    if k == 1:
        qc.z(0)
    else:
        qc.h(k - 1)
        if k > 1:
            qc.mcx(list(range(k - 1)), k - 1)
        qc.h(k - 1)

    for i, bit in enumerate(reversed(bottom_k)):
        if bit == '0':
            qc.x(i)
    return qc


def fine_oracle_subspace(target_bits: str, k: int) -> QuantumCircuit:
    """Phase-flips the exact target state within the high (n-k) qubits."""
    n_qubits = len(target_bits)  # Derive n_qubits from target_bits length
    qc = QuantumCircuit(n_qubits)
    high_bits = target_bits[:-k]
    m = len(high_bits)

    if m == 0:  # No high bits to work with
        return qc

    for idx, bit in enumerate(reversed(high_bits)):
        if bit == '0':
            qc.x(k + idx)

    if m == 1:
        qc.z(k)
    else:
        qc.h(k + m - 1)
        if m > 1:
            qc.mcx(list(range(k, k + m - 1)), k + m - 1)
        qc.h(k + m - 1)

    for idx, bit in enumerate(reversed(high_bits)):
        if bit == '0':
            qc.x(k + idx)
    return qc


def diffuser_subspace(n: int, k: int) -> QuantumCircuit:
    """Inversion about mean on the high (n-k) qubits."""
    qc = QuantumCircuit(n)
    high = list(range(k, n))

    if len(high) == 0:  # No high qubits to work with
        return qc

    qc.h(high)
    qc.x(high)

    if len(high) > 1:
        qc.h(high[-1])
        qc.mcx(high[:-1], high[-1])
    elif len(high) == 1:
        qc.z(high[0])

    qc.x(high)
    qc.h(high)
    return qc

# --- Optimized SEGC Searcher ---


class OptimizedSEGCSearcher:
    def __init__(self, n_qubits=4, k_coarse=2, shots=256, max_iterations=3, decay_rate=0.9, convergence_threshold=0.01):
        self.n_qubits = n_qubits
        self.k_coarse = min(k_coarse, n_qubits - 1)  # Ensure k_coarse is valid
        self.shots = shots
        self.max_iterations = max_iterations
        self.decay_rate = decay_rate
        self.convergence_threshold = convergence_threshold
        self.simulator = AerSimulator()
        self.search_history = []
        self.subspace_scores = defaultdict(float)
        self.last_success_rates = []
        self.parameter_analysis = {}

    def analyze_parameters(self, data_characteristics):
        """Analyze and set optimal SEGC parameters based on data characteristics."""
        data_size = data_characteristics.get('size', 100)
        data_complexity = data_characteristics.get('complexity', 'medium')

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

        if data_complexity == 'high':
            optimal_shots *= 2
            optimal_k = min(optimal_k + 1, optimal_qubits - 1)
        elif data_complexity == 'low':
            optimal_shots = max(optimal_shots // 2, 64)

        # Ensure k_coarse is valid
        optimal_k = min(optimal_k, optimal_qubits - 1)
        optimal_k = max(optimal_k, 1)

        self.n_qubits = optimal_qubits
        self.k_coarse = optimal_k
        self.shots = optimal_shots

        self.parameter_analysis = {
            'recommended_qubits': optimal_qubits,
            'recommended_k_coarse': optimal_k,
            'recommended_shots': optimal_shots,
            'data_size': data_size,
            'data_complexity': data_complexity,
            'convergence_strategy': 'early_stopping'
        }
        return self.parameter_analysis

    def adaptive_iteration_count(self, phase, subspace_score=1.0):
        """Dynamically adjust iteration counts."""
        if phase == "coarse":
            N = 2**self.n_qubits
            M = 2**(self.n_qubits - self.k_coarse)
            base_iterations = max(1, int(np.pi/4 * np.sqrt(N / M)))
            if self.search_history:
                avg_score = np.mean([h['best_subspace_score']
                                    for h in self.search_history])
                if avg_score > 2.0:
                    base_iterations = max(1, base_iterations - 1)
        else:
            N_sub = 2**(self.n_qubits - self.k_coarse)
            if N_sub == 0:
                return 1
            M_sub = 1
            base_iterations = max(1, int(np.pi/4 * np.sqrt(N_sub / M_sub)))
            if subspace_score > 1.5:
                base_iterations += 1
            elif subspace_score > 3.0:
                base_iterations += 2
        return base_iterations

    def build_adaptive_circuit(self, target_bits, iteration_num):
        """Build SEGC circuit with adaptive parameters."""
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        qc.h(range(self.n_qubits))

        # Coarse search
        r1 = self.adaptive_iteration_count("coarse")
        print(f"Iteration {iteration_num + 1}: Coarse iterations: {r1}")
        for _ in range(r1):
            # Create oracle with correct number of qubits
            oracle = coarse_oracle(target_bits, self.k_coarse)
            qc.append(oracle, range(self.n_qubits))
            qc.append(diffuser(self.n_qubits), range(self.n_qubits))

        qc.barrier()

        # Fine search
        target_subspace = target_bits[-self.k_coarse:]
        subspace_score = self.subspace_scores.get(target_subspace, 1.0)
        r2 = self.adaptive_iteration_count("fine", subspace_score)
        print(
            f"Iteration {iteration_num + 1}: Fine iterations: {r2} (subspace score: {subspace_score:.2f})")

        for _ in range(r2):
            # Create oracle with correct number of qubits
            oracle = fine_oracle_subspace(target_bits, self.k_coarse)
            qc.append(oracle, range(self.n_qubits))
            qc.append(diffuser_subspace(self.n_qubits,
                      self.k_coarse), range(self.n_qubits))

        qc.measure(range(self.n_qubits), range(self.n_qubits))
        return qc

    def analyze_subspace_distribution(self, counts, target_bits):
        """Analyze subspace distribution based on measurement results."""
        subspace_counts = defaultdict(int)
        target_subspace = target_bits[-self.k_coarse:]
        for bitstring, count in counts.items():
            subspace = bitstring[-self.k_coarse:]
            subspace_counts[subspace] += count

        total_shots = sum(counts.values())
        subspace_scores = {}
        for subspace, count in subspace_counts.items():
            probability = count / total_shots
            expected_uniform = 1 / (2**self.k_coarse)
            score = probability / expected_uniform
            subspace_scores[subspace] = score
            if subspace == target_subspace:
                score *= 1.5
        return subspace_scores, subspace_counts

    def search_with_feedback(self, target_value, feature_indices, X_train, y_train, X_val, y_val):
        """SEGC search with feedback based on QSVM accuracy."""
        target_bits = format(target_value, f"0{self.n_qubits}b")
        print(
            f"SEGC Search for target: {target_value} = {target_bits} (Features: {feature_indices})")
        print(
            f"Coarse on bottom {self.k_coarse} bits, fine on top {self.n_qubits - self.k_coarse} bits")
        print("-" * 60)

        best_result = None
        best_success_rate = 0.0

        for iteration in range(self.max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")
            qc = self.build_adaptive_circuit(target_bits, iteration)
            result = self.simulator.run(
                transpile(qc, self.simulator), shots=self.shots).result()
            counts = result.get_counts()

            # Evaluate QSVM accuracy for this subspace
            selected_features = X_train[:, feature_indices]
            val_features = X_val[:, feature_indices]
            train_kernel = compute_simplified_kernel(selected_features)
            val_kernel = compute_simplified_kernel(
                val_features, selected_features)
            svm = SVC(kernel='precomputed', C=1.0)
            svm.fit(train_kernel, y_train)
            success_rate = svm.score(val_kernel, y_val)
            target_count = int(success_rate * self.shots)

            self.last_success_rates.append(success_rate)
            print(
                f"QSVM Validation Accuracy: {success_rate:.3f} (Target count: {target_count}/{self.shots})")

            subspace_scores, subspace_counts = self.analyze_subspace_distribution(
                counts, target_bits)
            best_subspace = max(subspace_scores.keys(),
                                key=lambda x: subspace_scores[x])
            best_subspace_score = subspace_scores[best_subspace]
            print(
                f"Best subspace: {best_subspace} (score: {best_subspace_score:.2f})")

            for subspace in self.subspace_scores:
                self.subspace_scores[subspace] *= self.decay_rate
            for subspace, score in subspace_scores.items():
                self.subspace_scores[subspace] += score

            iteration_data = {
                'iteration': iteration + 1,
                'success_rate': success_rate,
                'target_count': target_count,
                'best_subspace': best_subspace,
                'best_subspace_score': best_subspace_score,
                'feature_indices': feature_indices,
                'counts': counts.copy()
            }
            self.search_history.append(iteration_data)

            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_result = iteration_data

            if success_rate >= 0.85:
                print("High success rate achieved! Stopping early.")
                break
            if len(self.last_success_rates) >= 3 and self.last_success_rates[-3] > self.last_success_rates[-2] > self.last_success_rates[-1]:
                print("Warning: Over-amplification suspected. Reducing fine iterations.")
                self.max_iterations = iteration + 1
                break
            if iteration > 2 and max([h['success_rate'] for h in self.search_history[-3:]]) - min([h['success_rate'] for h in self.search_history[-3:]]) < self.convergence_threshold:
                print("Convergence detected. Stopping.")
                break

        return best_result, self.search_history

# --- Optimized QSVM with SEGC ---


def compute_simplified_kernel(X1, X2=None):
    """Compute simplified quantum-inspired kernel matrix."""
    if X2 is None:
        X2 = X1
    gamma = 1.0 / X1.shape[1]
    X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
    X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)
    distances = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
    kernel = np.exp(-gamma * distances)
    return kernel


class OptimizedQuantumSVMWithSEGC:
    def __init__(self, segc_params=None):
        self.classifier = None
        self.X_train_selected = None  # Store selected training features
        self.y_train = None
        self.segc_stats = None
        self.training_time = 0
        self.feature_indices = None
        if segc_params is None:
            segc_params = {
                'n_qubits': 4,
                'k_coarse': 2,
                'shots': 256,
                'max_iterations': 3,
                'decay_rate': 0.9,
                'convergence_threshold': 0.01
            }
        self.segc_searcher = OptimizedSEGCSearcher(**segc_params)

    def compute_feature_target_corr(self, X, y):
        """Compute correlation between features and target."""
        correlations = []
        for i in range(X.shape[1]):
            corr = np.corrcoef(X[:, i], y)[0, 1]
            correlations.append(abs(corr))
        return np.array(correlations)

    def map_features_to_target_value(self, X, y, top_k=2):
        """Map top correlated features to a target value."""
        corrs = self.compute_feature_target_corr(X, y)
        ranked = np.argsort(-corrs)[:top_k]

        # Ensure we have enough bits for the target value
        max_bits = self.segc_searcher.n_qubits
        bitstring = ''
        for i, feat_idx in enumerate(ranked):
            if len(bitstring) + 3 <= max_bits:
                bitstring += format(feat_idx, '03b')
            else:
                remaining_bits = max_bits - len(bitstring)
                if remaining_bits > 0:
                    bitstring += format(feat_idx, f'0{remaining_bits}b')
                break

        # Pad with zeros if needed
        while len(bitstring) < max_bits:
            bitstring += '0'

        # Truncate if too long
        bitstring = bitstring[:max_bits]

        return int(bitstring, 2), ranked

    def fit(self, X, y, features):
        """Fit the optimized QSVM with SEGC."""
        start_time = time.time()
        self.y_train = y

        # Analyze data characteristics for SEGC parameters
        data_characteristics = {
            'size': len(X),
            'complexity': 'high' if X.shape[1] > 4 else 'medium'
        }
        params = self.segc_searcher.analyze_parameters(data_characteristics)
        print(f"SEGC Parameter Analysis: {params}")

        # Split for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Compute target value based on feature importance
        target_value, feature_indices = self.map_features_to_target_value(
            X_train, y_train, top_k=params['recommended_k_coarse'])
        self.feature_indices = feature_indices

        # Run SEGC search
        best_result, search_history = self.segc_searcher.search_with_feedback(
            target_value, feature_indices, X_train, y_train, X_val, y_val)

        # Train final SVM with best features
        # Store selected features
        self.X_train_selected = X_train[:, feature_indices]
        kernel_matrix = compute_simplified_kernel(self.X_train_selected)
        C = self.optimize_hyperparameters(self.X_train_selected, y_train)
        print(f"Training final SVM with C={C:.3f}...")
        self.classifier = SVC(kernel='precomputed', C=C)
        self.classifier.fit(kernel_matrix, y_train)

        self.training_time = time.time() - start_time
        self.segc_stats = {
            'parameter_analysis': params,
            'search_history': search_history,
            'subspace_scores': dict(self.segc_searcher.subspace_scores),
            'selected_features': [features[i] for i in feature_indices]
        }
        return self

    def optimize_hyperparameters(self, X, y):
        """Optimize SVM hyperparameters."""
        C_values = [0.1, 1.0, 10.0]
        best_C = 1.0
        best_score = 0.0
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42)
        for C in C_values:
            try:
                train_kernel = compute_simplified_kernel(X_train)
                val_kernel = compute_simplified_kernel(X_val, X_train)
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
        """Make predictions."""
        if self.classifier is None:
            raise ValueError("Model not fitted!")
        X_selected = X[:, self.feature_indices]
        kernel_matrix = compute_simplified_kernel(
            X_selected, self.X_train_selected)
        return self.classifier.predict(kernel_matrix)

    def score(self, X, y):
        """Compute accuracy score."""
        predictions = self.predict(X)
        return accuracy_score(y, predictions)

    def plot_results(self, features, X, y):
        """Visualize SEGC-QSVM results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # Success rate over iterations
        iterations = [h['iteration']
                      for h in self.segc_stats['search_history']]
        success_rates = [h['success_rate']
                         for h in self.segc_stats['search_history']]
        ax1.plot(iterations, success_rates, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Validation Accuracy')
        ax1.set_title('SEGC-QSVM Convergence')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # Feature importance
        corrs = self.compute_feature_target_corr(X, y)
        ax2.bar(features, corrs, color='skyblue', alpha=0.7)
        ax2.set_xlabel('Features')
        ax2.set_ylabel('Correlation with Regime')
        ax2.set_title('Feature Importance')
        ax2.tick_params(axis='x', rotation=45)

        # Subspace scores
        subspace_scores = self.segc_stats['subspace_scores']
        if subspace_scores:
            subspaces = list(subspace_scores.keys())[:5]
            scores = [subspace_scores[sub] for sub in subspaces]
            ax3.bar(subspaces, scores, color='lightcoral', alpha=0.7)
            ax3.set_xlabel('Subspace')
            ax3.set_ylabel('Score')
            ax3.set_title('Top Subspace Scores')
            ax3.tick_params(axis='x', rotation=45)

        # Summary
        ax4.axis('off')
        summary_text = f"""SEGC-QSVM Summary
Selected Features: {', '.join(self.segc_stats['selected_features'])}
Best Validation Accuracy: {max(success_rates):.3f}
Training Time: {self.training_time:.2f}s
SEGC Parameters:
• n_qubits: {self.segc_stats['parameter_analysis']['recommended_qubits']}
• k_coarse: {self.segc_stats['parameter_analysis']['recommended_k_coarse']}
• shots: {self.segc_stats['parameter_analysis']['recommended_shots']}
"""
        ax4.text(0.1, 0.5, summary_text, fontsize=12,
                 fontfamily='monospace', verticalalignment='center')
        plt.tight_layout()
        plt.suptitle('SEGC-QSVM Analysis', fontsize=16, y=0.98)
        plt.show()


def main():
    """Main function to demonstrate SEGC-enhanced QSVM."""
    print("🚀 SEGC-Enhanced QSVM with Dynamic Feature Oracle")
    print("="*60)

    try:
        # Create sample data for demonstration (since we don't have the CSV file)
        print("Creating sample financial data for demonstration...")
        np.random.seed(42)
        n_samples = 355

        # Generate sample financial features
        dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
        ma5_scaled = np.random.randn(n_samples)
        ma20_scaled = np.random.randn(n_samples)
        rsi_scaled = np.random.randn(n_samples)
        volatility_scaled = np.random.randn(n_samples)

        # Generate regime labels with some correlation to features
        regime = np.where(ma5_scaled + ma20_scaled > 0, 1, 0)


        # Load your own CSV
        print("Loading data from CSV...")
        df = pd.read_csv("ticker/GOOGL/GOOGL_data.csv", parse_dates=[
                        'Date'])  # Adjust if 'Date' not present

        # Drop NaNs if necessary
        df = df.dropna()

        # # Define your features and label (match exact column names)
        # features = ['ma5_scaled', 'ma20_scaled', 'rsi_scaled', 'volatility_scaled']
        # X = df[features].values
        # y = df['regime'].values.astype(int)
        # y = np.where(y == 0, -1, 1)  # For binary classification (-1, +1)


        # Use the sample data
        features = ['ma5_scaled', 'ma20_scaled',
                    'rsi_scaled', 'volatility_scaled']
        X = df[features].values
        y = df['regime'].values.astype(int)
        y = np.where(y == 0, -1, 1)  # Ensure binary classes

        # Preprocess
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42)
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

        # Train SEGC-QSVM
        qsvm = OptimizedQuantumSVMWithSEGC()
        qsvm.fit(X_train, y_train, features)

        # Evaluate
        train_score = qsvm.score(X_train, y_train)
        test_score = qsvm.score(X_test, y_test)
        print(f"\n🎯 Final Results")
        print(f"Train Accuracy: {train_score:.4f}")
        print(f"Test Accuracy: {test_score:.4f}")
        print(f"Training Time: {qsvm.training_time:.2f}s")
        print(
            f"Selected Features: {', '.join(qsvm.segc_stats['selected_features'])}")

        # Visualize
        qsvm.plot_results(features, X_scaled, y)

        print("\n✅ Demo completed successfully!")

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
