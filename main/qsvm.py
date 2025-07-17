"""
Created on 17/07/2025

@author: Aryan

Filename: qsvm.py

Relative Path: dynamic-oseq-qsvm/qsvm.py
"""

import time
import numpy as np
from collections import defaultdict
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- SEGC primitives ---
from segc.segc import diffuser, coarse_oracle, fine_oracle_subspace, diffuser_subspace, SEGCSearcher

# --- Kernel ---
from kernel_utils import compute_simplified_kernel

class OptimizedSEGCSearcher(SEGCSearcher):
    def __init__(self, n_qubits=4, k_coarse=2, shots=256,
                 max_iterations=3, decay_rate=0.9, convergence_threshold=0.01):
        super().__init__(n_qubits, k_coarse, shots, max_iterations, decay_rate)
        self.convergence_threshold = convergence_threshold
        self.parameter_analysis = {}

    def analyze_parameters(self, data_characteristics):
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


class OptimizedQuantumSVMWithSEGC:
    def __init__(self, segc_params=None):
        if segc_params is None:
            segc_params = dict(n_qubits=4, k_coarse=2, shots=256,
                               max_iterations=3, decay_rate=0.9,
                               convergence_threshold=0.01)
        self.segc_searcher = OptimizedSEGCSearcher(**segc_params)
        self.classifier = None
        self.X_train_selected = None
        self.y_train = None
        self.segc_stats = None
        self.training_time = 0
        self.feature_indices = None

    def compute_feature_target_corr(self, X, y):
        correlations = [abs(np.corrcoef(X[:, i], y)[0, 1])
                        for i in range(X.shape[1])]
        return np.array(correlations)

    def map_features_to_target_value(self, X, y, top_k=2):
        corrs = self.compute_feature_target_corr(X, y)
        ranked = np.argsort(-corrs)[:top_k]

        max_bits = self.segc_searcher.n_qubits
        bitstring = ''
        for feat_idx in ranked:
            if len(bitstring) + 3 <= max_bits:
                bitstring += format(feat_idx, '03b')
            else:
                remaining = max_bits - len(bitstring)
                if remaining > 0:
                    bitstring += format(feat_idx, f'0{remaining}b')
                break

        bitstring = (bitstring + '0' * max_bits)[:max_bits]
        return int(bitstring, 2), ranked

    def optimize_hyperparameters(self, X, y):
        best_C, best_score = 1.0, 0.0
        for C in [0.1, 1.0, 10.0]:
            try:
                X_tr, X_val, y_tr, y_val = train_test_split(
                    X, y, test_size=0.2, random_state=42)
                train_kernel = compute_simplified_kernel(X_tr)
                val_kernel = compute_simplified_kernel(X_val, X_tr)
                clf = SVC(kernel='precomputed', C=C)
                clf.fit(train_kernel, y_tr)
                score = clf.score(val_kernel, y_val)
                if score > best_score:
                    best_score, best_C = score, C
            except:
                continue
        return best_C
    
    def fit(self, X, y, features):
        start_time = time.time()
        self.y_train = y

        data_characteristics = {
            'size': len(X),
            'complexity': 'high' if X.shape[1] > 4 else 'medium'
        }
        params = self.segc_searcher.analyze_parameters(data_characteristics)
        print(f"SEGC Parameter Analysis: {params}")

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42)
        target_value, feature_indices = self.map_features_to_target_value(
            X_train, y_train, top_k=params['recommended_k_coarse'])
        self.feature_indices = feature_indices

        # Updated: only pass target_value as expected by simplified SEGCSearcher
        best_result, search_history = self.segc_searcher.search_with_feedback(
            target_value)

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


    def predict(self, X):
        if self.classifier is None:
            raise ValueError("Model not fitted!")
        X_selected = X[:, self.feature_indices]
        kernel_matrix = compute_simplified_kernel(
            X_selected, self.X_train_selected)
        return self.classifier.predict(kernel_matrix)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))

    def plot_results(self, features, X, y):
        import matplotlib.pyplot as plt
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

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

        corrs = self.compute_feature_target_corr(X, y)
        ax2.bar(features, corrs, color='skyblue', alpha=0.7)
        ax2.set_xlabel('Features')
        ax2.set_ylabel('Correlation with Regime')
        ax2.set_title('Feature Importance')
        ax2.tick_params(axis='x', rotation=45)

        subspace_scores = self.segc_stats['subspace_scores']
        if subspace_scores:
            subspaces = list(subspace_scores.keys())[:5]
            scores = [subspace_scores[sub] for sub in subspaces]
            ax3.bar(subspaces, scores, color='lightcoral', alpha=0.7)
            ax3.set_xlabel('Subspace')
            ax3.set_ylabel('Score')
            ax3.set_title('Top Subspace Scores')
            ax3.tick_params(axis='x', rotation=45)

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
