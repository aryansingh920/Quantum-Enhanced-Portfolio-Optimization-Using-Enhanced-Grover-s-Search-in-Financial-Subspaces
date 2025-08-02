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
from sklearn.model_selection import KFold
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

    def analyze_parameters(self, data_characteristics, n_features):
        data_size = data_characteristics.get('size', 100)
        data_complexity = data_characteristics.get('complexity', 'medium')

        # Constrain n_qubits to the number of features
        # Use at most 5 qubits or number of features
        optimal_qubits = min(n_features, 5)
        if data_size < 50:
            optimal_qubits = min(n_features, 3)
            optimal_k = 2
            optimal_shots = 128
        elif data_size < 200:
            optimal_qubits = min(n_features, 4)
            optimal_k = 2
            optimal_shots = 256
        else:
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

        max_bits = min(self.segc_searcher.n_qubits,
                       X.shape[1])  # Respect feature count
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


    def fit(self, X, y, features, feature_mask=None):
        start_time = time.time()
        self.y_train = y

        if feature_mask is not None:
            X = X[:, feature_mask]
            self.feature_indices = feature_mask
            self.segc_stats = {'selected_features': [
                features[i] for i in feature_mask]}
        else:
            data_characteristics = {
                'size': len(X),
                'complexity': 'high' if X.shape[1] > 4 else 'medium'
            }
            params = self.segc_searcher.analyze_parameters(
                data_characteristics, n_features=X.shape[1])  # Pass n_features
            print(f"SEGC Parameter Analysis: {params}")

            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42)
            target_value, feature_indices = self.map_features_to_target_value(
                X_train, y_train, top_k=params['recommended_k_coarse'])
            self.feature_indices = feature_indices

            def evaluate_subspace_score(bitstring):
                selected_indices = [
                    i for i, bit in enumerate(bitstring) if bit == '1']
                if len(selected_indices) == 0:
                    return 0
                # Ensure indices are within bounds
                selected_indices = [
                    i for i in selected_indices if i < X_train.shape[1]]
                if len(selected_indices) == 0:
                    return 0
                X_sub = X_train[:, selected_indices]
                X_val_sub = X_val[:, selected_indices]
                try:
                    train_kernel = compute_simplified_kernel(X_sub)
                    val_kernel = compute_simplified_kernel(X_val_sub, X_sub)
                    clf = SVC(kernel='precomputed', C=1.0)
                    clf.fit(train_kernel, y_train)
                    return clf.score(val_kernel, y_val)
                except:
                    return 0

            best_bits, best_score, subspace_scores = self.segc_searcher.search_with_feedback(
                self.segc_searcher.n_qubits,
                self.segc_searcher.max_iterations,
                evaluate_subspace_score,
                format(target_value, f'0{self.segc_searcher.n_qubits}b'),
                self.segc_searcher.k_coarse
            )

            SEGCSearcher.plot_subspace_scores(subspace_scores)

            # Initialize segc_stats with search results
            self.segc_stats = {
                'parameter_analysis': params,
                'search_history': self.segc_searcher.search_history,
                'subspace_scores': dict(subspace_scores),
                'selected_features': [features[i] for i in feature_indices]
            }

        self.X_train_selected = X[:, self.feature_indices]
        kernel_matrix = compute_simplified_kernel(self.X_train_selected)
        C = self.optimize_hyperparameters(self.X_train_selected, y)
        print(f"Training final SVM with C={C:.3f}...")
        self.classifier = SVC(kernel='precomputed', C=C)
        self.classifier.fit(kernel_matrix, y)

        self.training_time = time.time() - start_time
        return self

    def predict(self, X, feature_mask=None):
        if self.classifier is None:
            raise ValueError("Model not fitted!")
        if feature_mask is not None:
            X_selected = X[:, feature_mask]
        else:
            X_selected = X[:, self.feature_indices]
        kernel_matrix = compute_simplified_kernel(
            X_selected, self.X_train_selected)
        return self.classifier.predict(kernel_matrix)

    def score(self, X, y, feature_mask=None, noise_level=0.0):
        if self.classifier is None:
            raise ValueError("Model not fitted!")
        if feature_mask is not None:
            X_selected = X[:, feature_mask]
        else:
            X_selected = X[:, self.feature_indices]

        # Perform k-fold cross-validation for subspace evaluation
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = []
        for train_idx, val_idx in kfold.split(X_selected):
            X_fold_train, X_fold_val = X_selected[train_idx], X_selected[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]

            # Compute kernel matrices for the fold
            train_kernel = compute_simplified_kernel(X_fold_train)
            val_kernel = compute_simplified_kernel(X_fold_val, X_fold_train)

            # Train a temporary SVM for this fold
            temp_clf = SVC(kernel='precomputed', C=self.classifier.C)
            temp_clf.fit(train_kernel, y_fold_train)

            # Evaluate on validation fold
            fold_score = accuracy_score(
                y_fold_val, temp_clf.predict(val_kernel))
            cv_scores.append(fold_score)

        # Average the cross-validation scores
        score = np.mean(cv_scores)

        # Inject noise if specified
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, 1)[0]
            score = max(0.0, min(1.0, score + noise))
        return score

    def plot_results(self, features, X, y, noise_level=0.0):
        import matplotlib.pyplot as plt
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        iterations = [h['iteration']
                      for h in self.segc_stats['search_history']] if self.segc_stats.get('search_history') else []
        success_rates = [h['success_rate']
                         for h in self.segc_stats['search_history']] if self.segc_stats.get('search_history') else []
        if iterations:
            ax1.plot(iterations, success_rates,
                     'bo-', linewidth=2, markersize=8)
            ax1.set_xlabel('Iteration')
            ax1.set_ylabel('Validation Accuracy')
            ax1.set_title('SEGC-QSVM Convergence')
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
        else:
            ax1.text(0.5, 0.5, 'No SEGC Search (Fixed Features)',
                     ha='center', va='center')
            ax1.axis('off')

        corrs = self.compute_feature_target_corr(X, y)
        ax2.bar(features, corrs, color='skyblue', alpha=0.7)
        ax2.set_xlabel('Features')
        ax2.set_ylabel('Correlation with Regime')
        ax2.set_title('Feature Importance')
        ax2.tick_params(axis='x', rotation=45)

        subspace_scores = self.segc_stats.get('subspace_scores', {})
        if subspace_scores:
            subspaces = list(subspace_scores.keys())[:5]
            scores = [subspace_scores[sub] for sub in subspaces]
            ax3.bar(subspaces, scores, color='lightcoral', alpha=0.7)
            ax3.set_xlabel('Subspace')
            ax3.set_ylabel('Score')
            ax3.set_title('Top Subspace Scores')
            ax3.tick_params(axis='x', rotation=45)
        else:
            ax3.text(0.5, 0.5, 'No Subspace Scores (Fixed Features)',
                     ha='center', va='center')
            ax3.axis('off')

        ax4.axis('off')
        summary_text = f"""SEGC-QSVM Summary
Selected Features: {', '.join(self.segc_stats['selected_features'])}
Training Time: {self.training_time:.2f}s
Noise Level: {noise_level:.3f}
"""
        if self.segc_stats.get('parameter_analysis'):
            summary_text += f"""SEGC Parameters:
• n_qubits: {self.segc_stats['parameter_analysis']['recommended_qubits']}
• k_coarse: {self.segc_stats['parameter_analysis']['recommended_k_coarse']}
• shots: {self.segc_stats['parameter_analysis']['recommended_shots']}
"""
        ax4.text(0.1, 0.5, summary_text, fontsize=12,
                 fontfamily='monospace', verticalalignment='center')
        plt.tight_layout()
        plt.suptitle('SEGC-QSVM Analysis', fontsize=16, y=0.98)
        plt.show()
