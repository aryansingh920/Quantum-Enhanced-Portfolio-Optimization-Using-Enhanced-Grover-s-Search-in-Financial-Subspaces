"""
Created on 17/07/2025

@author: Aryan

Filename: main.py

Relative Path: dynamic-oseq-qsvm/main.py
"""

from qsvm import OptimizedQuantumSVMWithSEGC
from data_utils import load_financial_data
from sklearn.model_selection import KFold
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
import numpy as np
import itertools
import random
import matplotlib.pyplot as plt


def main(ticker="GOOGL"):
    print("SEGC Enhanced QSVM with Dynamic Feature Oracle")
    print("="*60)

    # Load data
    (X_train_orig, X_test_orig, y_train_orig, y_test_orig), features = load_financial_data(
        csv_path=f"ticker/{ticker}/{ticker}_data.csv")

    # Combine original train/test sets to perform custom splitting
    X = np.vstack((X_train_orig, X_test_orig))
    y = np.hstack((y_train_orig, y_test_orig))

    # Split data: 60% train, 20% validation, 20% test
    from sklearn.model_selection import train_test_split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2
    print(
        f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}, Test samples: {len(X_test)}")

    # Initialize comparison results
    results = {}

    # --- SEGC with k-fold cross-validation ---
    print("\n=== Running SEGC with K-Fold Cross-Validation ===")
    kfold = KFold(n_splits=5, shuffle=True, random_state=42)
    segc_scores = []
    segc_features = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_train, y_train)):
        print(f"\nFold {fold + 1}")
        X_fold_train, y_fold_train = X_train[train_idx], y_train[train_idx]
        qsvm = OptimizedQuantumSVMWithSEGC()
        qsvm.fit(X_fold_train, y_fold_train, features)
        val_score = qsvm.score(X_val, y_val)
        segc_scores.append(val_score)
        segc_features.append(qsvm.segc_stats['selected_features'])
        print(f"Fold {fold + 1} Validation Accuracy: {val_score:.4f}")

    # Train final SEGC model on full training set
    qsvm = OptimizedQuantumSVMWithSEGC()
    qsvm.fit(X_train, y_train, features)
    test_acc_segc = qsvm.score(X_test, y_test)
    results['SEGC'] = {
        'test_acc': test_acc_segc,
        'train_acc': qsvm.score(X_train, y_train),
        'training_time': qsvm.training_time,
        'selected_features': qsvm.segc_stats['selected_features'],
        'cv_scores': segc_scores
    }
    print(f"\nSEGC Test Accuracy: {test_acc_segc:.4f}")

    # --- Grid Search ---
    print("\n=== Running Grid Search ===")
    all_feature_subsets = list(itertools.combinations(
        range(X.shape[1]), 2))  # All pairs of features
    best_grid_score = 0
    best_grid_features = None
    for feature_idx in all_feature_subsets:
        X_train_subset = X_train[:, feature_idx]
        X_val_subset = X_val[:, feature_idx]
        svm = SVC(kernel='rbf')
        svm.fit(X_train_subset, y_train)
        score = svm.score(X_val_subset, y_val)
        if score > best_grid_score:
            best_grid_score = score
            best_grid_features = feature_idx
    # Train final model with best features
    svm = SVC(kernel='rbf')
    svm.fit(X_train[:, best_grid_features], y_train)
    test_acc_grid = svm.score(X_test[:, best_grid_features], y_test)
    results['Grid Search'] = {
        'test_acc': test_acc_grid,
        'train_acc': svm.score(X_train[:, best_grid_features], y_train),
        'selected_features': [features[i] for i in best_grid_features]
    }
    print(f"Grid Search Test Accuracy: {test_acc_grid:.4f}")

    # --- Random Search ---
    print("\n=== Running Random Search ===")
    # Same number of evaluations as grid search
    n_iterations = len(all_feature_subsets)
    best_random_score = 0
    best_random_features = None
    for _ in range(n_iterations):
        feature_idx = random.sample(range(X.shape[1]), 2)
        X_train_subset = X_train[:, feature_idx]
        X_val_subset = X_val[:, feature_idx]
        svm = SVC(kernel='rbf')
        svm.fit(X_train_subset, y_train)
        score = svm.score(X_val_subset, y_val)
        if score > best_random_score:
            best_random_score = score
            best_random_features = feature_idx
    # Train final model with best features
    svm = SVC(kernel='rbf')
    svm.fit(X_train[:, best_random_features], y_train)
    test_acc_random = svm.score(X_test[:, best_random_features], y_test)
    results['Random Search'] = {
        'test_acc': test_acc_random,
        'train_acc': svm.score(X_train[:, best_random_features], y_train),
        'selected_features': [features[i] for i in best_random_features]
    }
    print(f"Random Search Test Accuracy: {test_acc_random:.4f}")

    # --- RFE ---
    print("\n=== Running RFE ===")
    estimator = SVC(kernel='linear')  # Use linear kernel for RFE
    rfe = RFE(estimator, n_features_to_select=2)
    rfe.fit(X_train, y_train)
    selected_features = np.where(rfe.support_)[0]
    svm = SVC(kernel='rbf')  # Use RBF for final model to match other methods
    svm.fit(X_train[:, selected_features], y_train)
    test_acc_rfe = svm.score(X_test[:, selected_features], y_test)
    results['RFE'] = {
        'test_acc': test_acc_rfe,
        'train_acc': svm.score(X_train[:, selected_features], y_train),
        'selected_features': [features[i] for i in selected_features]
    }
    print(f"RFE Test Accuracy: {test_acc_rfe:.4f}")

    # --- Print Comparison ---
    print("\n=== Comparison of Feature Selection Methods ===")
    print(f"{'Method':<15} {'Test Accuracy':<15} {'Train Accuracy':<15} {'Selected Features':<30}")
    print("-"*75)
    for method, result in results.items():
        train_acc = result['train_acc']
        test_acc = result['test_acc']
        selected_features = ', '.join(result['selected_features'])
        print(
            f"{method:<15} {test_acc:.4f}{' ':<8} {train_acc:.4f}{' ':<8} {selected_features:<30}")
        if method == 'SEGC':
            print(
                f"{'':<15} CV Scores: {', '.join([f'{s:.4f}' for s in result['cv_scores']]):<30}")
            print(f"{'':<15} Training Time: {result['training_time']:.2f}s")

    # --- Plot Comparison ---
    fig, ax = plt.subplots(figsize=(10, 6))
    methods = list(results.keys())
    test_accuracies = [results[method]['test_acc'] for method in methods]
    ax.bar(methods, test_accuracies, color=[
           'blue', 'green', 'orange', 'red'], alpha=0.7)
    ax.set_xlabel('Feature Selection Method')
    ax.set_ylabel('Test Accuracy')
    ax.set_title('Comparison of Feature Selection Methods')
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.show()

    # Plot results for SEGC
    qsvm.plot_results(features, X_train, y_train)
    print("\nDemo completed successfully!")


if __name__ == "__main__":
    main(ticker="JPM")  # Change ticker as needed
