"""
Created on 17/07/2025

@author: Aryan

Filename: main.py

Relative Path: dynamic-oseq-qsvm/main.py
"""

from qsvm import OptimizedQuantumSVMWithSEGC
from data_utils import load_financial_data


def main(ticker="GOOGL"):
    print("SEGC Enhanced QSVM with Dynamic Feature Oracle")
    print("="*60)

    (X_train, X_test, y_train, y_test), features = load_financial_data(
        csv_path=f"ticker/{ticker}/{ticker}_data.csv")
    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    qsvm = OptimizedQuantumSVMWithSEGC()
    qsvm.fit(X_train, y_train, features)

    train_acc = qsvm.score(X_train, y_train)
    test_acc = qsvm.score(X_test,  y_test)

    print("\n Final Results")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test  Accuracy: {test_acc:.4f}")
    print(f"Training Time:  {qsvm.training_time:.2f}s")
    print(
        f"Selected Features: {', '.join(qsvm.segc_stats['selected_features'])}")

    # Optional visualisation (unchanged)
    qsvm.plot_results(features, X_train, y_train)
    print("\n Demo completed successfully!")


if __name__ == "__main__":
    main(ticker="AMZN")  # Change ticker as needed
