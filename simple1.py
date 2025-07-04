#!/usr/bin/env python3
"""
Simple usage example for QSVM with SEGC integration

This script shows the basic usage pattern for integrating SEGC
with Quantum Support Vector Machines.
"""

import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Import the modules (make sure these files are in your directory)
from segc import SEGCSearcher
from qsvm_with_segc import QuantumSVMWithSEGC


def simple_qsvm_segc_example():
    """
    Simple example showing how to use QSVM with SEGC
    """
    print("Simple QSVM with SEGC Example")
    print("-" * 40)

    # 1. Generate some data
    X, y = make_moons(n_samples=60, noise=0.1, random_state=42)

    # 2. Preprocess the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 3. Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    # 4. Create and train QSVM with SEGC
    print("\nTraining QSVM with SEGC...")
    qsvm = QuantumSVMWithSEGC(
        n_qubits=2,                    # Number of qubits for the quantum circuit
        # Type of entanglement ('linear', 'circular', 'full')
        entanglement_type='linear',
        depth=1,                       # Depth of the quantum circuit
        use_segc=True,                 # Enable SEGC optimization
        segc_params={                  # SEGC parameters
            'n_qubits': 5,
            'k_coarse': 2,
            'shots': 512,
            'max_iterations': 3
        }
    )

    # 5. Train the model
    qsvm.fit(X_train, y_train)

    # 6. Make predictions
    y_pred = qsvm.predict(X_test)

    # 7. Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {accuracy:.4f}")

    # 8. Get SEGC statistics
    segc_stats = qsvm.get_segc_statistics()
    if segc_stats:
        print(
            f"SEGC optimizations performed: {len(segc_stats['search_history'])}")
        print(f"SEGC parameters used: {segc_stats['segc_params']}")

    # 9. Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return qsvm, accuracy


def compare_with_without_segc():
    """
    Compare QSVM performance with and without SEGC
    """
    print("\n\nComparison: QSVM with vs without SEGC")
    print("-" * 50)

    # Generate data
    X, y = make_moons(n_samples=80, noise=0.1, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # QSVM without SEGC
    print("Training QSVM without SEGC...")
    qsvm_no_segc = QuantumSVMWithSEGC(
        n_qubits=2,
        entanglement_type='linear',
        depth=1,
        use_segc=False  # Disable SEGC
    )
    qsvm_no_segc.fit(X_train, y_train)
    accuracy_no_segc = qsvm_no_segc.score(X_test, y_test)

    # QSVM with SEGC
    print("Training QSVM with SEGC...")
    qsvm_with_segc = QuantumSVMWithSEGC(
        n_qubits=2,
        entanglement_type='linear',
        depth=1,
        use_segc=True,  # Enable SEGC
        segc_params={
            'n_qubits': 5,
            'k_coarse': 2,
            'shots': 512,
            'max_iterations': 3
        }
    )
    qsvm_with_segc.fit(X_train, y_train)
    accuracy_with_segc = qsvm_with_segc.score(X_test, y_test)

    # Compare results
    print(f"\nResults:")
    print(f"QSVM without SEGC: {accuracy_no_segc:.4f}")
    print(f"QSVM with SEGC:    {accuracy_with_segc:.4f}")
    print(f"Improvement:       {accuracy_with_segc - accuracy_no_segc:+.4f}")

    return accuracy_no_segc, accuracy_with_segc


def test_different_segc_params():
    """
    Test different SEGC parameter configurations
    """
    print("\n\nTesting Different SEGC Parameters")
    print("-" * 50)

    # Generate data
    X, y = make_moons(n_samples=60, noise=0.1, random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Different SEGC configurations to test
    configurations = [
        {'n_qubits': 4, 'k_coarse': 1, 'shots': 256, 'max_iterations': 2},
        {'n_qubits': 5, 'k_coarse': 2, 'shots': 512, 'max_iterations': 3},
        {'n_qubits': 6, 'k_coarse': 2, 'shots': 1024, 'max_iterations': 4},
    ]

    results = []

    for i, config in enumerate(configurations):
        print(f"\nTesting configuration {i+1}: {config}")

        qsvm = QuantumSVMWithSEGC(
            n_qubits=2,
            entanglement_type='linear',
            depth=1,
            use_segc=True,
            segc_params=config
        )

        qsvm.fit(X_train, y_train)
        accuracy = qsvm.score(X_test, y_test)

        results.append({
            'config': config,
            'accuracy': accuracy
        })

        print(f"Accuracy: {accuracy:.4f}")

    # Find best configuration
    best_result = max(results, key=lambda x: x['accuracy'])
    print(f"\nBest configuration:")
    print(f"Parameters: {best_result['config']}")
    print(f"Accuracy: {best_result['accuracy']:.4f}")

    return results


def main():
    """
    Main function running all examples
    """
    print("🚀 QSVM with SEGC - Simple Usage Examples")
    print("=" * 60)

    try:
        # Example 1: Basic usage
        qsvm, accuracy = simple_qsvm_segc_example()

        # Example 2: Comparison
        acc_no_segc, acc_with_segc = compare_with_without_segc()

        # Example 3: Parameter testing
        param_results = test_different_segc_params()

        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        print(f"✅ Basic QSVM with SEGC worked: {accuracy:.4f} accuracy")
        print(f"✅ SEGC improvement: {acc_with_segc - acc_no_segc:+.4f}")
        print(f"✅ Tested {len(param_results)} different SEGC configurations")
        print("\n🎉 All examples completed successfully!")

    except Exception as e:
        print(f"❌ Example failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
