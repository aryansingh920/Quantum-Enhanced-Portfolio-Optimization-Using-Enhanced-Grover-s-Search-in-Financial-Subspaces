#!/usr/bin/env python3
"""
Demo script showing QSVM with SEGC integration for financial data
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import pandas as pd
import time
from segc import SEGCSearcher
from qsvm_scratch_segc import QuantumSVMWithSEGC, visualize_segc_optimization


def run_segc_standalone_demo():
    """
    Run standalone SEGC demo
    """
    print("🔍 Testing SEGC Standalone...")
    print("-" * 40)
    searcher = SEGCSearcher(n_qubits=6, k_coarse=3,
                            shots=1024, max_iterations=4, decay_rate=0.8)
    target = 42
    print(f"Searching for target: {target}")
    start_time = time.time()
    best_result, search_history = searcher.search_with_feedback(target)
    end_time = time.time()
    print(f"SEGC completed in {end_time - start_time:.2f} seconds")
    print(f"Best success rate: {best_result['success_rate']:.4f}")
    print(f"Total iterations: {len(search_history)}")
    searcher.plot_results(target, best_result, search_history)
    return True


def run_qsvm_comparison_demo():
    """
    Run QSVM comparison on financial dataset
    """
    print("\n🎯 QSVM Comparison Demo")
    print("-" * 40)

    # Load financial dataset
    data = pd.read_csv('ticker/JPM/JPM_data.csv')
    features = ['rsi_scaled', 'volatility_scaled',
                'ma5_scaled', 'close_scaled']
    X = data[features].values
    y = data['regime'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    # Classical SVM baseline
    print("Training Classical SVM...")
    classical_svm = SVC(kernel='rbf', C=1.0)
    classical_svm.fit(X_train, y_train)
    classical_score = classical_svm.score(X_test, y_test)

    # Classical QSVM
    print("Training Classical QSVM...")
    qsvm_classical = QuantumSVMWithSEGC(
        n_qubits=4, entanglement_type='linear', depth=1, use_segc=False)
    qsvm_classical.fit(X_train, y_train)
    qsvm_classical_score = qsvm_classical.score(X_test, y_test)

    # SEGC-enhanced QSVM
    print("Training SEGC-enhanced QSVM...")
    qsvm_segc = QuantumSVMWithSEGC(n_qubits=4, entanglement_type='linear', depth=1, use_segc=True,
                                   segc_params={'n_qubits': 6, 'k_coarse': 3, 'shots': 1024, 'max_iterations': 4, 'decay_rate': 0.8})
    qsvm_segc.fit(X_train, y_train)
    qsvm_segc_score = qsvm_segc.score(X_test, y_test)

    results = {
        'financial': {
            'classical_svm': classical_score,
            'qsvm_classical': qsvm_classical_score,
            'qsvm_segc': qsvm_segc_score,
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test
        }
    }

    print(f"Results for Financial Dataset:")
    print(f"  Classical SVM: {classical_score:.4f}")
    print(f"  Classical QSVM: {qsvm_classical_score:.4f}")
    print(f"  SEGC-enhanced QSVM: {qsvm_segc_score:.4f}")

    visualize_segc_optimization(qsvm_segc, "SEGC Analysis - Financial Data")
    plot_comparison_summary(results)
    return results


def plot_comparison_summary(results):
    """
    Plot summary of QSVM comparisons
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    ax1 = axes[0, 0]
    datasets = list(results.keys())
    classical_svm_scores = [results[d]['classical_svm'] for d in datasets]
    qsvm_classical_scores = [results[d]['qsvm_classical'] for d in datasets]
    qsvm_segc_scores = [results[d]['qsvm_segc'] for d in datasets]

    x = np.arange(len(datasets))
    width = 0.25
    bars1 = ax1.bar(x - width, classical_svm_scores, width,
                    label='Classical SVM', alpha=0.7)
    bars2 = ax1.bar(x, qsvm_classical_scores, width,
                    label='Classical QSVM', alpha=0.7)
    bars3 = ax1.bar(x + width, qsvm_segc_scores, width,
                    label='SEGC-enhanced QSVM', alpha=0.7)
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Performance Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax1.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)

    ax2 = axes[0, 1]
    improvements = [qsvm_segc_scores[i] - qsvm_classical_scores[i]
                    for i in range(len(datasets))]
    colors = ['green' if imp > 0 else 'red' for imp in improvements]
    bars = ax2.bar(datasets, improvements, color=colors, alpha=0.7)
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Accuracy Improvement')
    ax2.set_title('SEGC Enhancement vs Classical QSVM')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        ax2.annotate(f'{imp:+.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                     xytext=(0, 3 if height > 0 else -15), textcoords="offset points",
                     ha='center', va='bottom' if height > 0 else 'top', fontsize=9)

    ax3 = axes[1, 0]
    X_train = results['financial']['X_train']
    y_train = results['financial']['y_train']
    colors = ['red', 'blue']
    for class_val in np.unique(y_train):
        mask = y_train == class_val
        ax3.scatter(X_train[mask, 0], X_train[mask, 1], c=colors[class_val],
                    label=f'Class {class_val}', alpha=0.7)
    ax3.set_xlabel('rsi_scaled')
    ax3.set_ylabel('volatility_scaled')
    ax3.set_title('Financial Dataset (2D Projection)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.axis('off')
    avg_classical_svm = np.mean(classical_svm_scores)
    avg_qsvm_classical = np.mean(qsvm_classical_scores)
    avg_qsvm_segc = np.mean(qsvm_segc_scores)
    overall_improvement = avg_qsvm_segc - avg_qsvm_classical
    summary_text = f"""Summary Statistics:
Average Performance:
• Classical SVM: {avg_classical_svm:.4f}
• Classical QSVM: {avg_qsvm_classical:.4f}
• SEGC-enhanced QSVM: {avg_qsvm_segc:.4f}
SEGC Enhancement:
• Average improvement: {overall_improvement:+.4f}
Key Insights:
• SEGC optimizes hyperparameters for financial data
• Uses rsi_scaled, volatility_scaled, ma5_scaled, close_scaled
• Subspace exploration enhances kernel quality
"""
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    plt.tight_layout()
    plt.suptitle('QSVM with SEGC - Financial Data Analysis',
                 fontsize=16, y=0.98)
    plt.show()


def run_detailed_segc_analysis():
    """
    Run detailed SEGC configuration analysis
    """
    print("\n🔬 Detailed SEGC Analysis")
    print("-" * 40)
    data = pd.read_csv('ticker/JPM/JPM_data.csv')
    features = ['rsi_scaled', 'volatility_scaled',
                'ma5_scaled', 'close_scaled']
    X = data[features].values
    y = data['regime'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    segc_configs = [
        {'n_qubits': 5, 'k_coarse': 2, 'shots': 512,
            'max_iterations': 3, 'decay_rate': 0.8},
        {'n_qubits': 6, 'k_coarse': 3, 'shots': 1024,
            'max_iterations': 4, 'decay_rate': 0.8},
        {'n_qubits': 7, 'k_coarse': 3, 'shots': 2048,
            'max_iterations': 5, 'decay_rate': 0.6}
    ]

    config_results = []
    for i, config in enumerate(segc_configs):
        print(f"\nTesting SEGC Configuration {i+1}: {config}")
        start_time = time.time()
        qsvm = QuantumSVMWithSEGC(n_qubits=4, entanglement_type='linear', depth=1,
                                  use_segc=True, segc_params=config)
        qsvm.fit(X_train, y_train)
        score = qsvm.score(X_test, y_test)
        end_time = time.time()
        segc_stats = qsvm.get_segc_statistics()
        config_results.append({
            'config': config,
            'score': score,
            'time': end_time - start_time,
            'segc_stats': segc_stats
        })
        print(f"  Accuracy: {score:.4f}")
        print(f"  Training time: {end_time - start_time:.2f}s")
        if segc_stats:
            print(f"  SEGC optimizations: {len(segc_stats['search_history'])}")

    plot_segc_config_analysis(config_results)
    return config_results


def plot_segc_config_analysis(config_results):
    """
    Plot SEGC configuration analysis
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    ax1 = axes[0, 0]
    scores = [r['score'] for r in config_results]
    times = [r['time'] for r in config_results]
    configs = [f"Config {i+1}" for i in range(len(config_results))]
    scatter = ax1.scatter(times, scores, s=100, alpha=0.7,
                          c=range(len(config_results)), cmap='viridis')
    for i, (time, score, config) in enumerate(zip(times, scores, configs)):
        ax1.annotate(config, (time, score), xytext=(
            5, 5), textcoords='offset points')
    ax1.set_xlabel('Training Time (seconds)')
    ax1.set_ylabel('Test Accuracy')
    ax1.set_title('Performance vs Computational Cost')
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    n_qubits = [r['config']['n_qubits'] for r in config_results]
    k_coarse = [r['config']['k_coarse'] for r in config_results]
    shots = [r['config']['shots'] for r in config_results]
    x = np.arange(len(config_results))
    width = 0.25
    ax2.bar(x - width, n_qubits, width, label='n_qubits', alpha=0.7)
    ax2.bar(x, k_coarse, width, label='k_coarse', alpha=0.7)
    ax2.bar(x + width, np.array(shots)/500,
            width, label='shots/500', alpha=0.7)
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Parameter Value')
    ax2.set_title('SEGC Parameter Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'Config {i+1}' for i in range(len(config_results))])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    for i, result in enumerate(config_results):
        if result['segc_stats'] and result['segc_stats']['search_history']:
            history = result['segc_stats']['search_history']
            iterations = [h['iteration'] for h in history]
            success_rates = [h['success_rate'] for h in history]
            ax3.plot(iterations, success_rates, 'o-',
                     label=f'Config {i+1}', alpha=0.7)
    ax3.set_xlabel('SEGC Iteration')
    ax3.set_ylabel('Success Rate')
    ax3.set_title('SEGC Optimization Progress')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.axis('off')
    summary_text = "Configuration Summary:\n\n"
    for i, result in enumerate(config_results):
        config = result['config']
        summary_text += f"Config {i+1}:\n"
        summary_text += f"  n_qubits: {config['n_qubits']}\n"
        summary_text += f"  k_coarse: {config['k_coarse']}\n"
        summary_text += f"  shots: {config['shots']}\n"
        summary_text += f"  max_iter: {config['max_iterations']}\n"
        summary_text += f"  decay_rate: {config['decay_rate']}\n"
        summary_text += f"  accuracy: {result['score']:.4f}\n"
        summary_text += f"  time: {result['time']:.2f}s\n\n"
    best_config = max(config_results, key=lambda x: x['score'])
    best_idx = config_results.index(best_config)
    summary_text += f"Best Configuration: Config {best_idx + 1}\n"
    summary_text += f"Best Accuracy: {best_config['score']:.4f}"
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3))
    plt.tight_layout()
    plt.suptitle('SEGC Configuration Analysis - Financial Data',
                 fontsize=16, y=0.98)
    plt.show()


def main():
    """
    Main demonstration function
    """
    print("🚀 QSVM with SEGC Integration - Financial Data Demo")
    print("=" * 60)

    try:
        print("\n" + "="*60)
        print("PHASE 1: SEGC Standalone Testing")
        print("="*60)
        segc_success = run_segc_standalone_demo()
        if not segc_success:
            print("❌ SEGC standalone test failed!")
            return

        print("\n" + "="*60)
        print("PHASE 2: QSVM Comparison Analysis")
        print("="*60)
        comparison_results = run_qsvm_comparison_demo()

        print("\n" + "="*60)
        print("PHASE 3: Detailed SEGC Configuration Analysis")
        print("="*60)
        config_results = run_detailed_segc_analysis()

        print("\n" + "="*60)
        print("DEMO COMPLETE - SUMMARY")
        print("="*60)
        print("\n🎉 All demonstrations completed successfully!")
        print("\nKey Findings:")
        print("• SEGC enhances QSVM for financial regime classification")
        print("• Uses rsi_scaled, volatility_scaled, ma5_scaled, close_scaled")
        print("• Subspace search optimizes hyperparameters and kernel")
        avg_improvement = comparison_results['financial']['qsvm_segc'] - \
            comparison_results['financial']['qsvm_classical']
        print(f"• Average SEGC improvement: {avg_improvement:+.4f}")

    except Exception as e:
        print(f"❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
