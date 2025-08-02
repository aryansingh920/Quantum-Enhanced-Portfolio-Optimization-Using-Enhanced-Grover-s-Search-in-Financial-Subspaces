"""
Created on 17/07/2025

@author: Aryan

Filename: segc.py

Relative Path: main/segc/segc.py
"""
import matplotlib.pyplot as plt
from collections import Counter
from qiskit import transpile
from math import floor, sqrt, pi
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict, deque
# Import modular components
from segc.oracle import diffuser_grover, weighted_oracle, coarse_oracle, fine_oracle_subspace, diffuser, diffuser_subspace, weighted_coarse_oracle

TOL_IMPROVEMENT = 0.01
PATIENCE = 3
SCORE_WINDOW = 5


def run_and_measure(qc, shots=512):
    """
    Runs the quantum circuit and returns measured counts as a dict.
    """
    sim = AerSimulator()
    qc_trans = transpile(qc, sim)
    result = sim.run(qc_trans, shots=shots).result()
    return result.get_counts(qc_trans)

class SEGCSearcher:
    """Subspace-Enhanced Grover with Classical feedback (SEGC)"""

    def __init__(self, n_qubits=6, k_coarse=3, shots=1024, max_iterations=4, decay_rate=0.8, noise_level=0.05):
        self.n_qubits = n_qubits
        self.k_coarse = k_coarse
        self.shots = shots
        self.max_iterations = max_iterations
        self.decay_rate = decay_rate
        self.noise_level = noise_level
        self.simulator = AerSimulator()
        self.search_history = []
        self.subspace_scores = defaultdict(int)
        self.last_success_rates = []
        self.subspace_evaluations = 0
        self.selected_features = None
        self.n_features = 4  # Number of features (adjust based on input data)

    def analyze_subspace_distribution(self, counts, target_bits):
        """Analyze subspace distribution based on measurement results with noise."""
        subspace_counts = defaultdict(int)
        target_subspace = target_bits[-self.k_coarse:]
        for bitstring, count in counts.items():
            subspace = bitstring[-self.k_coarse:]
            subspace_counts[subspace] += count
            self.subspace_evaluations += 1

        total_shots = sum(counts.values())
        subspace_scores = {}
        for subspace, count in subspace_counts.items():
            probability = count / total_shots
            expected_uniform = 1 / (2**self.k_coarse)
            score = probability / expected_uniform
            noise = np.random.normal(0, self.noise_level, 1)[0]
            score = max(0.0, score + noise)
            subspace_scores[subspace] = score
            if subspace == target_subspace:
                score *= 1.5
        return subspace_scores, subspace_counts

    def adaptive_iteration_count(self, phase, subspace_score=1.0):
        """Dynamically adjust iteration counts."""
        if phase == "coarse":
            N = 2**self.n_qubits
            M = 2**(self.n_qubits - self.k_coarse)
            base_iterations = max(1, floor(pi/4 * sqrt(N / M)))
            if len(self.search_history) > 0:
                avg_score = np.mean([h['best_subspace_score']
                                    for h in self.search_history])
                if avg_score > 2.0:
                    base_iterations = max(1, base_iterations - 1)
        else:
            N_sub = 2**(self.n_qubits - self.k_coarse)
            M_sub = 1
            base_iterations = max(1, floor(pi/4 * sqrt(N_sub / M_sub)))
            if subspace_score > 1.5:
                base_iterations += 1
            elif subspace_score > 3.0:
                base_iterations += 2
        return base_iterations

    def build_adaptive_circuit(self, target_bits, iteration_num=0, top_subspaces=None, weights=None):
        """Build SEGC circuit with adaptive parameters."""
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        qc.h(range(self.n_qubits))
        r1 = self.adaptive_iteration_count("coarse")
        print(f"Iteration {iteration_num + 1}: Coarse iterations: {r1}")
        for _ in range(r1):
            if top_subspaces and weights:
                qc.append(weighted_coarse_oracle(
                    target_bits, self.k_coarse, top_subspaces, weights), range(self.n_qubits))
            else:
                qc.append(coarse_oracle(target_bits, self.k_coarse),
                          range(self.n_qubits))
            qc.append(diffuser(self.n_qubits), range(self.n_qubits))
        qc.barrier()
        target_subspace = target_bits[-self.k_coarse:]
        subspace_score = self.subspace_scores.get(target_subspace, 1.0)
        r2 = self.adaptive_iteration_count("fine", subspace_score)
        print(
            f"Iteration {iteration_num + 1}: Fine iterations: {r2} (subspace score: {subspace_score:.2f})")
        for _ in range(r2):
            qc.append(fine_oracle_subspace(
                target_bits, self.k_coarse), range(self.n_qubits))
            qc.append(diffuser_subspace(self.n_qubits,
                      self.k_coarse), range(self.n_qubits))
        qc.measure(range(self.n_qubits), range(self.n_qubits))
        print("Final Circuit:")
        print(qc.draw('mpl'))  # Display the circuit
        return qc

    def plot_subspace_scores(subspace_scores, marked_threshold=0.9):
        bitstrings = list(subspace_scores.keys())
        scores = list(subspace_scores.values())

        # Normalize scores to [0, 1] for threshold marking
        max_score = max(scores)
        normalized_scores = [s / max_score for s in scores]

        colors = ['green' if score >=
                  marked_threshold else 'gray' for score in normalized_scores]

        plt.figure(figsize=(12, 6))
        bars = plt.bar(bitstrings, scores, color=colors)

        plt.xlabel("Bitstring (Feature Subspace)")
        plt.ylabel("QSVM Validation Score")
        plt.title("Subspace Scores After SEGC Search")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show()


    def search_with_feedback(self, n_qubits, max_iters, evaluate_subspace_score, initial_target_bits, initial_k_coarse):
        target_bits = initial_target_bits
        k_coarse = initial_k_coarse
        best_score = 0.0
        best_bits = target_bits
        subspace_scores = defaultdict(float)
        recent_scores = deque(maxlen=SCORE_WINDOW)
        stall_counter = 0

        # initial oracle & grover circuit
        oracle_circ = weighted_oracle([target_bits], n_qubits)
        grover_circ = diffuser_grover(n_qubits, oracle_circ, k_coarse)

        for iter in range(max_iters):
            # simulate Grover circuit & measure
            bitstrings = run_and_measure(grover_circ, shots=512)
            iteration_scores = []
            for candidate_bits, count in bitstrings.items():
                score = evaluate_subspace_score(candidate_bits)
                subspace_scores[candidate_bits] = score
                iteration_scores.append(score)
                recent_scores.append(score)

                # Update best score & target_bits
                if score > best_score + TOL_IMPROVEMENT:
                    best_score = score
                    best_bits = candidate_bits
                    target_bits = best_bits
                    stall_counter = 0
                    # Update oracle and Grover circuit
                    oracle_circ = weighted_oracle([best_bits], n_qubits)
                    grover_circ = diffuser_grover(
                        n_qubits, oracle_circ, k_coarse)
                else:
                    stall_counter += 1

            # Update search_history
            self.search_history.append({
                'iteration': iter + 1,
                'success_rate': max(iteration_scores) if iteration_scores else 0.0,
                'best_subspace_score': best_score,
                'counts': bitstrings
            })

            # ⬇️ Dynamic k_coarse tuning
            if stall_counter >= PATIENCE:
                k_coarse = min(k_coarse + 1, n_qubits)
                oracle_circ = weighted_oracle([best_bits], n_qubits)
                grover_circ = diffuser_grover(n_qubits, oracle_circ, k_coarse)
                stall_counter = 0

            if len(recent_scores) == SCORE_WINDOW and np.var(recent_scores) > 0.02:
                k_coarse = max(1, k_coarse - 1)
                oracle_circ = weighted_oracle([best_bits], n_qubits)
                grover_circ = diffuser_grover(n_qubits, oracle_circ, k_coarse)

        return best_bits, best_score, subspace_scores

    def plot_results(self, target_num, best_result, search_history):
        """Visualize SEGC search results."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        iterations = [h['iteration'] for h in search_history]
        success_rates = [h['success_rate'] for h in search_history]
        ax1.plot(iterations, success_rates, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Success Rate')
        ax1.set_title('SEGC Convergence')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        target_bits = format(target_num, f"0{self.n_qubits}b")
        best_counts = best_result['counts']
        sorted_counts = sorted(best_counts.items(), key=lambda x: -x[1])[:15]
        states = [int(state, 2) for state, _ in sorted_counts]
        counts = [count for _, count in sorted_counts]
        colors = ['red' if int(
            state, 2) == target_num else 'blue' for state, _ in sorted_counts]
        ax2.bar(range(len(states)), counts, color=colors, alpha=0.7)
        ax2.set_xlabel('State (decimal)')
        ax2.set_ylabel('Count')
        ax2.set_title(f'Best Result (Iteration {best_result["iteration"]})')
        ax2.set_xticks(range(len(states)))
        ax2.set_xticklabels(states, rotation=45)

        subspace_history = defaultdict(list)
        for h in search_history:
            subspace_scores, _ = self.analyze_subspace_distribution(
                h['counts'], target_bits)
            for subspace, score in subspace_scores.items():
                subspace_history[subspace].append(score)
        target_subspace = target_bits[-self.k_coarse:]
        for subspace, scores in subspace_history.items():
            if len(scores) == len(iterations):
                color = 'red' if subspace == target_subspace else 'gray'
                alpha = 1.0 if subspace == target_subspace else 0.3
                ax3.plot(iterations, scores, 'o-', color=color, alpha=alpha,
                         label=f'Subspace {subspace}' if subspace == target_subspace else None)
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel('Subspace Score')
        ax3.set_title('Subspace Learning')
        ax3.grid(True, alpha=0.3)
        if target_subspace in subspace_history:
            ax3.legend()

        ax4.axis('off')
        total_subspaces = 2 ** self.n_features
        summary_text = f"""SEGC Search Summary
Target: {target_num} ({target_bits})
Total Iterations: {len(search_history)}
Best Success Rate: {best_result['success_rate']:.3f}
Best Iteration: {best_result['iteration']}
Subspace Evaluations: {self.subspace_evaluations}
Fraction of Subspaces Evaluated: {self.subspace_evaluations / total_subspaces:.3f}
Noise Level: {self.noise_level:.3f}
Selected Features: {self.selected_features}
Final Subspace Scores:
"""
        for subspace, score in sorted(self.subspace_scores.items(), key=lambda x: -x[1])[:5]:
            marker = "★" if subspace == target_subspace else ""
            summary_text += f"{subspace}: {score:.2f} {marker}\n"
        ax4.text(0.1, 0.5, summary_text, fontsize=12, fontfamily='monospace',
                 verticalalignment='center')
        plt.tight_layout()
        plt.suptitle(
            f'SEGC Analysis for Target {target_num}', fontsize=16, y=0.98)
        plt.show()
