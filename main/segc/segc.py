"""
Created on 17/07/2025

@author: Aryan

Filename: segc.py

Relative Path: main/segc/segc.py
"""

from math import floor, sqrt, pi
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# Import modular components
from segc.oracle import coarse_oracle, fine_oracle_subspace, diffuser, diffuser_subspace, weighted_coarse_oracle


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
        return qc

    def search_with_feedback(self, target_num):
        """Main SEGC search with classical feedback."""
        target_bits = format(target_num, f"0{self.n_qubits}b")
        print(f"SEGC Search for target: {target_num} = {target_bits}")
        print(
            f"Coarse on bottom {self.k_coarse} bits, fine on top {self.n_qubits - self.k_coarse} bits")
        print("-" * 60)

        best_result = None
        best_success_rate = 0.0
        current_target_num = target_num
        current_target_bits = target_bits

        for iteration in range(self.max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")
            # Get top subspaces for weighted oracle
            top_subspaces = sorted(
                self.subspace_scores.items(), key=lambda x: x[1], reverse=True)[:3]
            top_subspaces, weights = zip(
                *top_subspaces) if top_subspaces else ([], [])
            weights = [w / sum(weights) if sum(weights) > 0 else 1/len(weights)
                       for w in weights] if weights else None
            qc = self.build_adaptive_circuit(
                current_target_bits, iteration, top_subspaces, weights)
            result = self.simulator.run(
                transpile(qc, self.simulator), shots=self.shots).result()
            counts = result.get_counts()
            target_count = counts.get(current_target_bits, 0)
            success_rate = target_count / self.shots
            self.last_success_rates.append(success_rate)
            print(
                f"Target found: {target_count}/{self.shots} ({success_rate:.3f})")

            subspace_scores, subspace_counts = self.analyze_subspace_distribution(
                counts, current_target_bits)
            best_subspace = max(subspace_scores.keys(),
                                key=lambda x: subspace_scores[x])
            best_subspace_score = subspace_scores[best_subspace]
            print(
                f"Best subspace: {best_subspace} (score: {best_subspace_score:.2f})")

            for subspace in self.subspace_scores:
                self.subspace_scores[subspace] *= self.decay_rate
            for subspace, score in subspace_scores.items():
                self.subspace_scores[subspace] += score

            # Dynamic k_coarse adjustment
            if len(self.search_history) > 2 and max([h['best_subspace_score'] for h in self.search_history[-3:]]) < 1.0:
                self.k_coarse = min(self.k_coarse + 1, self.n_qubits - 1)
                print(
                    f"Adjusted k_coarse to {self.k_coarse} due to low subspace scores")

            # Dynamic target_value update
            if best_subspace_score > 2.0:
                # Recompute target_value based on best subspace
                new_feature_indices = []
                for i in range(0, len(best_subspace) * 2, 3):  # Assuming 3 bits per feature
                    chunk = best_subspace[i:i+3]
                    if len(chunk) == 3:
                        new_feature_indices.append(int(chunk, 2))
                # Limit to top 2 features
                new_feature_indices = new_feature_indices[:2]
                new_bitstring = ''
                for feat_idx in new_feature_indices:
                    if len(new_bitstring) + 3 <= self.n_qubits:
                        new_bitstring += format(feat_idx, '03b')
                    else:
                        remaining = self.n_qubits - len(new_bitstring)
                        if remaining > 0:
                            new_bitstring += format(feat_idx, f'0{remaining}b')
                        break
                new_bitstring = (new_bitstring + '0' *
                                 self.n_qubits)[:self.n_qubits]
                current_target_bits = new_bitstring
                current_target_num = int(current_target_bits, 2)
                print(
                    f"Updated target to: {current_target_num} = {current_target_bits}")

            iteration_data = {
                'iteration': iteration + 1,
                'success_rate': success_rate,
                'target_count': target_count,
                'best_subspace': best_subspace,
                'best_subspace_score': best_subspace_score,
                'counts': counts.copy()
            }
            self.search_history.append(iteration_data)

            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_result = iteration_data
                # Store feature indices derived from the current target bitstring
                bitstring = current_target_bits
                feature_indices = []
                for i in range(0, len(bitstring), 3):
                    chunk = bitstring[i:i+3]
                    if len(chunk) == 3:
                        feature_indices.append(int(chunk, 2))
                self.selected_features = feature_indices[:2]

            if success_rate >= 0.89:
                print("High success rate achieved! Stopping early.")
                break
            if len(self.last_success_rates) >= 3 and self.last_success_rates[-3] > self.last_success_rates[-2] > self.last_success_rates[-1]:
                print("Warning: Over-amplification suspected. Reducing fine iterations.")
                self.max_iterations = iteration + 1
                break
            if iteration > 2 and max([h['success_rate'] for h in self.search_history[-3:]]) - min([h['success_rate'] for h in self.search_history[-3:]]) < 0.01:
                print("Convergence detected. Stopping.")
                break

        return best_result, self.search_history

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
