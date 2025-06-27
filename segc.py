from math import floor, sqrt, pi, ceil
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# ---------------------------- Core helpers -------------------------------


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
    n = len(target_bits)
    qc = QuantumCircuit(n)
    bottom_k = target_bits[-k:]

    # prepare controls
    for i, bit in enumerate(reversed(bottom_k)):
        if bit == '0':
            qc.x(i)

    # multi-controlled Z on bottom-k qubits
    if k == 1:
        qc.z(0)
    else:
        qc.h(k - 1)
        qc.mcx(list(range(k - 1)), k - 1)
        qc.h(k - 1)

    # revert controls
    for i, bit in enumerate(reversed(bottom_k)):
        if bit == '0':
            qc.x(i)
    return qc


def fine_oracle_subspace(target_bits: str, k: int) -> QuantumCircuit:
    """Phase-flips the exact target state within the high (n-k) qubits only."""
    n = len(target_bits)
    qc = QuantumCircuit(n)
    high_bits = target_bits[:-k]  # top n-k bits
    m = len(high_bits)

    # flip where high bit should be 0
    for idx, bit in enumerate(reversed(high_bits)):
        if bit == '0':
            qc.x(k + idx)

    # multi-controlled Z on high bits
    if m == 1:
        qc.z(k)
    else:
        qc.h(k + m - 1)
        qc.mcx(list(range(k, k + m - 1)), k + m - 1)
        qc.h(k + m - 1)

    # revert Xs
    for idx, bit in enumerate(reversed(high_bits)):
        if bit == '0':
            qc.x(k + idx)
    return qc


def diffuser_subspace(n: int, k: int) -> QuantumCircuit:
    """Inversion about mean on the high (n-k) qubits only."""
    qc = QuantumCircuit(n)
    high = list(range(k, n))
    qc.h(high)
    qc.x(high)
    qc.h(high[-1])
    qc.mcx(high[:-1], high[-1])
    qc.h(high[-1])
    qc.x(high)
    qc.h(high)
    return qc

# ---------------------------- SEGC Implementation --------------------------------


class SEGCSearcher:
    """Subspace-Enhanced Grover with Classical feedback (SEGC)"""

    def __init__(self, n_qubits=7, k_coarse=3, shots=1024, max_iterations=5):
        self.n_qubits = n_qubits
        self.k_coarse = k_coarse
        self.shots = shots
        self.max_iterations = max_iterations
        self.simulator = AerSimulator()

        # Classical feedback tracking
        self.search_history = []
        self.subspace_scores = defaultdict(int)
        self.iteration_results = []

    def analyze_subspace_distribution(self, counts, target_bits):
        """Analyze which subspaces are most promising based on measurement results."""
        subspace_counts = defaultdict(int)
        target_subspace = target_bits[-self.k_coarse:]

        for bitstring, count in counts.items():
            # bottom k bits define subspace
            subspace = bitstring[-self.k_coarse:]
            subspace_counts[subspace] += count

        # Score subspaces based on concentration of measurements
        total_shots = sum(counts.values())
        subspace_scores = {}

        for subspace, count in subspace_counts.items():
            probability = count / total_shots
            expected_uniform = 1 / (2**self.k_coarse)

            # Higher score for subspaces with higher than expected concentration
            score = probability / expected_uniform
            subspace_scores[subspace] = score

            # Bonus for target subspace
            if subspace == target_subspace:
                score *= 1.5

        return subspace_scores, subspace_counts

    def adaptive_iteration_count(self, phase, subspace_score=1.0):
        """Dynamically adjust iteration counts based on classical feedback."""
        if phase == "coarse":
            N = 2**self.n_qubits
            M = 2**(self.n_qubits - self.k_coarse)
            base_iterations = max(1, floor(pi/4 * sqrt(N / M)))

            # Reduce iterations if we're getting good subspace concentration
            if len(self.search_history) > 0:
                avg_score = np.mean([h['best_subspace_score']
                                    for h in self.search_history])
                if avg_score > 2.0:  # Good concentration
                    base_iterations = max(1, base_iterations - 1)

        else:  # fine phase
            N_sub = 2**(self.n_qubits - self.k_coarse)
            M_sub = 1
            base_iterations = max(1, floor(pi/4 * sqrt(N_sub / M_sub)))

            # Increase iterations if subspace shows promise
            if subspace_score > 1.5:
                base_iterations += 1
            elif subspace_score > 3.0:
                base_iterations += 2

        return base_iterations

    def build_adaptive_circuit(self, target_bits, iteration_num=0):
        """Build SEGC circuit with adaptive parameters based on classical feedback."""
        qc = QuantumCircuit(self.n_qubits, self.n_qubits)
        qc.h(range(self.n_qubits))  # uniform superposition

        # Adaptive coarse iterations
        r1 = self.adaptive_iteration_count("coarse")
        print(f"Iteration {iteration_num + 1}: Coarse iterations: {r1}")

        for _ in range(r1):
            qc.append(coarse_oracle(target_bits, self.k_coarse),
                      range(self.n_qubits))
            qc.append(diffuser(self.n_qubits), range(self.n_qubits))

        qc.barrier()

        # Get subspace score for adaptive fine search
        target_subspace = target_bits[-self.k_coarse:]
        subspace_score = self.subspace_scores.get(target_subspace, 1.0)

        # Adaptive fine iterations
        r2 = self.adaptive_iteration_count("fine", subspace_score)
        print(
            f"Iteration {iteration_num + 1}: Fine iterations: {r2} (subspace score: {subspace_score:.2f})")

        for _ in range(r2):
            qc.append(fine_oracle_subspace(
                target_bits, self.k_coarse), range(self.n_qubits))
            qc.append(diffuser_subspace(self.n_qubits,
                      self.k_coarse), range(self.n_qubits))

        qc.measure(range(self.n_qubits), range(self.n_qubits))
        qc.draw('mpl', scale=0.5, idle_wires=False, fold=-1)
        return qc

    def search_with_feedback(self, target_num):
        """Main SEGC search with classical feedback loop."""
        target_bits = format(target_num, f"0{self.n_qubits}b")
        print(f"SEGC Search for target: {target_num} = {target_bits}")
        print(
            f"Coarse on bottom {self.k_coarse} bits, fine on top {self.n_qubits - self.k_coarse} bits")
        print(f"Max iterations: {self.max_iterations}")
        print("-" * 60)

        best_result = None
        best_success_rate = 0.0

        for iteration in range(self.max_iterations):
            print(f"\n--- Iteration {iteration + 1} ---")

            # Build adaptive circuit
            qc = self.build_adaptive_circuit(target_bits, iteration)

            # Execute circuit
            result = self.simulator.run(
                transpile(qc, self.simulator), shots=self.shots).result()
            counts = result.get_counts()

            # Analyze results
            target_count = counts.get(target_bits, 0)
            success_rate = target_count / self.shots

            print(
                f"Target found: {target_count}/{self.shots} ({success_rate:.3f})")

            # Classical feedback analysis
            subspace_scores, subspace_counts = self.analyze_subspace_distribution(
                counts, target_bits)
            best_subspace = max(subspace_scores.keys(),
                                key=lambda x: subspace_scores[x])
            best_subspace_score = subspace_scores[best_subspace]

            print(
                f"Best subspace: {best_subspace} (score: {best_subspace_score:.2f})")

            # Update classical feedback
            self.subspace_scores.update(subspace_scores)
            iteration_data = {
                'iteration': iteration + 1,
                'success_rate': success_rate,
                'target_count': target_count,
                'best_subspace': best_subspace,
                'best_subspace_score': best_subspace_score,
                'counts': counts.copy()
            }
            self.search_history.append(iteration_data)

            # Track best result
            if success_rate > best_success_rate:
                best_success_rate = success_rate
                best_result = iteration_data

            # Early stopping if we achieve high success rate
            if success_rate > 0.9:
                print(f"High success rate achieved! Stopping early.")
                break

            # Convergence check
            if iteration > 2:
                recent_rates = [h['success_rate']
                                for h in self.search_history[-3:]]
                if max(recent_rates) - min(recent_rates) < 0.01:
                    print("Convergence detected. Stopping.")
                    break

        return best_result, self.search_history

    def plot_results(self, target_num, best_result, search_history):
        """Visualize SEGC search results and convergence."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

        # 1. Success rate over iterations
        iterations = [h['iteration'] for h in search_history]
        success_rates = [h['success_rate'] for h in search_history]

        ax1.plot(iterations, success_rates, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Success Rate')
        ax1.set_title('SEGC Convergence')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1)

        # 2. Best result histogram
        target_bits = format(target_num, f"0{self.n_qubits}b")
        best_counts = best_result['counts']

        # Show top results
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

        # 3. Subspace scores evolution
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

        # 4. Classical feedback summary
        ax4.axis('off')
        summary_text = f"""SEGC Search Summary
        
Target: {target_num} ({target_bits})
Total Iterations: {len(search_history)}
Best Success Rate: {best_result['success_rate']:.3f}
Best Iteration: {best_result['iteration']}

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

# ---------------------------- Demo -----------------------------------------


def run_segc_demo():
    """Run SEGC demonstration with multiple targets."""
    # Test parameters
    targets = [58, 42, 100]
    n_qubits = 7
    k_coarse = 3
    shots = 2048
    max_iterations = 4

    for target in targets:
        print(f"\n{'='*80}")
        print(f"SEGC DEMO: Target {target}")
        print(f"{'='*80}")

        searcher = SEGCSearcher(n_qubits=n_qubits, k_coarse=k_coarse,
                                shots=shots, max_iterations=max_iterations)

        best_result, search_history = searcher.search_with_feedback(target)

        print(f"\n--- Final Results for Target {target} ---")
        print(f"Best success rate: {best_result['success_rate']:.3f}")
        print(f"Achieved in iteration: {best_result['iteration']}")

        # Show top results from best iteration
        print("\nTop measurement results:")
        target_bits = format(target, f"0{n_qubits}b")
        sorted_counts = sorted(
            best_result['counts'].items(), key=lambda x: -x[1])[:8]

        for bitstring, count in sorted_counts:
            decimal_value = int(bitstring, 2)
            marker = "★" if decimal_value == target else ""
            print(f"{decimal_value:3d} ({bitstring}): {count:4d} {marker}")

        # Plot results
        searcher.plot_results(target, best_result, search_history)


if __name__ == "__main__":
    run_segc_demo()
