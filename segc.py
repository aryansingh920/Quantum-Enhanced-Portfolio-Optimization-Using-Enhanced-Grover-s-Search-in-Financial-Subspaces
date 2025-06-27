"""
SEGS Grover-on-Grover (Corrected Implementation)
Author: you
Licence: Apache License 2.0
Description: This script implements the Subspace-Enhanced Grover Search (SEGS) algorithm,
which combines coarse and fine Grover searches to efficiently find a target number in a binary space.
It uses Qiskit for quantum circuit construction and simulation.
"""
from math import ceil, sqrt, pi
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# ---------------------------- Core helpers -------------------------------


def diffuser(n):
    """Standard Grover diffuser (inversion about average)"""
    d = QuantumCircuit(n)
    d.h(range(n))
    d.x(range(n))
    d.h(n - 1)
    d.mcx(list(range(n - 1)), n - 1)
    d.h(n - 1)
    d.x(range(n))
    d.h(range(n))
    return d


def fine_oracle(target_bits):
    """Oracle that phase-flips only the exact target state"""
    n = len(target_bits)
    o = QuantumCircuit(n)

    # Flip bits that should be 0 in target
    for i, bit in enumerate(reversed(target_bits)):
        if bit == "0":
            o.x(i)

    # Multi-controlled Z gate
    o.h(n - 1)
    o.mcx(list(range(n - 1)), n - 1)
    o.h(n - 1)

    # Flip back the bits
    for i, bit in enumerate(reversed(target_bits)):
        if bit == "0":
            o.x(i)

    return o


def coarse_oracle(target_bits: str, k: int) -> QuantumCircuit:
    """
    Phase-flips all states whose bottom-k bits match the target's bottom-k bits.
    This creates a subspace of 2^(n-k) states that share the same bottom-k bits.
    """
    n = len(target_bits)
    q = QuantumCircuit(n)

    # Get the bottom-k bits of the target
    bottom_k_bits = target_bits[-k:]  # Last k characters (bottom k bits)

    # Apply X gates for bits that should be 0
    for i, bit in enumerate(reversed(bottom_k_bits)):
        if bit == '0':
            q.x(i)

    # Multi-controlled Z on bottom-k qubits
    if k == 1:
        q.z(0)
    else:
        q.h(k-1)
        q.mcx(list(range(k-1)), k-1)
        q.h(k-1)

    # Flip back the X gates
    for i, bit in enumerate(reversed(bottom_k_bits)):
        if bit == '0':
            q.x(i)

    return q

# ---------------------------- SEGS circuit builder -----------------------


def build_segs(num: int, n_qubits: int = 7, k_coarse: int = 3, shots: int = 1024):
    """
    Build and execute SEGS circuit.
    
    Args:
        num: Target number to search for
        n_qubits: Total number of qubits
        k_coarse: Number of bits to use in coarse search (bottom bits)
        shots: Number of measurement shots
    """
    target_bits = format(num, f"0{n_qubits}b")
    print(f"Searching for target: {num} = {target_bits}")
    print(f"Bottom-{k_coarse} bits: {target_bits[-k_coarse:]}")

    qc = QuantumCircuit(n_qubits, n_qubits)

    # Initialize superposition
    qc.h(range(n_qubits))

    # -------- Coarse round: Amplify subspace --------
    # Number of marked states in coarse search = 2^(n-k)
    N_coarse = 2**n_qubits
    M_coarse = 2**(n_qubits - k_coarse)  # Size of target subspace

    # Optimal iterations for coarse amplification
    r1 = max(1, round(pi/4 * sqrt(N_coarse / M_coarse)))
    print(f"Coarse iterations: {r1}")

    for i in range(r1):
        qc.append(coarse_oracle(target_bits, k_coarse), qc.qubits)
        qc.append(diffuser(n_qubits), qc.qubits)

    qc.barrier()

    # -------- Fine round: Find exact target within subspace --------
    # Now we're searching within the amplified subspace
    N_fine = M_coarse  # Search space size after coarse amplification
    M_fine = 1         # Only one target state

    # Optimal iterations for fine search
    r2 = max(1, round(pi/4 * sqrt(N_fine / M_fine)))
    print(f"Fine iterations: {r2}")

    for i in range(r2):
        qc.append(fine_oracle(target_bits), qc.qubits)
        qc.append(diffuser(n_qubits), qc.qubits)

    # Measure all qubits
    qc.measure(range(n_qubits), range(n_qubits))

    # Execute circuit
    sim = AerSimulator()
    result = sim.run(transpile(qc, sim), shots=shots).result()
    counts = result.get_counts()

    return qc, counts


def analyze_results(counts, target_num, n_qubits, k_coarse):
    """Analyze and display results"""
    target_bits = format(target_num, f"0{n_qubits}b")
    bottom_k = target_bits[-k_coarse:]

    print("\nTop 10 results:")
    sorted_counts = sorted(counts.items(), key=lambda x: -x[1])

    target_found = False
    subspace_total = 0

    for i, (bitstring, count) in enumerate(sorted_counts[:10]):
        decimal = int(bitstring, 2)
        is_target = decimal == target_num
        in_subspace = bitstring[-k_coarse:] == bottom_k

        marker = "🎯" if is_target else ("✓" if in_subspace else "✗")

        print(f"{marker} {decimal:>3} ({bitstring}): {count:>4} shots")

        if is_target:
            target_found = True
        if in_subspace:
            subspace_total += count

    total_shots = sum(counts.values())
    if target_found:
        target_probability = counts.get(
            format(target_num, f"0{n_qubits}b"), 0) / total_shots
        print(f"\n🎯 Target found with probability: {target_probability:.3f}")

    subspace_probability = subspace_total / total_shots
    print(
        f"✓ Total probability in correct subspace: {subspace_probability:.3f}")

    return target_found, subspace_probability

# ---------------------------- Example run ---------------------------------


if __name__ == "__main__":
    TARGET = 58      # 111010 in binary (7 bits)
    N_QUBITS = 7
    K = 3            # Use bottom 3 bits for coarse search

    print("=" * 60)
    print("SEGS (Subspace-Enhanced Grover Search)")
    print("=" * 60)

    # Build and run SEGS
    circuit, counts = build_segs(TARGET, N_QUBITS, K, shots=2048)

    # Analyze results
    target_found, subspace_prob = analyze_results(counts, TARGET, N_QUBITS, K)

    # Plot histogram
    plot_histogram(counts, figsize=(15, 8), number_to_keep=20)
    plt.title(
        f'SEGS Results for Target {TARGET} (binary: {format(TARGET, f"0{N_QUBITS}b")})')
    plt.show()

    # Compare with theoretical expectations
    print(f"\nTheoretical Analysis:")
    print(f"- Search space size: 2^{N_QUBITS} = {2**N_QUBITS}")
    print(f"- Coarse subspace size: 2^{N_QUBITS-K} = {2**(N_QUBITS-K)}")
    print(
        f"- Expected coarse success rate: ~{2**(N_QUBITS-K)/2**N_QUBITS:.3f}")
    print(f"- Actual subspace probability: {subspace_prob:.3f}")
