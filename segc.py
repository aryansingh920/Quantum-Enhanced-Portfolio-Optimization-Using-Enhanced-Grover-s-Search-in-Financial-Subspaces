from math import floor, sqrt, pi
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

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

# ---------------------------- SEGS builder --------------------------------


def build_segs_improved(num: int, n_qubits: int = 7, k_coarse: int = 3, shots: int = 1024):
    target_bits = format(num, f"0{n_qubits}b")
    print(f"Searching for target: {num} = {target_bits}")
    print(
        f"Coarse on bottom {k_coarse} bits, fine on top {n_qubits - k_coarse} bits")

    qc = QuantumCircuit(n_qubits, n_qubits)
    qc.h(range(n_qubits))  # uniform superposition

    # --- Coarse amplification ---
    N = 2**n_qubits
    M = 2**(n_qubits - k_coarse)
    theta = pi/4 * sqrt(N / M)
    r1 = max(1, floor(theta))
    print(f"Coarse iterations (floor): {r1}")
    for _ in range(r1):
        qc.append(coarse_oracle(target_bits, k_coarse), range(n_qubits))
        qc.append(diffuser(n_qubits), range(n_qubits))
    qc.barrier()

    # --- Fine search within subspace ---
    N_sub = M
    M_sub = 1
    phi = pi/4 * sqrt(N_sub / M_sub)
    r2 = max(1, floor(phi))
    print(f"Fine iterations (floor): {r2}")
    for _ in range(r2):
        qc.append(fine_oracle_subspace(target_bits, k_coarse), range(n_qubits))
        qc.append(diffuser_subspace(n_qubits, k_coarse), range(n_qubits))

    # Measure
    qc.measure(range(n_qubits), range(n_qubits))

    sim = AerSimulator()
    result = sim.run(transpile(qc, sim), shots=shots).result()
    counts = result.get_counts()
    qc.draw('mpl', scale=0.5, idle_wires=False, fold=-1)
    return qc, counts

# ---------------------------- Demo -----------------------------------------


if __name__ == "__main__":
    TARGET = 58
    N_Q = 7
    K_COARSE = 3
    SHOTS = 2048

    circuit, counts = build_segs_improved(TARGET, N_Q, K_COARSE, SHOTS)
    print("Results:")
    for bitstring, count in sorted(counts.items(), key=lambda x: -x[1])[:10]:
        # Convert bitstring to decimal for better readability
        decimal_value = int(bitstring, 2)
        print(f"{decimal_value} ({bitstring}): {count}")

    plot_histogram(counts, figsize=(10, 6), number_to_keep=20)
    plt.title(
        f"Improved SEGS for target {TARGET} ({format(TARGET, f'0{N_Q}b')})")
    plt.show()
