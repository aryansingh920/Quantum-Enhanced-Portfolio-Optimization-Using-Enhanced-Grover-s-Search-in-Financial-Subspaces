from qiskit import QuantumCircuit
import numpy as np


def coarse_oracle(target_bits: str, k: int) -> QuantumCircuit:
    """Marks states whose bottom-k bits match the target's bottom-k bits."""
    n = len(target_bits)
    qc = QuantumCircuit(n)
    bottom_k = target_bits[-k:]
    for i, bit in enumerate(reversed(bottom_k)):
        if bit == '0':
            qc.x(i)
    if k == 1:
        qc.z(0)
    else:
        qc.h(k - 1)
        qc.mcx(list(range(k - 1)), k - 1)
        qc.h(k - 1)
    for i, bit in enumerate(reversed(bottom_k)):
        if bit == '0':
            qc.x(i)
    return qc


def fine_oracle_subspace(target_bits: str, k: int) -> QuantumCircuit:
    """Phase-flips the exact target state within the high (n-k) qubits."""
    n = len(target_bits)
    qc = QuantumCircuit(n)
    high_bits = target_bits[:-k]
    m = len(high_bits)
    for idx, bit in enumerate(reversed(high_bits)):
        if bit == '0':
            qc.x(k + idx)
    if m == 1:
        qc.z(k)
    else:
        qc.h(k + m - 1)
        qc.mcx(list(range(k, k + m - 1)), k + m - 1)
        qc.h(k + m - 1)
    for idx, bit in enumerate(reversed(high_bits)):
        if bit == '0':
            qc.x(k + idx)
    return qc


def weighted_coarse_oracle(target_bits: str, k: int, top_subspaces: list, weights: list) -> QuantumCircuit:
    """Marks multiple subspaces with weighted phase shifts based on validation scores."""
    n = len(target_bits)
    qc = QuantumCircuit(n)
    if not top_subspaces or not weights:
        return coarse_oracle(target_bits, k)
    for subspace, weight in zip(top_subspaces, weights):
        for i, bit in enumerate(reversed(subspace)):
            if bit == '0':
                qc.x(i)
        if k == 1:
            qc.rz(weight * np.pi, 0)
        else:
            qc.h(k - 1)
            qc.mcx(list(range(k - 1)), k - 1, weight=weight)
            qc.h(k - 1)
        for i, bit in enumerate(reversed(subspace)):
            if bit == '0':
                qc.x(i)
    return qc


def weighted_oracle(bitstrings, n):
    """Oracle that flips the phase of all bitstrings in the list."""
    qc = QuantumCircuit(n)
    for bits in bitstrings:
        for i, b in enumerate(reversed(bits)):
            if b == '0':
                qc.x(i)
        qc.h(n - 1)
        qc.mcx(list(range(n - 1)), n - 1)
        qc.h(n - 1)
        for i, b in enumerate(reversed(bits)):
            if b == '0':
                qc.x(i)
    return qc


def diffuser(n_qubits):
    """Standard Grover diffuser on n qubits."""
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)
    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


def diffuser_subspace(n: int, k: int) -> QuantumCircuit:
    """Inversion about mean on the high (n-k) qubits."""
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


def diffuser_grover(n_qubits, oracle_circuit, k):
    """
    Constructs a Grover circuit using the given oracle and diffuser.
    Applies k iterations.
    """
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    for _ in range(k):
        qc.append(oracle_circuit.to_gate(), range(n_qubits))
        qc.append(diffuser(n_qubits).to_gate(), range(n_qubits))
    qc.measure_all()
    return qc
