"""
Created on 17/07/2025

@author: Aryan

Filename: oracle.py

Relative Path: main/segc/oracle.py
"""


from qiskit import QuantumCircuit


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
