"""
Created on 17/07/2025

@author: Aryan

Filename: grover.py

Relative Path: main/segc/grover.py
"""


from qiskit import QuantumCircuit


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
