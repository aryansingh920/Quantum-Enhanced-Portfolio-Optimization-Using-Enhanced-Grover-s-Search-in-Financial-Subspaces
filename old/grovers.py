"""
Created on 27/06/2025

@author: Aryan

Filename: grovers.py

Relative Path: grovers.py
"""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.visualization import plot_histogram
import matplotlib.pyplot as plt

# --- Parameters ---
n_qubits = 7
# num = input("Enter Number")
num: int = 58
target_bin = format(int(num), f'0{n_qubits}b')


# Oracle that marks |1001001⟩
def grover_oracle():
    qc = QuantumCircuit(n_qubits)

    # Flip bits where target has 0
    for i, bit in enumerate(reversed(target_bin)):
        if bit == '0':
            qc.x(i)

    # Multi-controlled Z via H + multi-controlled Toffoli
    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)

    # Undo flips
    for i, bit in enumerate(reversed(target_bin)):
        if bit == '0':
            qc.x(i)

    return qc

# Diffuser (inversion about mean)


def diffuser():
    qc = QuantumCircuit(n_qubits)
    qc.h(range(n_qubits))
    qc.x(range(n_qubits))

    qc.h(n_qubits - 1)
    qc.mcx(list(range(n_qubits - 1)), n_qubits - 1)
    qc.h(n_qubits - 1)

    qc.x(range(n_qubits))
    qc.h(range(n_qubits))
    return qc


# Build the Grover circuit
qc = QuantumCircuit(n_qubits, n_qubits)
qc.h(range(n_qubits))


# Iterations: round(pi/4 * sqrt(N)) → sqrt(100) ≈ 10 → ~8 iterations
for _ in range(8):
    qc.append(grover_oracle(), range(n_qubits))
    qc.append(diffuser(), range(n_qubits))

qc.measure(range(n_qubits), range(n_qubits))
# print(qc.draw('text'))
qc.draw('mpl', scale=0.5, idle_wires=False, fold=-1)
# Run simulation
sim = AerSimulator()
compiled = transpile(qc, sim)
result = sim.run(compiled, shots=1024).result()
counts = result.get_counts()

# Output
print("Top results:")
for k, v in sorted(counts.items(), key=lambda x: -x[1])[:5]:
    print(f"{int(k, 2)} ({k}): {v} times")
    # plot

plot_histogram(counts)
plt.show()
