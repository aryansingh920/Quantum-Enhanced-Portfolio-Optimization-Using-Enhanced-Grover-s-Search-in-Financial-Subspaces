from qiskit import QuantumCircuit


def coarse_oracle_7qubits():
    # 7 qubits total
    qc = QuantumCircuit(7)

    # Step 1: Flip the first 4 qubits so |0000> becomes |1111>
    qc.x([0, 1, 2, 3])

    # Step 2: Apply a multi-controlled Z using last qubit as target
    qc.h(6)                 # Change Z to X basis
    qc.mcx([0, 1, 2, 3], 6)    # Multi-controlled X
    qc.h(6)                 # Back to Z basis

    # Step 3: Undo the flips
    qc.x([0, 1, 2, 3])

    return qc


# Build and visualize
qc = coarse_oracle_7qubits()
print(qc.draw(fold=-1))
