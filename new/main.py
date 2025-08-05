import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit_aer import StatevectorSimulator

from qsvm import QSVM
from segc import SEGCAlgorithm


def map_features_to_states(X, n_qubits, subspace_size):
    """
    Map scaled features to quantum state indices in the coarse subspace.
    
    Parameters:
    - X: Scaled feature data.
    - n_qubits: Total number of qubits.
    - subspace_size: Size of the coarse subspace (e.g., 8 for 0000****).
    
    Returns:
    - state_indices: List of state indices.
    - X_mapped: Corresponding feature vectors.
    """
    state_indices = list(range(subspace_size))
    X_mapped = X[:subspace_size]
    if len(X_mapped) < subspace_size:
        X_mapped = np.tile(
            X_mapped, (subspace_size // len(X_mapped) + 1, 1))[:subspace_size]
    return state_indices, X_mapped



def main():
    # Initialize SEGC
    segc = SEGCAlgorithm()

    # Initialize QSVM with CSV file
    # Replace with actual CSV file path
    csv_file = "../ticker/JPM/JPM_data.csv"

    qsvm = QSVM(csv_file=csv_file, n_qubits=4, feature_dim=4,
                iterations=3, initial_theta=0.5, max_train_samples=20)

    # Train QSVM
    qsvm.train()

    # Create quantum circuit
    qr = QuantumRegister(segc.n, 'q')
    ar = QuantumRegister(1, 'anc')
    cr = ClassicalRegister(segc.n, 'c')
    qc = QuantumCircuit(qr, ar, cr)

    # Initialize superposition
    qc.h(qr)
    print("Debug: Initialized superposition")

    # Execute coarse iterations
    print("Executing coarse iterations...")
    for i in range(segc.coarse_iterations):
        print(f"  Coarse iteration {i+1}/{segc.coarse_iterations}")
        segc.coarse_oracle(qc, qr, ar)
        segc.coarse_diffuser(qc, qr)
        state = Statevector.from_instruction(qc)
        segc.analyze_state(state, f"After coarse iteration {i+1}")

    # Map financial features to coarse subspace states
    state_indices, X_test = map_features_to_states(
        qsvm.X_scaled, segc.n, segc.N_c)
    print(
        f"Debug: Mapped {len(state_indices)} states to coarse subspace with features")

    # Execute feedback loop with QSVM
    print("Executing QSVM feedback loop...")
    final_state = qsvm.feedback_loop(segc, qc, qr, ar, state_indices, X_test)

    # Measure
    qc.measure(qr, cr)

    # Run simulation
    segc.run_simulation(qc, final_state)


if __name__ == "__main__":
    main()
