import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit_aer import StatevectorSimulator
from qsvm import QSVM
from segc import SEGCAlgorithm
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("segc_qsvm.log"),   # Log to file
        logging.StreamHandler()                 # Log to console
    ]
)


def qiskit_to_standard(state):
    """Convert Qiskit qubit ordering to standard ordering."""
    return state[::-1]


def apply_multi_controlled_z(qc, qr, target_state):
    """
    Apply a multi-controlled Z gate to mark the target state.
    
    Parameters:
    - qc: QuantumCircuit to apply the gate.
    - qr: QuantumRegister for data qubits.
    - target_state: Binary string (Qiskit ordering) for the target state.
    """
    n_qubits = len(target_state)
    for i, bit in enumerate(target_state):
        if bit == '0':
            qc.x(qr[i])  # Flip qubit if target state has 0
    qc.h(qr[n_qubits-1])  # Apply H to last qubit for Z gate
    # Multi-controlled X (equivalent to Z with H)
    qc.mcx(qr[:-1], qr[n_qubits-1], mode='noancilla')
    qc.h(qr[n_qubits-1])  # Undo H
    for i, bit in enumerate(target_state):
        if bit == '0':
            qc.x(qr[i])  # Undo X gates


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
    
    Raises:
    - ValueError: If dataset is too small for subspace mapping.
    """
    if len(X) < subspace_size:
        raise ValueError(
            f"Dataset size {len(X)} < subspace size {subspace_size}")
    state_indices = list(range(subspace_size))
    X_mapped = X[:subspace_size]
    logging.info("Feature mapping:")
    for idx, features in zip(state_indices, X_mapped):
        logging.info(f"  State {format(idx, f'0{n_qubits}b')}: {features}")
    return state_indices, X_mapped


def main():
    # Initialize SEGC for 7 qubits (N = 128)
    try:
        segc = SEGCAlgorithm()
    except TypeError:
        logging.warning(
            "SEGCAlgorithm initialization with parameters failed, using default constructor")
        segc = SEGCAlgorithm()
        segc.n = 7
        segc.N_c = 8

    target_state_standard = "1010000"  # As per PDF
    target_state_qiskit = qiskit_to_standard(target_state_standard)
    logging.info(
        f"Target state: {target_state_standard} (Standard), {target_state_qiskit} (Qiskit)")

    # Initialize QSVM with 7 qubits
    csv_file = "../ticker/JPM/JPM_data.csv"
    try:
        qsvm = QSVM(csv_file=csv_file, n_qubits=7, feature_dim=4,
                    iterations=5, initial_theta=0.1, max_train_samples=200)
    except FileNotFoundError:
        logging.error(f"CSV file {csv_file} not found")
        return
    except Exception as e:
        logging.error(f"Error loading CSV file: {e}")
        return

    # Train QSVM
    logging.info("Training QSVM...")
    qsvm.train()

    # Create quantum circuit
    qr = QuantumRegister(segc.n, 'q')
    ar = QuantumRegister(1, 'anc')
    cr = ClassicalRegister(segc.n, 'c')
    qc = QuantumCircuit(qr, ar, cr)
    qc.h(qr)
    logging.info("Initialized superposition")

    # Coarse iterations (cumulative)
    logging.info("Executing coarse iterations...")
    for i in range(segc.coarse_iterations):
        logging.info(f"Coarse iteration {i+1}/{segc.coarse_iterations}")
        segc.coarse_oracle(qc, qr, ar)
        segc.coarse_diffuser(qc, qr)
        state = Statevector.from_instruction(qc)
        prob = segc.analyze_state(state, f"After coarse iteration {i+1}")
        if prob is None:
            logging.warning(
                f"analyze_state returned None for coarse iteration {i+1}")
            prob = state.probabilities_dict().get(target_state_qiskit, 0.0)
        logging.info(f"Coarse iteration {i+1} target probability: {prob:.4f}")

    # Map features
    try:
        state_indices, X_test = map_features_to_states(
            qsvm.X_scaled, segc.n, segc.N_c)
        logging.info(
            f"Mapped {len(state_indices)} states to coarse subspace with features")
    except ValueError as e:
        logging.error(f"Feature mapping error: {e}")
        return

    # Feedback loop (cumulative)
    logging.info("Executing QSVM feedback loop...")
    prev_prob = 0.0
    for i in range(qsvm.iterations):
        logging.info(f"Feedback iteration {i+1}/{qsvm.iterations}")
        qsvm.feedback_loop(segc, qc, qr, ar, state_indices,
                           X_test)  # Apply on existing circuit
        state = Statevector.from_instruction(qc)
        current_prob = segc.analyze_state(
            state, f"After feedback iteration {i+1}")
        if current_prob is None:
            logging.warning(
                f"analyze_state returned None for feedback iteration {i+1}")
            current_prob = state.probabilities_dict().get(target_state_qiskit, prev_prob)
        logging.info(
            f"Target probability: {current_prob:.4f}, Threshold: {qsvm.theta:.4f}")

        # Fallback if no improvement
        if abs(current_prob - prev_prob) < 1e-4 and i < qsvm.iterations - 1:
            logging.warning("No improvement, forcing target state marking")
            for _ in range(3):  # Apply 3 times for stronger effect
                apply_multi_controlled_z(qc, qr, target_state_qiskit)
            # Lower threshold more aggressively
            qsvm.theta = max(0.0, qsvm.theta * 0.7)

        # Dynamic threshold adjustment
        target_prob = 0.6
        qsvm.theta = max(0.05, min(qsvm.theta + 0.05 *
                         (target_prob - current_prob), 0.5))
        prev_prob = current_prob

    # Measure and simulate
    qc.measure(qr, cr)
    segc.run_simulation(qc, state)

    logging.info("Note: Classical SVM integration not implemented. "
                 "Consider quantum approximation techniques as per PDF Section 8.4.")


if __name__ == "__main__":
    main()
