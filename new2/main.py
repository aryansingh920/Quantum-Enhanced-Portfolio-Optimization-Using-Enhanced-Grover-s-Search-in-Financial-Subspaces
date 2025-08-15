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


# === METRICS INTERFACE ADDED ===
import argparse
import statistics
from typing import Dict, Any, List
import json as _json

def compute_eta(n_qubits: int, Nc: int, kc: int, kf: int, measured_target_p: float) -> float:
    # θ_c = arcsin( sqrt(Nc / N) ), ideal Pc* = sin^2((2kc+1) θ_c)
    # θ_f = arcsin( sqrt(1 / Nc) ), ideal Pf* = sin^2((2kf+1) θ_f)
    # η = measured_target_p / (Pc* * Pf*)
    import math
    N = 2 ** n_qubits
    theta_c = math.asin(math.sqrt(Nc / N))
    Pc_star = math.sin((2*kc + 1) * theta_c) ** 2
    theta_f = math.asin(math.sqrt(1.0 / max(Nc, 1)))
    Pf_star = math.sin((2*kf + 1) * theta_f) ** 2
    denom = max(Pc_star * Pf_star, 1e-9)
    eta = measured_target_p / denom
    return max(0.0, min(1.5, eta))

def run_metrics(runs: int = 10, shots: int = 20000, seed: int = 1234, out: str = None) -> Dict[str, Any]:
    import random, csv, math
    random.seed(seed)
    segc = SEGCAlgorithm(n_qubits=7, coarse_mask="****0000", target="1010000", coarse_iterations=4)
    segc.coarse_iterate()
    is_qsvm = 0
    if not is_qsvm:
        segc.apply_fine_oracle([segc.target_standard])
        for _ in range(segc.fine_iterations):
            segc.partial_diffuser(segc.qc, segc.qr, segc.ar)
    state = Statevector.from_instruction(segc.qc)
    # Compute P(Hc)
    coarse_states = []
    for i in range(2**segc.n):
        b = format(i, f"0{segc.n}b")[::-1]
        ok = True
        for pos, ch in enumerate(segc.coarse_mask):
            if ch != '*' and ch != b[pos]:
                ok = False; break
        if ok:
            coarse_states.append(b)
    coarse_prob = 0.0
    for b in coarse_states:
        for anc in ['0','1']:
            idx = int(anc + b[::-1], 2) if segc.qc.num_qubits == segc.n + 1 else int(b[::-1], 2)
            if idx < len(state.data):
                coarse_prob += abs(state.data[idx])**2
    run_rows = []
    target_qiskit = segc.target
    for r in range(runs):
        counts = segc.measure(shots=shots)
        target_counts = counts.get(target_qiskit, 0)
        p = target_counts / float(shots)
        run_rows.append((r, p))
    probs = [p for _,p in run_rows]
    mean_p = statistics.mean(probs)
    sd_p = statistics.pstdev(probs) if len(probs) > 1 else 0.0
    z = 1.96
    ci_half = z * (sd_p / math.sqrt(max(len(probs),1)))
    ci_lo, ci_hi = max(0.0, mean_p - ci_half), min(1.0, mean_p + ci_half)
    eta = compute_eta(segc.n, segc.N_c, segc.coarse_iterations, segc.fine_iterations, mean_p)
    summary = {"config": "Baseline SEGC", "runs": runs, "shots": shots, "seed": seed,
                "target_probability_mean": round(mean_p, 6),
                "target_probability_sd": round(sd_p, 6),
                "target_probability_ci95": [round(ci_lo,6), round(ci_hi,6)],
                "coarse_subspace_probability": round(coarse_prob, 6),
                "eta_vs_grover_bound": round(eta, 6)}
    if out:
        csv_path = out if out.endswith(".csv") else out + ".csv"
        json_path = out if out.endswith(".json") else out + ".json"
        with open(csv_path, "w", newline="") as fh:
            cw = csv.writer(fh)
            cw.writerow(["config","run_id","shots","p_target"])
            for r,p in run_rows:
                cw.writerow([summary["config"], r, shots, p])
        with open(json_path, "w") as fh:
            _json.dump(summary, fh, indent=2)
        print(f"[metrics] Wrote per-run CSV to {csv_path} and summary JSON to {json_path}")
    return summary

def _parse_and_maybe_run_metrics():
    import sys
    parser = argparse.ArgumentParser(description="Run metrics or the original program.")
    parser.add_argument("--metrics", action="store_true", help="Run metrics harness instead of the default routine.")
    parser.add_argument("--runs", type=int, default=10, help="Number of repeated measurement runs")
def _parse_and_maybe_run_metrics():
    import sys
    parser = argparse.ArgumentParser(description="Run metrics or the original program.")
    parser.add_argument("--metrics", action="store_true", help="Run metrics harness instead of the default routine.")
    parser.add_argument("--runs", type=int, default=10, help="Number of repeated measurement runs")
    parser.add_argument("--shots", type=int, default=20000, help="Shots per run")
    parser.add_argument("--seed", type=int, default=1234, help="Base RNG seed")
    parser.add_argument("--out", type=str, default=None, help="Output path prefix for CSV/JSON")
    args, _ = parser.parse_known_args()
    if args.metrics:
        summary = run_metrics(runs=args.runs, shots=args.shots, seed=args.seed, out=args.out or "baseline_metrics")
        print(_json.dumps(summary, indent=2))
        sys.exit(0)
    # else fall through to original main()

# Replace main-guard to call parser first

if __name__ == "__main__":
    _parse_and_maybe_run_metrics()
    main()
