#!/usr/bin/env python3
# main2.py — QSVM-enhanced SEGC hybrid driver

import logging
import numpy as np
from segc import SEGCAlgorithm
from qsvm import QSVM


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    # Configuration
    n_qubits = 7
    coarse_mask = "****0000"  # Bottom 4 qubits are coarse-fixed
    target_state = "1010000"   # Standard ordering
    n_coarse_it = 4
    n_fine_it = 2
    n_feedback = 5
    initial_thresh = 0.10
    csv_file = "../ticker/JPM/JPM_data.csv"  # Replace with your actual CSV file

    # Initialize SEGC
    logging.info("Initializing SEGC state")
    segc = SEGCAlgorithm(
        n_qubits=n_qubits,
        coarse_mask=coarse_mask,
        target=target_state,
        coarse_iterations=n_coarse_it
    )
    segc.initialize()

    # Run coarse phase
    logging.info(f"Running {n_coarse_it} coarse iterations")
    segc.run_coarse()
    coarse_states = segc.get_coarse_states()  # Returns in Qiskit ordering
    logging.info(f"Coarse phase yielded {len(coarse_states)} candidate states")

    # Feature extraction
    X = []
    for s in coarse_states:
        # Convert to standard ordering for features
        feat = segc.extract_features(s[::-1])
        X.append(feat)
    X = np.stack(X, axis=0)

    # Label acquisition (placeholder: use real labels from data)
    y = np.array([1 if s[::-1] == target_state else 0 for s in coarse_states])

    # Train QSVM
    logging.info(f"Training QSVM on {len(y)} samples")
    qsvm = QSVM(csv_file=csv_file, n_qubits=7, feature_dim=len(X[0]))
    qsvm.train()

    # Hybrid feedback loop
    threshold = initial_thresh
    for fb in range(1, n_feedback+1):
        X_scaled = qsvm.scaler.transform(X)
        scores = qsvm.compute_scores(list(range(len(coarse_states))), X_scaled)
        marked = [s for s, score in zip(
            coarse_states, scores) if score > threshold]
        logging.info(
            f"Feedback {fb}/{n_feedback}: threshold={threshold:.3f}, marked={len(marked)}")
        if not marked:
            logging.warning(
                "No states marked above threshold – forcing target_state")
            marked = [target_state[::-1]]  # Convert to Qiskit ordering

        segc.update_fine_oracle(marked)
        state = segc.run_fine_iteration()

        # Perform measurement separately
        counts = segc.measure()
        total = sum(counts.values())
        # Convert to standard ordering
        probs = {s[::-1]: counts.get(s, 0)/total for s in coarse_states}
        topk = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:5]
        logging.info(
            f"Top 5 coarse-subspace probs after feedback {fb}: {topk}")

        threshold = float(np.percentile(scores, 80))

    # Final measurement
    logging.info("Running final fine-phase measurement")
    final_counts = segc.measure(shots=20000)
    sorted_final = sorted([(k[::-1], v) for k, v in final_counts.items()],
                          key=lambda kv: kv[1], reverse=True)[:10]
    logging.info(f"Final Top-10 outcomes: {sorted_final}")


if __name__ == "__main__":
    main()
