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
    _parse_and_maybe_run_metrics()
    main()


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
    is_qsvm = 1
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
    summary = {"config": "SEGC+QSVM", "runs": runs, "shots": shots, "seed": seed,
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
    parser.add_argument("--shots", type=int, default=20000, help="Shots per run")
    parser.add_argument("--seed", type=int, default=1234, help="Base RNG seed")
    parser.add_argument("--out", type=str, default=None, help="Output path prefix for CSV/JSON")
    args, _ = parser.parse_known_args()
    if args.metrics:
        summary = run_metrics(runs=args.runs, shots=args.shots, seed=args.seed, out=args.out or "qsvm_metrics")
        print(_json.dumps(summary, indent=2))
        sys.exit(0)
    # else fall through to original main()

# Replace main-guard to call parser first
