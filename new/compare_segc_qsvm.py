#!/usr/bin/env python3
import time
import math
import logging
from typing import List, Tuple, Dict

import numpy as np

from segc import SEGCAlgorithm, matches_mask
from qsvm import QSVM

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def eta_vs_grover(n_qubits: int, Nc: int, kc: int, kf: int, measured_p: float) -> float:
    N = 2 ** n_qubits
    theta_c = math.asin(math.sqrt(Nc / N))
    Pc_star = math.sin((2 * kc + 1) * theta_c) ** 2
    theta_f = math.asin(math.sqrt(1.0 / max(Nc, 1)))
    Pf_star = math.sin((2 * kf + 1) * theta_f) ** 2
    denom = max(Pc_star * Pf_star, 1e-12)
    return max(0.0, min(5.0, measured_p / denom))


def summarize(name: str, probs: List[float]) -> Tuple[float, float, Tuple[float, float]]:
    arr = np.asarray(probs, dtype=float)
    mean = float(arr.mean()) if arr.size else 0.0
    var = float(arr.var(ddof=1)) if arr.size > 1 else 0.0
    sd = math.sqrt(var)
    ci = 1.96 * (sd / math.sqrt(max(1, arr.size)))
    logging.info(
        f"{name}: mean={mean:.6f}, sd={sd:.6f}, 95% CI=[{mean - ci:.6f}, {mean + ci:.6f}]")
    return mean, sd, (mean - ci, mean + ci)


def run_baseline(n_qubits=7, mask="****0000", target="1010000", shots=20000) -> Tuple[float, float, int, int]:
    t0 = time.perf_counter()
    segc = SEGCAlgorithm(n_qubits=n_qubits, coarse_mask=mask,
                         target=target, shots=shots)
    segc.coarse_iterate()
    # Fine: mark only the target for baseline
    segc.fine_iterate([target])
    target_p, coarse_p, _ = segc.analyze_state()
    dt_ms = int((time.perf_counter() - t0) * 1000)
    logging.info(
        f"Baseline SEGC: target={target_p:.5f}, coarse={coarse_p:.5f}, time={dt_ms}ms")
    return target_p, coarse_p, segc.coarse_iterations, segc.fine_iterations


def run_qsvm_segc(n_qubits=7, mask="****0000", target="1010000", shots=20000, k=2, feedback_rounds=3) -> Tuple[float, float, int, int]:
    t0 = time.perf_counter()
    segc = SEGCAlgorithm(n_qubits=n_qubits, coarse_mask=mask,
                         target=target, shots=shots)
    segc.coarse_iterate()

    # Build the list of STANDARD-ordering states that lie in Hc
    _, _, probs_std = segc.analyze_state()
    coarse_states = [s for s in probs_std.keys() if matches_mask(s, mask)]
    # QSVM in 'state' mode
    qsvm = QSVM(segc, feature_map_type="statevector", topk=k, mode="state")
    qsvm.train(coarse_states_std=coarse_states)

    # Feedback rounds: update fine oracle with top-k each round, run k_f diffuser steps
    for r in range(feedback_rounds):
        marked, scores = qsvm.select_topk_states(coarse_states)
        logging.info(f"Feedback {r+1}/{feedback_rounds}: marked={marked}")
        segc.fine_iterate(marked)

    target_p, coarse_p, _ = segc.analyze_state()
    dt_ms = int((time.perf_counter() - t0) * 1000)
    logging.info(
        f"QSVM-SEGC: target={target_p:.5f}, coarse={coarse_p:.5f}, time={dt_ms}ms")
    return target_p, coarse_p, segc.coarse_iterations, segc.fine_iterations


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--shots", type=int, default=20000)
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--feedback", type=int, default=3)
    args = parser.parse_args()

    # Baseline runs
    base_target_probs: List[float] = []
    base_coarse_probs: List[float] = []
    base_eta: List[float] = []

    # QSVM-SEGC runs
    q_target_probs: List[float] = []
    q_coarse_probs: List[float] = []
    q_eta: List[float] = []

    n_qubits = 7
    mask = "****0000"
    target = "1010000"

    for i in range(args.runs):
        logging.info(f"Run {i+1}/{args.runs} for k={args.k}")

        t_p, c_p, kc, kf = run_baseline(n_qubits, mask, target, args.shots)
        base_target_probs.append(t_p)
        base_coarse_probs.append(c_p)
        base_eta.append(eta_vs_grover(
            n_qubits, 2 ** mask.count('*'), kc, kf, t_p))

        tq_p, cq_p, kc2, kf2 = run_qsvm_segc(
            n_qubits, mask, target, args.shots, k=args.k, feedback_rounds=args.feedback)
        q_target_probs.append(tq_p)
        q_coarse_probs.append(cq_p)
        q_eta.append(eta_vs_grover(
            n_qubits, 2 ** mask.count('*'), kc2, kf2, tq_p))

    logging.info("Statistical Analysis:")
    summarize("Baseline target prob", base_target_probs)
    summarize("Baseline coarse prob", base_coarse_probs)
    summarize("Baseline eta", base_eta)

    summarize("QSVM-SEGC target prob", q_target_probs)
    summarize("QSVM-SEGC coarse prob", q_coarse_probs)
    summarize("QSVM-SEGC eta", q_eta)

    # Quick relative improvement
    b_mean = float(np.mean(base_target_probs)) if base_target_probs else 0.0
    q_mean = float(np.mean(q_target_probs)) if q_target_probs else 0.0
    rel = 0.0 if b_mean == 0 else 100.0 * (q_mean - b_mean) / b_mean
    logging.info(f"Relative improvement (target prob): {rel:.2f}%")

    # Brief objective linkage lines (for your report)
    logging.info("Objective checks:")
    logging.info(
        "1) Amplify target efficiently: report mean p and eta above. Expect QSVM-SEGC >= baseline if feedback works.")
    logging.info(
        "2) Reduce dispersion: compare SD across runs from the summaries.")
    logging.info(
        "3) Integrate QSVM effectively: marked set should stabilize across rounds (log output).")
    logging.info(
        "4) Efficiency vs bound (eta): compare baseline vs QSVM-SEGC means; both << 1 is expected on NISQ-like logic.")


if __name__ == "__main__":
    main()
