# main.py
# Pipeline:
#  - Build 7-bit encoding from CSV
#  - Choose coarse mask (MSB..LSB) from fixed features
#  - Baseline SEGS: mark top-K frequent coarse states
#  - Dynamic SEGS (QSVM): mark classifier-selected coarse states
#  - Log after each coarse/fine oracle and diffuser (system-only)
from __future__ import annotations

import argparse
import logging
import math
from typing import Dict, List, Sequence, Callable, Tuple, Optional

import numpy as np
import pandas as pd
from qiskit.quantum_info import Statevector

from segs import SEGSAlgorithm, IterationSchedule
from qsvm import QSVM, QSVMConfig

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

REQUIRED_MIN = ['returns', 'rsi_scaled', 'volatility_scaled',
                'ma5_scaled', 'ma20_scaled', 'close_scaled', 'volume_scaled']

# ---- encoding helpers ----


def _require_cols(df: pd.DataFrame, cols: Sequence[str], path: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns: {missing}")


def int_to_bits_lsb(x: int, n: int) -> List[int]:
    return [(x >> i) & 1 for i in range(n)]


def bits_lsb_to_msb(bits: List[int]) -> str:
    return ''.join(str(b) for b in reversed(bits))


def encode_row(row: pd.Series) -> Tuple[int, List[int]]:
    # LSB-first mapping (bit 0 rightmost):
    b0 = 1 if float(row['returns']) > 0 else 0
    b1 = 1 if float(row.get('rsi_scaled', 0.5)) > 0.7 else 0
    b2 = 1 if float(row.get('volatility_scaled', 0.5)) > 0.5 else 0
    b3 = 1 if float(row.get('ma5_scaled', 0.5)) - \
        float(row.get('ma20_scaled', 0.5)) > 0 else 0
    b4 = 1 if float(row.get('close_scaled', 0.5)) > 0.5 else 0
    b5 = 1 if float(row.get('volume_scaled', 0.5)) > 0.5 else 0
    if 'regime' in row and not pd.isna(row['regime']):
        b6 = 1 if int(row['regime']) == 1 else 0
    else:
        b6 = 1 if (b0 == 1 and b3 == 1) else 0
    bits = [b0, b1, b2, b3, b4, b5, b6]
    idx = 0
    for i, bit in enumerate(bits):
        idx |= (bit << i)
    return idx, bits


def load_csvs(paths: Sequence[str]) -> pd.DataFrame:
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        _require_cols(df, REQUIRED_MIN, p)
        df['__source__'] = p
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    logging.info(
        f"Loaded {len(combined)} rows from {len(paths)} CSV(s). Columns: {combined.columns.tolist()}")
    return combined


def aggregate_features(df: pd.DataFrame) -> Tuple[Dict[int, np.ndarray], Dict[int, int], Dict[int, int]]:
    feats: Dict[int, List[np.ndarray]] = {}
    labels: Dict[int, List[int]] = {}
    freqs: Dict[int, int] = {}
    n = 7

    for _, row in df.iterrows():
        idx, bits = encode_row(row)
        v = np.array([
            float(row.get('close_scaled', 0.5)),
            float(row.get('high_scaled', 0.5)),
            float(row.get('low_scaled', 0.5)),
            float(row.get('rsi_scaled', 0.5)),
            float(row.get('volatility_scaled', 0.5)),
            float(row.get('ma5_scaled', 0.5)) -
            float(row.get('ma20_scaled', 0.5)),
            float(row.get('volume_scaled', 0.5)),
            float(row.get('returns', 0.0)),
        ], dtype=float)
        feats.setdefault(idx, []).append(v)
        freqs[idx] = freqs.get(idx, 0) + 1
        lab = int(bits[0] == 1 and bits[3] == 1) if 'regime' not in row or pd.isna(
            row['regime']) else (1 if int(row['regime']) == 1 else 0)
        labels.setdefault(idx, []).append(lab)

    features_map = {i: np.mean(np.vstack(v), axis=0) for i, v in feats.items()}
    label_map = {i: (1 if (np.mean(v) >= 0.5) else 0)
                 for i, v in labels.items()}

    logging.info(
        f"Aggregated {len(features_map)} unique states in feature subspace.")
    for i in sorted(features_map.keys()):
        logging.info(
            f"State {i} (MSB: {bits_lsb_to_msb(int_to_bits_lsb(i, n))}): Freq={freqs.get(i, 0)}, Label={label_map[i]}, Avg Features={features_map[i].tolist()}")

    return features_map, label_map, freqs

# ---- mask & selector ----


def build_coarse_mask() -> str:
    # Fixes b3=1 (MA crossover), b0=1 (positive returns), as per report Ch5 example with Nc=32
    mask = "***1**1"
    logging.info(f"Built coarse mask: {mask} (MSB..LSB)")
    return mask


def make_coarse_selector_from_mask(mask_msb: str) -> Callable[[int], bool]:
    rev = mask_msb[::-1]  # LSB-first

    def sel(idx: int) -> bool:
        for i, ch in enumerate(rev):
            if ch == '*':
                continue
            want = 1 if ch == '1' else 0
            if ((idx >> i) & 1) != want:
                return False
        return True
    return sel

# ---- logging ----


def system_probs(qc, n: int) -> Dict[str, float]:
    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities_dict(qargs=list(range(n)))
    return {k: float(v) for k, v in probs.items()}


def coarse_mass_from_probs(probs: Dict[str, float], mask_msb: str) -> float:
    total = 0.0
    for bitstr, p in probs.items():
        ok = True
        for pos, ch in enumerate(mask_msb):
            if ch == '*':
                continue
            if bitstr[pos] != ch:
                ok = False
                break
        if ok:
            total += p
    return total


def topk(probs: Dict[str, float], k: int = 12) -> List[Tuple[str, float]]:
    return sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:k]


def log_snapshot(fh, label: str, qc, n: int, mask_msb: str, highlights_msb: List[str]) -> None:
    probs = system_probs(qc, n)
    mass = coarse_mass_from_probs(probs, mask_msb)
    fh.write(f"\n=== {label} ===\n")
    fh.write(f"Coarse mass: {mass:.6f}\n")
    if highlights_msb:
        fh.write("Highlights (target states):\n")
        total_highlight_prob = sum(probs.get(h, 0.0)
                                   for h in highlights_msb)  # UPDATED: Compute p_any
        fh.write(
            f"  Total highlight prob (p_any): {total_highlight_prob:.6f}\n")
        for h in highlights_msb:
            fh.write(f"  {h}: {probs.get(h, 0.0):.6f}\n")
    fh.write("Top states (global):\n")
    for k, v in topk(probs):
        fh.write(f"  {k}: {v:.6f}\n")
    fh.write("Top states (coarse-only):\n")
    c_only = {k: v for k, v in probs.items() if all(
        m == '*' or k[i] == m for i, m in enumerate(mask_msb))}
    for k, v in topk(c_only):
        fh.write(f"  {k}: {v:.6f}\n")
    fh.flush()
    logging.info(
        f"Snapshot {label}: Coarse mass={mass:.6f}, Top highlight prob={max(probs.get(h, 0) for h in highlights_msb) if highlights_msb else 0:.6f}")

# ---- baselines ----


def pick_fine_targets_by_frequency(freqs: Dict[int, int], mask_msb: str, top_K: int, n: int = 7) -> List[str]:
    sel = make_coarse_selector_from_mask(mask_msb)
    inside = [(i, f) for i, f in freqs.items() if sel(i)]
    inside.sort(key=lambda kv: kv[1], reverse=True)
    chosen = [i for i, _ in inside[:top_K]]
    targets_msb = [bits_lsb_to_msb(int_to_bits_lsb(i, n)) for i in chosen]
    logging.info(
        f"Baseline selected {len(targets_msb)} fine targets (MSB..LSB): {targets_msb}")
    return targets_msb

# ---- runs ----


def run_segs_baseline(df: pd.DataFrame, topk_states: int, logfile: str) -> None:
    n = 7
    features_map, label_map, freqs = aggregate_features(df)
    mask = build_coarse_mask()
    targets_msb = pick_fine_targets_by_frequency(freqs, mask, topk_states, n=n)
    if not targets_msb:
        logging.warning("No baseline targets selected; skipping run.")
        return

    N = 2 ** n
    Nc = 2 ** mask.count('*')
    theta_c = math.asin(math.sqrt(Nc / N))
    k_c = max(1, round((math.pi / (4 * theta_c)) - 0.5))
    t = len(targets_msb)
    theta_f = math.asin(math.sqrt(t / Nc)) if t > 0 else 0.0
    k_f = max(1, round((math.pi / (4 * theta_f)) - 0.5)) if theta_f > 0 else 1
    sched = IterationSchedule(k_coarse=k_c, k_fine=k_f)
    logging.info(
        f"Baseline schedule: k_coarse={k_c} (theta_c={theta_c:.4f}), k_fine={k_f} (theta_f={theta_f:.4f}, t={t})")

    segs = SEGSAlgorithm(n_qubits=n, coarse_mask=mask, schedule=sched)
    segs.initialize()
    with open(logfile, "w", encoding="utf-8") as fh:
        log_snapshot(fh, "init/hadamard", segs.qc, n, mask, targets_msb)
        for i in range(sched.k_coarse):
            segs.coarse_oracle()
            log_snapshot(
                fh, f"coarse/oracle_{i+1}", segs.qc, n, mask, targets_msb)
            segs.coarse_diffuser()
            log_snapshot(
                fh, f"coarse/diffuser_{i+1}", segs.qc, n, mask, targets_msb)
        for i in range(sched.k_fine):
            segs.fine_oracle_states(targets_msb)
            log_snapshot(fh, f"fine/oracle_{i+1}",
                         segs.qc, n, mask, targets_msb)
            segs.partial_diffuser()
            log_snapshot(
                fh, f"fine/diffuser_{i+1}", segs.qc, n, mask, targets_msb)

    logging.info(f"Baseline SEGS log written to {logfile}")


def run_segs_qsvm(df: pd.DataFrame, qtopk: int, qthreshold: float, logfile: str) -> None:
    n = 7
    features_map, label_map, freqs = aggregate_features(df)
    mask = build_coarse_mask()
    sel = make_coarse_selector_from_mask(mask)

    idxs = list(features_map.keys())
    X = np.vstack([features_map[i] for i in idxs])
    y = np.array([label_map[i] for i in idxs], dtype=int)

    # UPDATED: Enable quantum if available, per report Ch6 QSVM use
    has_qsvc = False
    try:
        from qiskit_machine_learning.algorithms.classifiers import QSVC
        has_qsvc = True
    except ImportError:
        has_qsvc = False
    clf = QSVM(QSVMConfig(use_quantum=has_qsvc,
               top_k=qtopk, threshold=qthreshold))
    clf.fit(X, y)

    dim = next(iter(features_map.values())).shape[0] if features_map else 8
    full_map = {i: features_map.get(i, np.zeros(
        dim, dtype=float)) for i in range(2 ** n)}
    marked_msb = clf.select_marked_states(full_map, coarse_selector=sel, n=n)
    logging.info(
        f"QSVM selected {len(marked_msb)} coarse states (MSB..LSB): {marked_msb}")

    if not marked_msb:
        logging.warning("No QSVM targets selected; skipping run.")
        return

    N = 2 ** n
    Nc = 2 ** mask.count('*')
    theta_c = math.asin(math.sqrt(Nc / N))
    k_c = max(1, round((math.pi / (4 * theta_c)) - 0.5))
    t = len(marked_msb)
    theta_f = math.asin(math.sqrt(t / Nc)) if t > 0 else 0.0
    k_f = max(1, round((math.pi / (4 * theta_f)) - 0.5)) if theta_f > 0 else 1
    sched = IterationSchedule(k_coarse=k_c, k_fine=k_f)
    logging.info(
        f"QSVM SEGS schedule: k_coarse={k_c} (theta_c={theta_c:.4f}), k_fine={k_f} (theta_f={theta_f:.4f}, t={t})")

    segs = SEGSAlgorithm(n_qubits=n, coarse_mask=mask, schedule=sched)
    segs.initialize()
    with open(logfile, "w", encoding="utf-8") as fh:
        log_snapshot(fh, "init/hadamard", segs.qc, n, mask, marked_msb)
        for i in range(sched.k_coarse):
            segs.coarse_oracle()
            log_snapshot(
                fh, f"coarse/oracle_{i+1}", segs.qc, n, mask, marked_msb)
            segs.coarse_diffuser()
            log_snapshot(
                fh, f"coarse/diffuser_{i+1}", segs.qc, n, mask, marked_msb)
        for i in range(sched.k_fine):
            segs.fine_oracle_states(marked_msb)
            log_snapshot(fh, f"fine/oracle_{i+1}",
                         segs.qc, n, mask, marked_msb)
            segs.partial_diffuser()
            log_snapshot(
                fh, f"fine/diffuser_{i+1}", segs.qc, n, mask, marked_msb)

    logging.info(f"QSVM SEGS log written to {logfile}")

# ---- CLI ----


def main() -> None:
    ap = argparse.ArgumentParser(
        description="SEGS / SEGS+QSVM (report-faithful) with logging")
    ap.add_argument('--csv', action='append', required=True,
                    help='CSV path(s) with scaled features')
    ap.add_argument(
        '--mode', choices=['regime', 'portfolio'], default='regime')
    ap.add_argument('--topk', type=int, default=3,
                    help='Top-K for baseline fine targets')
    ap.add_argument('--qtopk', type=int, default=5,
                    help='Top-K for QSVM selection')
    ap.add_argument('--qthreshold', type=float, default=0.0,
                    help='QSVM decision threshold')
    ap.add_argument('--logfile', type=str, default='segs_run.log')
    args = ap.parse_args()

    df = load_csvs(args.csv)

    run_segs_baseline(df, topk_states=args.topk, logfile=args.logfile)
    run_segs_qsvm(df, qtopk=args.qtopk, qthreshold=args.qthreshold,
                  logfile=args.logfile.replace('.log', '_qsvm.log'))


if __name__ == '__main__':
    main()
