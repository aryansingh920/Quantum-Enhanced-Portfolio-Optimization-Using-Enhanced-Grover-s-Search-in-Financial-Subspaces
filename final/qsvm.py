# qsvm.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Sequence, Optional
import numpy as np
import logging  # UPDATED: Added for enhanced logging

try:
    from qiskit_machine_learning.algorithms.classifiers import QSVC  # type: ignore
    _HAS_QSVC = True
except Exception:
    _HAS_QSVC = False

try:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
except Exception:
    SVC = None  # type: ignore
    StandardScaler = None  # type: ignore
    make_pipeline = None  # type: ignore


@dataclass
class QSVMConfig:
    use_quantum: bool = False
    C: float = 1.0
    gamma: str = "scale"
    top_k: Optional[int] = None
    threshold: Optional[float] = 0.0


class QSVM:
    def __init__(self, cfg: QSVMConfig = QSVMConfig()) -> None:
        self.cfg = cfg
        self.model = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.cfg.use_quantum and _HAS_QSVC:
            self.model = QSVC()
            self.model.fit(X, y)
            # UPDATED: Log after fit
            logging.info("QSVM fitted using quantum QSVC.")
        else:
            if SVC is None:
                raise RuntimeError("scikit-learn not available")
            self.model = make_pipeline(StandardScaler(), SVC(
                C=self.cfg.C, gamma=self.cfg.gamma, probability=False))
            self.model.fit(X, y)
            # UPDATED: Log model details
            svm = self.model.named_steps['svc']
            logging.info(
                f"Classical SVM fitted: C={svm.C}, gamma={svm.gamma}, n_support={len(svm.support_)}")

    def _decision(self, X: np.ndarray) -> np.ndarray:
        if hasattr(self.model, "decision_function"):
            return self.model.decision_function(X)  # type: ignore
        preds = self.model.predict(X)  # type: ignore
        return 2 * (preds.astype(float)) - 1.0

    def select_marked_states(
        self,
        features_map: Dict[int, np.ndarray],
        coarse_selector: Callable[[int], bool],
        n: int,
    ) -> List[str]:
        """Return selected MSB..LSB bitstrings inside coarse subspace."""
        if self.model is None:
            raise RuntimeError("fit() first")

        idx = [i for i in features_map.keys() if coarse_selector(i)]
        if not idx:
            return []

        X = np.vstack([features_map[i] for i in idx])
        scores = self._decision(X)

        # UPDATED: Log decision scores for debugging
        logging.info(f"QSVM decision scores for {len(idx)} coarse states:")
        for j, s in enumerate(scores):
            logging.info(f"  State {idx[j]}: score={s:.4f}")

        chosen = list(range(len(idx)))
        if self.cfg.threshold is not None:
            chosen = [j for j in chosen if scores[j]
                      >= float(self.cfg.threshold)]
            logging.info(
                f"After threshold {self.cfg.threshold}: {len(chosen)} states remain.")

        if self.cfg.top_k is not None and self.cfg.top_k > 0 and len(chosen) > self.cfg.top_k:
            order = np.argsort(scores[chosen])[::-1]
            chosen = [chosen[k] for k in order[: self.cfg.top_k]]
            logging.info(
                f"After top_k {self.cfg.top_k}: Selected states { [idx[j] for j in chosen] }")

        sel = [idx[j] for j in chosen]
        bitstrings_msb: List[str] = []
        for s in sel:
            bits_lsb = [(s >> b) & 1 for b in range(n)]
            msb = ''.join(str(b) for b in reversed(bits_lsb))
            bitstrings_msb.append(msb)
        return bitstrings_msb
