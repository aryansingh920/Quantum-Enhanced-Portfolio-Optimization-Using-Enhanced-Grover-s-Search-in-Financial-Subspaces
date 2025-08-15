import logging
from typing import List, Tuple, Optional

import numpy as np
from sklearn.svm import SVC

# Expect SEGCAlgorithm from segc.py
# from segc import SEGCAlgorithm  # avoid circular import in this snippet

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


class QSVM:
    """
    QSVM guidance for SEGC.

    Modes:
      - "state": train on coarse-subspace states with constructed labels (target + nearest neighbours = 1)
      - "market": train on a provided market dataframe (NOT used in this experiment)
      - "auto": chooses "market" if a suitable dataframe with label_col exists; else "state"

    For your current experiment, use mode="state".
    """

    def __init__(self,
                 segc,
                 feature_map_type: str = "statevector",
                 topk: int = 2,
                 mode: str = "state",
                 market_df=None,
                 label_col: str = "Close"):
        self.segc = segc
        self.feature_map_type = feature_map_type
        self.topk = int(topk)
        self.mode = mode
        self.market_df = market_df
        self.label_col = label_col
        self.svc: Optional[SVC] = None

    # ===== Feature maps =====

    def _bitstring_to_features(self, bitstr_std: str) -> np.ndarray:
        """
        Very simple feature embedding for bitstring: map {0,1} to [0,pi] and use sin/cos pairs.
        Produces 2*n features. You can swap in a proper ZZFeatureMap if you want.
        """
        x = np.array([float(b) for b in bitstr_std], dtype=float) * np.pi
        # stack [sin, cos] per qubit
        feats = np.concatenate([np.sin(x), np.cos(x)], axis=0)
        return feats

    def _make_features(self, states_std: List[str], feature_map_type: str) -> np.ndarray:
        if feature_map_type == "statevector":
            return np.stack([self._bitstring_to_features(s) for s in states_std], axis=0)
        # Fallback: raw bits
        return np.stack([[int(b) for b in s] for s in states_std], axis=0)

    # ===== Modes =====

    def _auto_mode(self) -> str:
        if self.mode in ("market", "state"):
            return self.mode
        if (self.market_df is not None) and (self.label_col in self.market_df.columns):
            return "market"
        return "state"

    def _build_market_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        df = self.market_df.copy()
        if self.label_col not in df.columns:
            raise KeyError(
                f"Column '{self.label_col}' not in market dataframe")
        df["label"] = (df[self.label_col].shift(-1) >
                       df[self.label_col]).astype(int)
        df = df.dropna()
        y = df["label"].to_numpy(dtype=int)
        X = df.drop(columns=[self.label_col, "label"],
                    errors="ignore").to_numpy(dtype=float)
        return X, y

    def _construct_state_labels(self, coarse_states_std: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        target_std = self.segc.target_standard

        def hamming(a: str, b: str) -> int:
            return sum(ch1 != ch2 for ch1, ch2 in zip(a, b))

        X = self._make_features(coarse_states_std, self.feature_map_type)
        d = np.array([hamming(s, target_std)
                     for s in coarse_states_std], dtype=int)
        m = max(1, self.topk)
        # nearest m + target (dist 0)
        pos_idx = set(np.argsort(d)[:m+1].tolist())
        y = np.array([1 if i in pos_idx else 0 for i in range(
            len(coarse_states_std))], dtype=int)
        return X, y

    # ===== Training / Selection =====

    def train(self, coarse_states_std: Optional[List[str]] = None):
        mode = self._auto_mode()
        if mode == "market":
            X, y = self._build_market_labels()
        else:
            if not coarse_states_std:
                raise ValueError(
                    "State mode requires coarse_states_std (bitstrings in Hc).")
            X, y = self._construct_state_labels(coarse_states_std)

        # Keep it simple and robust
        # decision_function available
        self.svc = SVC(kernel="rbf", probability=False)
        self.svc.fit(X, y)
        logging.info("QSVM training completed.")

    def select_topk_states(self, coarse_states_std: List[str]) -> Tuple[List[str], np.ndarray]:
        if self.svc is None:
            raise RuntimeError("Call train() before select_topk_states().")
        X = self._make_features(coarse_states_std, self.feature_map_type)
        scores = self.svc.decision_function(
            X).ravel()  # higher -> more likely class 1
        idxs = np.argsort(scores)[-max(1, self.topk):]
        picked = set(int(i) for i in idxs.tolist())
        # Always include target if present
        try:
            t_idx = coarse_states_std.index(self.segc.target_standard)
            picked.add(int(t_idx))
        except ValueError:
            pass
        marked = [coarse_states_std[i] for i in sorted(picked)]
        logging.debug(f"Selected states: {marked}")
        return marked, scores
