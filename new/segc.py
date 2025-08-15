import math
import logging
from typing import List, Dict, Tuple, Optional

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector

# Configure logging defaults (override in caller if needed)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")


def std_to_qiskit(bitstr_std: str) -> str:
    """
    Convert standard-ordering bitstring (q_{n-1} ... q_0) to Qiskit's little-endian ordering.
    We assume the target and coarse masks are provided in standard ordering.
    """
    return bitstr_std[::-1]


def matches_mask(std_bits: str, mask: str) -> bool:
    """Return True if std_bits matches a coarse mask like '****0000' (stars = don't care)."""
    for b, m in zip(std_bits, mask):
        if m != '*' and m != b:
            return False
    return True


class SEGCAlgorithm:
    """
    SEGC: Subspace-Enhanced Grover with Coarse (mask) then Fine (target-aware) phases.

    Public config:
      - n_qubits: total problem qubits (without ancilla)
      - coarse_mask: standard-order mask, e.g. '****0000'
      - target: target in standard ordering, e.g. '1010000'
      - coarse_iterations / fine_iterations: if None, auto-chosen from Nc
    """

    def __init__(self,
                 n_qubits: int = 7,
                 coarse_mask: str = "****0000",
                 target: str = "1010000",
                 coarse_iterations: Optional[int] = None,
                 fine_iterations: Optional[int] = None,
                 shots: int = 20000):
        self.n = n_qubits
        self.N = 2 ** n_qubits
        self.coarse_mask = coarse_mask
        self.target_standard = target  # standard ordering
        self.target = std_to_qiskit(target)  # qiskit ordering
        self.shots = shots

        # Derived: size of coarse subspace
        self.N_c = 2 ** self.coarse_mask.count('*')

        # Circuit with ancilla for phase kickback if needed
        self.qr = QuantumRegister(self.n, "q")
        self.ar = QuantumRegister(1, "a")
        self.cr = ClassicalRegister(self.n, "c")
        self.qc = QuantumCircuit(self.qr, self.ar, self.cr)

        logging.debug(f"SEGC init: n_qubits={self.n}, coarse_mask={self.coarse_mask}, "
                      f"target={self.target_standard}")

        # Initialize |+>^n on problem qubits, ancilla |-> for phase kickback
        self._initialize_superposition()

        # Choose kc/kf
        theta_c = math.asin(math.sqrt(self.N_c / self.N))
        kc_opt = max(1, round((math.pi / (4 * theta_c)) - 0.5))
        theta_f = math.asin(math.sqrt(1.0 / max(self.N_c, 1)))
        kf_opt = max(1, round((math.pi / (4 * theta_f)) - 0.5))

        if coarse_iterations is None:
            coarse_iterations = kc_opt
        if fine_iterations is None:
            fine_iterations = kf_opt

        self.coarse_iterations = coarse_iterations
        self.fine_iterations = fine_iterations

        logging.info(
            f"Auto-set iterations: coarse={self.coarse_iterations}, fine={self.fine_iterations} (Nc={self.N_c})")
        logging.debug(
            f"SEGC config: k_c={self.coarse_iterations}, k_f={self.fine_iterations}")

    # --- Circuit building primitives ---

    def _initialize_superposition(self):
        # Put ancilla in |-> = (|0>-|1>)/sqrt(2) for phase kickback
        self.qc.x(self.ar[0])
        self.qc.h(self.ar[0])
        # Uniform superposition on problem register
        for q in self.qr:
            self.qc.h(q)
        logging.debug("Initialized uniform superposition")

    def apply_coarse_oracle(self):
        """
        Phase-flip any basis state within the coarse mask. A simple demonstration oracle that
        controls on fixed 0/1 positions from the mask and flips ancilla.
        """
        fixed_ones = [i for i, ch in enumerate(
            self.coarse_mask[::-1]) if ch == '1']  # qiskit order positions
        fixed_zeros = [i for i, ch in enumerate(
            self.coarse_mask[::-1]) if ch == '0']

        # X on zeros -> convert to 1-controls
        for idx in fixed_zeros:
            self.qc.x(self.qr[idx])

        # Multi-controlled Z on ancilla conditioned by the fixed bits
        controls = [self.qr[i] for i in fixed_ones + fixed_zeros]
        if len(controls) > 0:
            self.qc.mcx(controls, self.ar[0])  # flip ancilla phase

        # Undo X on zeros
        for idx in fixed_zeros:
            self.qc.x(self.qr[idx])

        logging.debug("Applied coarse oracle")

    def coarse_diffuser(self):
        """
        Grover diffuser over the entire problem register (simple version).
        For strict subspace diffusion you’d restrict to Hc indices only – we keep this simple and
        rely on kc selection to avoid overshoot.
        """
        # H, X
        for q in self.qr:
            self.qc.h(q)
            self.qc.x(q)
        # multi-controlled Z targeting the ancilla as a phase kickback mirror
        self.qc.h(self.qr[-1])
        self.qc.mcx(self.qr[:-1], self.qr[-1])
        self.qc.h(self.qr[-1])
        # X, H
        for q in self.qr:
            self.qc.x(q)
            self.qc.h(q)
        logging.debug("Applied coarse diffuser")

    def coarse_iterate(self):
        logging.info(f"Running {self.coarse_iterations} coarse iterations")
        for _ in range(self.coarse_iterations):
            self.apply_coarse_oracle()
            self.coarse_diffuser()
        logging.info(f"Completed {self.coarse_iterations} coarse iterations")

    # --- Fine phase ---

    def apply_fine_oracle(self, marked_standard_bitstrings: List[str]):
        """
        Phase-flip specific STANDARD-ordering bitstrings (e.g., target and top-k from QSVM).
        Converted internally to Qiskit ordering and controlled to ancilla.
        """
        if not marked_standard_bitstrings:
            return
        for s_std in marked_standard_bitstrings:
            s_q = std_to_qiskit(s_std)
            # For each bit == '0', apply X to turn to control-on-1
            zero_idx = [i for i, b in enumerate(s_q) if b == '0']
            for idx in zero_idx:
                self.qc.x(self.qr[idx])
            # multi-controlled Z via ancilla
            self.qc.mcx(self.qr[:], self.ar[0])
            # undo X
            for idx in zero_idx:
                self.qc.x(self.qr[idx])
        logging.debug(
            f"Applied fine oracle for states: {marked_standard_bitstrings}")

    def partial_diffuser(self, qc: QuantumCircuit, qr: QuantumRegister, ar: QuantumRegister):
        """
        Demonstration partial diffuser: we mirror around the mean but keep the oracle structure.
        If you already have a stricter subspace-only diffuser, wire it here.
        """
        # Same diffuser as coarse for simplicity (replace with subspace diffuser if you have it).
        for q in qr:
            qc.h(q)
            qc.x(q)
        qc.h(qr[-1])
        qc.mcx(qr[:-1], qr[-1])
        qc.h(qr[-1])
        for q in qr:
            qc.x(q)
            qc.h(q)
        logging.debug("Applied partial diffuser")

    def fine_iterate(self, marked_standard_bitstrings: List[str]):
        logging.debug(f"Running {self.fine_iterations} fine iterations")
        for i in range(self.fine_iterations):
            logging.debug(f"Fine iteration {i+1}/{self.fine_iterations}")
            self.apply_fine_oracle(marked_standard_bitstrings)
            self.partial_diffuser(self.qc, self.qr, self.ar)
        logging.info(f"Completed {self.fine_iterations} fine iterations")

    # --- Analysis ---

    def get_statevector(self) -> Statevector:
        return Statevector.from_instruction(self.qc)

    def analyze_state(self) -> Tuple[float, float, Dict[str, float]]:
        """
        Returns: (target_prob, coarse_prob, probs_std) where probs_std maps STANDARD ordering
        strings (ancilla traced out) to probabilities.
        """
        sv = self.get_statevector()
        probs_std: Dict[str, float] = {}
        coarse_prob = 0.0
        target_prob = 0.0

        # Iterate over problem space only; ancilla is traced out by summing both branches
        for i in range(self.N):
            bits_q = format(i, f"0{self.n}b")  # qiskit ordering
            bits_std = bits_q[::-1]
            # prob = |amp(anc=0, state)|^2 + |amp(anc=1, state)|^2
            idx0 = int('0' + bits_q, 2)
            idx1 = int('1' + bits_q, 2)
            p = 0.0
            if idx0 < len(sv.data):
                p += abs(sv.data[idx0]) ** 2
            if idx1 < len(sv.data):
                p += abs(sv.data[idx1]) ** 2

            probs_std[bits_std] = p
            if matches_mask(bits_std, self.coarse_mask):
                coarse_prob += p
            if bits_std == self.target_standard:
                target_prob += p

        return target_prob, coarse_prob, probs_std

    def measure(self, shots: Optional[int] = None) -> Dict[str, int]:
        s = shots or self.shots
        qc = self.qc.copy()
        qc.measure(self.qr, self.cr)
        # Fake measurement by sampling statevector probabilities (since we’re in pure python file)
        sv = Statevector.from_instruction(self.qc)
        probs = []
        for i in range(self.N):
            bits_q = format(i, f"0{self.n}b")
            idx0 = int('0' + bits_q, 2)
            idx1 = int('1' + bits_q, 2)
            p = 0.0
            if idx0 < len(sv.data):
                p += abs(sv.data[idx0]) ** 2
            if idx1 < len(sv.data):
                p += abs(sv.data[idx1]) ** 2
            probs.append(p)
        # sample
        import random
        outcomes = random.choices(range(self.N), weights=probs, k=s)
        counts: Dict[str, int] = {}
        for idx in outcomes:
            bits_q = format(idx, f"0{self.n}b")
            bits_std = bits_q[::-1]
            counts[bits_std] = counts.get(bits_std, 0) + 1

        # Debug top outcomes
        top5 = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
        logging.debug(f"Measurement counts (top 5): {top5}")
        return counts
