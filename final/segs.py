# segs.py
# Strict SEGS per report:
# - Uniform init on system; phase ancilla prepared in |->
# - Coarse oracle: phase-flip on states matching coarse mask (using ancilla kickback)
# - Coarse diffuser: *global* diffuser over all system qubits
# - Fine oracle: phase-flip specific states (MCX to ancilla)
# - Partial diffuser: only on free ('*') qubits from mask
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Sequence, Tuple

from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector


@dataclass
class IterationSchedule:
    k_coarse: int
    k_fine: int


class SEGSAlgorithm:
    """
    Bit order conventions:
      - Mask string is MSB..LSB (leftmost is bit n-1, rightmost is bit 0).
      - Probabilities dict keys from qiskit are also MSB..LSB.
    Qiskit qubit index i refers to LSB==0 by default.
    """

    def __init__(self, n_qubits: int, coarse_mask: str, schedule: IterationSchedule) -> None:
        assert len(
            coarse_mask) == n_qubits, "coarse_mask length must equal n_qubits"
        self.n = n_qubits
        self.mask = coarse_mask
        self.schedule = schedule

        self.qr = QuantumRegister(n_qubits, 'q')
        self.ar = QuantumRegister(1, 'anc')
        self.cr = ClassicalRegister(n_qubits, 'c')
        self.qc = QuantumCircuit(self.qr, self.ar, self.cr, name="SEGS")

    # ---------------- core steps ----------------
    def initialize(self) -> None:
        # |s> on system; ancilla = |->  for phase kickback (X then H)
        for i in range(self.n):
            self.qc.h(self.qr[i])
        self.qc.x(self.ar[0])
        self.qc.h(self.ar[0])
        logging.info(
            "Circuit initialized: Hadamard on system, |-> on ancilla.")

    def _fixed_bits_lsb(self) -> List[Tuple[int, str]]:
        rev = self.mask[::-1]  # LSB-first view
        fixed = [(i, b) for i, b in enumerate(rev) if b in ('0', '1')]
        logging.debug(f"Fixed bits (LSB): {fixed}")
        return fixed

    def _free_bits_lsb(self) -> List[int]:
        rev = self.mask[::-1]
        free = [i for i, b in enumerate(rev) if b == '*']
        logging.debug(f"Free bits (LSB): {free}")
        return free

    def coarse_oracle(self) -> None:
        fixed = self._fixed_bits_lsb()
        if not fixed:
            logging.debug("Coarse oracle skipped (mask is all '*').")
            return

        zeros = [i for i, b in fixed if b == '0']
        for i in zeros:
            self.qc.x(self.qr[i])

        ctrls = [self.qr[i] for i, _ in fixed]
        self.qc.mcx(ctrls, self.ar[0])

        for i in zeros:
            self.qc.x(self.qr[i])
        logging.debug("Coarse oracle applied.")

    def coarse_diffuser(self) -> None:
        for i in range(self.n):
            self.qc.h(self.qr[i])
            self.qc.x(self.qr[i])
        self.qc.h(self.qr[self.n - 1])
        self.qc.mcx([self.qr[i]
                    for i in range(self.n - 1)], self.qr[self.n - 1])
        self.qc.h(self.qr[self.n - 1])
        for i in range(self.n):
            self.qc.x(self.qr[i])
            self.qc.h(self.qr[i])
        logging.debug("Coarse diffuser applied.")

    def fine_oracle_states(self, states_msb: Sequence[str]) -> None:
        for s_msb in states_msb:
            assert len(s_msb) == self.n
            s_lsb = s_msb[::-1]
            zeros = [i for i, ch in enumerate(s_lsb) if ch == '0']
            for i in zeros:
                self.qc.x(self.qr[i])
            self.qc.mcx([self.qr[i] for i in range(self.n)], self.ar[0])
            for i in zeros:
                self.qc.x(self.qr[i])
        logging.debug(f"Fine oracle applied to {len(states_msb)} states.")

    def partial_diffuser(self) -> None:
        free = self._free_bits_lsb()
        if not free:
            logging.debug("Partial diffuser skipped (no free bits).")
            return

        for i in free:
            self.qc.h(self.qr[i])
            self.qc.x(self.qr[i])

        t = free[-1]
        if len(free) == 1:
            self.qc.h(self.qr[t])
            self.qc.z(self.qr[t])
            self.qc.h(self.qr[t])
        else:
            self.qc.h(self.qr[t])
            self.qc.mcx([self.qr[i] for i in free[:-1]], self.qr[t])
            self.qc.h(self.qr[t])

        for i in free:
            self.qc.x(self.qr[i])
            self.qc.h(self.qr[i])
        logging.debug("Partial diffuser applied.")

    # ---------------- driver helpers ----------------
    def run_coarse(self) -> None:
        for _ in range(self.schedule.k_coarse):
            self.coarse_oracle()
            self.coarse_diffuser()

    def run_fine(self, marked_states_msb: Sequence[str]) -> None:
        for _ in range(self.schedule.k_fine):
            self.fine_oracle_states(marked_states_msb)
            self.partial_diffuser()
