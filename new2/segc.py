import math
import numpy as np
import logging
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector


class SEGCAlgorithm:
    def __init__(self, n_qubits, coarse_mask, target, coarse_iterations=None, fine_iterations=None):
        self.n_qubits = n_qubits
        self.coarse_mask = coarse_mask
        self.target = target  # Qiskit ordering
        self.target_standard = target[::-1]  # Standard ordering
        self.N = 2 ** n_qubits
        self.N_c = 2 ** coarse_mask.count('*')

        # Auto-compute optimal Grover iteration counts based on mask
        import math, logging
        theta_c = math.asin(math.sqrt(self.N_c / self.N))
        kc_opt = max(1, round((math.pi / (4*theta_c)) - 0.5))
        theta_f = math.asin(math.sqrt(1.0 / max(self.N_c,1)))
        kf_opt = max(1, round((math.pi / (4*theta_f)) - 0.5))
        # Override provided iterations with optimal values
        self.coarse_iterations = kc_opt
        self.fine_iterations = kf_opt
        logging.info(f"Auto-set iterations: coarse={self.coarse_iterations}, fine={self.fine_iterations} (Nc={self.N_c})")

        # Calculate optimal Grover iterations
        theta_c = math.asin(math.sqrt(self.N_c / self.N))
        self.coarse_iterations = max(1, round(
            (math.pi / (4 * theta_c)) - 0.5)) if coarse_iterations is None else coarse_iterations
        theta_f = math.asin(math.sqrt(1.0 / self.N_c))
        self.fine_iterations = max(1, round(
            (math.pi / (4 * theta_f)) - 0.5)) if fine_iterations is None else fine_iterations

        self.qr = QuantumRegister(n_qubits, 'q')
        self.ar = QuantumRegister(1, 'anc')
        self.cr = ClassicalRegister(n_qubits, 'c')
        self.qc = QuantumCircuit(self.qr, self.ar, self.cr)
        self.coarse_states = []
        self.fine_oracle_states = []
        logging.debug(
            f"SEGC init: n_qubits={n_qubits}, coarse_mask={coarse_mask}, target={target}, k_c={self.coarse_iterations}, k_f={self.fine_iterations}")

    def initialize(self):
        """Initialize uniform superposition over all qubits except ancilla."""
        for i in range(self.n_qubits):
            self.qc.h(self.qr[i])
        self.qc.x(self.ar)
        self.qc.h(self.ar)
        logging.debug("Initialized uniform superposition")

    def apply_coarse_oracle(self):
        """Apply coarse oracle for states matching coarse_mask."""
        for i, bit in enumerate(self.coarse_mask[::-1]):
            if bit != '*':
                if bit == '0':
                    self.qc.x(self.qr[i])
                self.qc.mcx([self.qr[j]
                            for j in range(self.n_qubits)], self.ar)
                if bit == '0':
                    self.qc.x(self.qr[i])
        logging.debug("Applied coarse oracle")

    def coarse_diffuser(self):
        """Apply diffuser for coarse phase."""
        for i in range(self.n_qubits):
            self.qc.h(self.qr[i])
            self.qc.x(self.qr[i])
        self.qc.h(self.qr[self.n_qubits-1])
        self.qc.mcx([self.qr[i] for i in range(self.n_qubits-1)],
                    self.qr[self.n_qubits-1])
        self.qc.h(self.qr[self.n_qubits-1])
        for i in range(self.n_qubits):
            self.qc.x(self.qr[i])
            self.qc.h(self.qr[i])
        logging.debug("Applied coarse diffuser")

    def run_coarse(self):
        """Run coarse phase with optimal iterations."""
        for _ in range(self.coarse_iterations):
            self.apply_coarse_oracle()
            self.coarse_diffuser()
        logging.info(f"Completed {self.coarse_iterations} coarse iterations")

    def get_coarse_states(self):
        """Get states matching coarse mask in standard ordering."""
        statevector = Statevector(self.qc)
        probs = statevector.probabilities_dict()
        coarse_states = []
        for state, prob in probs.items():
            if prob > 1e-10 and all(state[i] == self.coarse_mask[::-1][i] for i in range(self.n_qubits) if self.coarse_mask[::-1][i] != '*'):
                coarse_states.append(state[:self.n_qubits])
        self.coarse_states = coarse_states
        logging.debug(f"Coarse states: {coarse_states[:5]}...")
        return coarse_states

    def extract_features(self, state):
        """Extract binary features from state."""
        return [int(bit) for bit in state]

    def apply_fine_oracle(self, marked_states):
        """Apply fine oracle for marked states."""
        self.fine_oracle_states = marked_states
        for state in marked_states:
            for i, bit in enumerate(state[::-1]):
                if bit == '0':
                    self.qc.x(self.qr[i])
            self.qc.mcx([self.qr[i] for i in range(self.n_qubits)], self.ar)
            for i, bit in enumerate(state[::-1]):
                if bit == '0':
                    self.qc.x(self.qr[i])
        logging.debug(f"Applied fine oracle for states: {marked_states}")

    def partial_diffuser(self, qc, qr, ar):
        """Apply partial diffuser over coarse subspace."""
        coarse_indices = [i for i in range(
            self.n_qubits) if self.coarse_mask[i] == '*']
        logging.debug(f"Coarse indices for diffuser: {coarse_indices}")
        if not coarse_indices:
            logging.warning("No coarse indices found, skipping diffuser")
            return
        for i in coarse_indices:
            if i >= len(qr):
                logging.error(f"Invalid qubit index {i} for qr size {len(qr)}")
                raise IndexError(f"Qubit index {i} out of range")
            qc.h(qr[i])
            qc.x(qr[i])
        target_idx = coarse_indices[-1]
        if len(coarse_indices) > 1:
            qc.h(qr[target_idx])
            qc.mcx([qr[i] for i in coarse_indices[:-1]], qr[target_idx])
            qc.h(qr[target_idx])
        else:
            qc.h(qr[target_idx])
            qc.z(qr[target_idx])
            qc.h(qr[target_idx])
        for i in coarse_indices:
            qc.x(qr[i])
            qc.h(qr[i])
        logging.debug("Applied partial diffuser")

    def run_fine_iteration(self):
        """Run fine phase with optimal iterations."""
        logging.debug(f"Starting {self.fine_iterations} fine iterations")
        for i in range(self.fine_iterations):
            logging.debug(f"Fine iteration {i+1}/{self.fine_iterations}")
            self.apply_fine_oracle(self.fine_oracle_states)
            self.partial_diffuser(self.qc, self.qr, self.ar)
        logging.info(f"Completed {self.fine_iterations} fine iterations")

    def measure(self, shots=8192):
        """Measure the circuit."""
        measure_qc = self.qc.copy()
        measure_qc.measure(self.qr, self.cr)
        from qiskit_aer import AerSimulator
        simulator = AerSimulator()
        job = simulator.run(measure_qc, shots=shots)
        counts = job.result().get_counts()
        logging.debug(
            f"Measurement counts (top 5): {sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]}")
        return counts
