import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit_aer import StatevectorSimulator


class SEGCAlgorithm:
    def __init__(self, n_qubits, coarse_mask, target, coarse_iterations):
        self.n = n_qubits
        self.N = 2 ** self.n
        self.target = target[::-1]  # Convert standard to Qiskit ordering
        self.target_standard = target
        self.coarse_mask = coarse_mask
        self.coarse_iterations = coarse_iterations
        self.fine_iterations = 1
        self.N_c = 2 ** coarse_mask.count('*')
        self.t = 1
        self.theta_coarse = np.arcsin(np.sqrt(self.N_c / self.N))
        self.theta_fine = np.arcsin(np.sqrt(self.t / self.N_c))
        self.simulator = StatevectorSimulator()
        self.qc = None
        self.qr = None
        self.ar = None
        self.cr = None
        self.marked_states = [self.target]

        print(
            f"SEGC Configuration:\nTarget (Qiskit): {self.target}, (Standard): {self.target_standard}")
        print(f"N = {self.N}, N_c = {self.N_c}")
        print(
            f"Coarse angle: {self.theta_coarse:.4f} rad\nFine angle: {self.theta_fine:.4f} rad")
        print(
            f"Coarse iterations: {self.coarse_iterations}\nFine iterations: {self.fine_iterations}")
        print(
            f"Expected coarse subspace prob: {np.sin((2 * self.coarse_iterations + 1) * self.theta_coarse)**2:.4f}")

    def initialize(self):
        self.qr = QuantumRegister(self.n, 'q')
        self.ar = QuantumRegister(1, 'anc')
        self.cr = ClassicalRegister(self.n, 'c')
        self.qc = QuantumCircuit(self.qr, self.ar, self.cr)
        self.qc.h(self.qr)
        print("Debug: Initialized superposition")

    def coarse_oracle(self, qc, qubits, ancilla):
        coarse_indices = [i for i, c in enumerate(
            self.coarse_mask[::-1]) if c == '0']
        if coarse_indices:
            qc.x(qubits[coarse_indices])
            qc.mcx(qubits[coarse_indices], ancilla[0], mode='noancilla')
            qc.z(ancilla[0])
            qc.mcx(qubits[coarse_indices], ancilla[0], mode='noancilla')
            qc.x(qubits[coarse_indices])
        print(f"Debug: Coarse oracle applied for {self.coarse_mask}")

    def coarse_diffuser(self, qc, qubits):
        qc.h(qubits)
        qc.x(qubits)
        qc.h(qubits[-1])
        qc.mcx(qubits[:-1], qubits[-1], mode='noancilla')
        qc.h(qubits[-1])
        qc.x(qubits)
        qc.h(qubits)
        print("Debug: Coarse diffuser applied")

    def fine_oracle(self, qc, qubits, ancilla):
        for state in self.marked_states:
            binary = state  # Already in Qiskit ordering
            for i, bit in enumerate(binary):
                if bit == '0':
                    qc.x(qubits[i])
            qc.mcx(qubits, ancilla[0], mode='noancilla')
            qc.z(ancilla[0])
            qc.mcx(qubits, ancilla[0], mode='noancilla')
            for i, bit in enumerate(binary):
                if bit == '0':
                    qc.x(qubits[i])
        print(
            f"Debug: Fine oracle applied, marked {len(self.marked_states)} states")

    def partial_diffuser(self, qc, qubits, ancilla):
        coarse_indices = [i for i, c in enumerate(
            self.coarse_mask[::-1]) if c == '0']
        fine_indices = [i for i in range(self.n) if i not in coarse_indices]
        coarse_qubits = [qubits[i] for i in coarse_indices]
        fine_qubits = [qubits[i] for i in fine_indices]

        if coarse_qubits:
            for q in coarse_qubits:
                qc.x(q)
            qc.mcx(coarse_qubits, ancilla[0], mode='noancilla')
        qc.h(fine_qubits)
        qc.x(fine_qubits)
        qc.h(fine_qubits[-1])
        qc.mcx(fine_qubits[:-1], fine_qubits[-1], mode='noancilla')
        qc.h(fine_qubits[-1])
        qc.x(fine_qubits)
        qc.h(fine_qubits)
        if coarse_qubits:
            qc.mcx(coarse_qubits, ancilla[0], mode='noancilla')
            for q in coarse_qubits:
                qc.x(q)
        print("Debug: Enhanced partial diffuser applied")

    def run_coarse(self):
        for i in range(self.coarse_iterations):
            print(f"  Coarse iteration {i+1}/{self.coarse_iterations}")
            self.coarse_oracle(self.qc, self.qr, self.ar)
            self.coarse_diffuser(self.qc, self.qr)
            state = Statevector.from_instruction(self.qc)
            self.analyze_state(state, f"After coarse iteration {i+1}")

    def get_coarse_states(self):
        coarse_states = []
        for i in range(self.N):
            binary = format(i, f'0{self.n}b')  # Qiskit ordering
            if all(binary[j] == c for j, c in enumerate(self.coarse_mask[::-1]) if c != '*'):
                coarse_states.append(binary)  # Return in Qiskit ordering
        return coarse_states

    def extract_features(self, state):
        # Convert state (in standard ordering) to 7D features
        # Qiskit ordering
        return np.array([int(c) for c in state[::-1]], dtype=float)

    def update_fine_oracle(self, marked_states):
        self.marked_states = marked_states  # Already in Qiskit ordering
        print(
            f"Debug: Updated fine oracle with {len(self.marked_states)} marked states")

    def run_fine_iteration(self):
        print(f"  Fine iteration")
        for _ in range(self.fine_iterations):
            self.fine_oracle(self.qc, self.qr, self.ar)
            state = Statevector.from_instruction(self.qc)
            self.analyze_state(state, f"After fine oracle")
            self.partial_diffuser(self.qc, self.qr, self.ar)
            state = Statevector.from_instruction(self.qc)
            self.analyze_state(state, f"After partial diffusion")
        return state

    def measure(self, shots=8192):
        # Create a new circuit for measurement to avoid modifying self.qc
        measure_qc = self.qc.copy()
        measure_qc.measure(self.qr, self.cr)
        job = self.simulator.run(measure_qc, shots=shots)
        result = job.result()
        counts = result.get_counts()
        return counts

    def analyze_state(self, state, step_name):
        print(f"{step_name}:")
        target_idx_anc0 = int('0' + self.target, 2)
        target_idx_anc1 = int('1' + self.target, 2)
        target_prob = (abs(state.data[target_idx_anc0])
                       ** 2 + abs(state.data[target_idx_anc1])**2)
        print(
            f"  Target {self.target} (Qiskit) / {self.target_standard} (Standard): prob: {target_prob:.4f}")

        coarse_prob = 0
        coarse_states = self.get_coarse_states()
        for binary in coarse_states:
            for anc in ['0', '1']:
                idx = int(anc + binary, 2)
                coarse_prob += abs(state.data[idx])**2
        print(f"  Coarse subspace prob: {coarse_prob:.4f}")

        print("  Coarse-valid states (Standard ordering, ancilla traced out):")
        for binary in coarse_states:
            prob = sum(
                abs(state.data[int(anc + binary, 2)])**2 for anc in ['0', '1'])
            # Convert to standard ordering for display
            state_standard = binary[::-1]
            mark = '← TARGET' if state_standard == self.target_standard else ''
            print(f"    {state_standard}: prob: {prob:.4f} {mark}")

        return target_prob
