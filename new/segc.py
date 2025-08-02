import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
from qiskit.quantum_info import Statevector
from qiskit_aer import StatevectorSimulator


class SEGCAlgorithm:
    def __init__(self):
        self.n = 7
        self.N = 2 ** self.n
        self.target = '0000101'  # Qiskit
        self.target_standard = '1010000'  # Standard
        self.N_c = 8
        self.t = 1
        self.coarse_iterations = 4
        self.fine_iterations = 2
        self.theta_coarse = np.arcsin(np.sqrt(self.N_c / self.N))
        self.theta_fine = np.arcsin(np.sqrt(self.t / self.N_c))
        print(
            f"SEGC Configuration:\nTarget (Qiskit): {self.target}, (Standard): {self.target_standard}\nN = {self.N}, N_c = {self.N_c}")
        print(
            f"Coarse angle: {self.theta_coarse:.4f} rad\nFine angle: {self.theta_fine:.4f} rad")
        print(
            f"Coarse iterations: {self.coarse_iterations}\nFine iterations: {self.fine_iterations}")
        print(
            f"Expected coarse subspace prob: {np.sin((2 * self.coarse_iterations + 1) * self.theta_coarse)**2:.4f}")

    def coarse_oracle(self, qc, qubits, ancilla):
        for i in range(4):
            qc.x(qubits[i])
        qc.h(qubits[4])
        # Use q4 as target for phase flip
        qc.mcx(qubits[0:4], qubits[4], mode='noancilla')
        qc.h(qubits[4])
        for i in range(4):
            qc.x(qubits[i])
        print("Debug: Coarse oracle applied")

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
        for i in range(4):
            qc.x(qubits[i])
        qc.mcx(qubits[0:4], ancilla[0], mode='noancilla')
        qc.x(qubits[5])
        qc.h(ancilla[0])
        qc.mcx([qubits[4], qubits[5], qubits[6]], ancilla[0], mode='noancilla')
        qc.h(ancilla[0])
        qc.x(qubits[5])
        qc.mcx(qubits[0:4], ancilla[0], mode='noancilla')
        for i in range(4):
            qc.x(qubits[i])
        print("Debug: Fine oracle applied")

    def partial_diffuser(self, qc, qubits, ancilla):
        for i in range(4):
            qc.x(qubits[i])
        qc.mcx(qubits[0:4], ancilla[0], mode='noancilla')
        fine_qubits = qubits[4:7]
        qc.h(fine_qubits)
        qc.x(fine_qubits)
        qc.h(fine_qubits[-1])
        qc.mcx(fine_qubits[:-1], fine_qubits[-1], mode='noancilla')
        qc.h(fine_qubits[-1])
        qc.x(fine_qubits)
        qc.h(fine_qubits)
        qc.mcx(qubits[0:4], ancilla[0], mode='noancilla')
        for i in range(4):
            qc.x(qubits[i])
        print("Debug: Partial diffuser applied")

    def debug_oracle(self, oracle_type='coarse'):
        qr = QuantumRegister(self.n, 'q')
        ar = QuantumRegister(1, 'anc')
        qc = QuantumCircuit(qr, ar)
        if oracle_type == 'coarse':
            qc.h(qr[4:7])
            self.coarse_oracle(qc, qr, ar)
        else:
            qc.x(qr[4])
            qc.x(qr[6])
            self.fine_oracle(qc, qr, ar)
        state = Statevector.from_instruction(qc)
        target_idx = int(self.target, 2)
        coarse_prob = sum(abs(state.data[i])**2 for i in range(8))
        print(f"Debug {oracle_type} oracle:")
        print(f"  Target amplitude = {state.data[target_idx]:.4f}")
        print(f"  Coarse subspace prob = {coarse_prob:.4f}")
        if oracle_type == 'coarse':
            print("  Coarse-valid states (Qiskit ordering):")
            for i in range(8):
                state_str = format(i, '07b')
                amp = state.data[i]
                mark = '← TARGET' if state_str == self.target else ''
                print(
                    f"    {state_str}: {amp:.4f} (prob: {abs(amp)**2:.4f}) {mark}")

    def create_circuit(self):
        qr = QuantumRegister(self.n, 'q')
        ar = QuantumRegister(1, 'anc')
        cr = ClassicalRegister(self.n, 'c')
        qc = QuantumCircuit(qr, ar, cr)
        qc.h(qr)
        print("Debug: Initialized superposition")
        print("Executing coarse iterations...")
        for i in range(self.coarse_iterations):
            print(f"  Coarse iteration {i+1}/{self.coarse_iterations}")
            self.coarse_oracle(qc, qr, ar)
            self.coarse_diffuser(qc, qr)
            state = Statevector.from_instruction(qc)
            self.analyze_state(state, f"After coarse iteration {i+1}")
        print("Executing fine iterations...")
        for i in range(self.fine_iterations):
            print(f"  Fine iteration {i+1}/{self.fine_iterations}")
            self.fine_oracle(qc, qr, ar)
            state = Statevector.from_instruction(qc)
            self.analyze_state(state, f"After fine oracle {i+1}")
            self.partial_diffuser(qc, qr, ar)
            state = Statevector.from_instruction(qc)
            self.analyze_state(state, f"After partial diffusion {i+1}")
        final_state = Statevector.from_instruction(qc)
        qc.measure(qr, cr)
        return qc, final_state

    def analyze_state(self, state, step_name):
        print(f"{step_name}:")
        target_idx = int(self.target, 2)
        target_amp = state.data[target_idx]
        print(
            f"  Target {self.target} (Qiskit) / {self.target_standard} (Standard): {target_amp:.4f} (prob: {abs(target_amp)**2:.4f})")
        coarse_prob = sum(abs(state.data[i])**2 for i in range(8))
        print(f"  Coarse subspace prob: {coarse_prob:.4f}")
        print("  Coarse-valid states (Qiskit ordering):")
        for i in range(8):
            state_str = format(i, '07b')
            amp = state.data[i]
            mark = '← TARGET' if state_str == self.target else ''
            print(f"    {state_str}: {amp:.4f} (prob: {abs(amp)**2:.4f}) {mark}")

    def run_simulation(self, qc, final_state):
        simulator = StatevectorSimulator()
        job = simulator.run(qc, shots=8192)
        result = job.result()
        counts = result.get_counts()
        print("==================================================")
        print("FINAL RESULTS")
        print("==================================================")
        total_shots = sum(counts.values())
        target_prob = counts.get(self.target_standard, 0) / total_shots
        coarse_prob = sum(counts.get(
            format(i, '07b')[::-1], 0) / total_shots for i in range(8))
        print(
            f"Target state probability (Standard {self.target_standard}): {target_prob:.4f}")
        print(f"Coarse subspace probability: {coarse_prob:.4f}")
        print(
            f"Theoretical target prob: {abs(final_state.data[int(self.target, 2)])**2:.4f}")
        print("Top 10 measured states (Standard ordering):")
        sorted_counts = sorted(
            counts.items(), key=lambda x: x[1], reverse=True)[:10]
        for state, count in sorted_counts:
            prob = count / total_shots
            state_standard = state[::-1]
            coarse_valid = 'coarse-valid' if int(
                state_standard, 2) < 8 else 'coarse-invalid'
            mark = '← TARGET' if state == self.target_standard else ''
            print(f"  {state_standard}: {prob:.4f} [{coarse_valid}] {mark}")


def main():
    segc = SEGCAlgorithm()
    print("\nDebugging oracles...")
    segc.debug_oracle('coarse')
    segc.debug_oracle('fine')
    qc, final_state = segc.create_circuit()
    segc.run_simulation(qc, final_state)


if __name__ == "__main__":
    main()
