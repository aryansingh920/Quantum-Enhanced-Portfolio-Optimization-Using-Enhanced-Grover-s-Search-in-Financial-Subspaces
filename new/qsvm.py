import numpy as np
import pandas as pd
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
from qiskit_aer import StatevectorSimulator
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


class QSVM:
    def __init__(self, csv_file, n_qubits=4, feature_dim=4, iterations=3, initial_theta=0.5, max_train_samples=100):
        """
        Initialize QSVM for integration with SEGC.
        
        Parameters:
        - csv_file: Path to the CSV file with financial data.
        - n_qubits: Number of qubits for the quantum kernel (default: 4 for a small example).
        - feature_dim: Number of features to use in the quantum kernel (must match n_qubits).
        - iterations: Number of feedback iterations to refine the threshold.
        - initial_theta: Initial threshold for marking states.
        - max_train_samples: Maximum number of samples to use for training (to reduce computation).
        """
        self.n_qubits = n_qubits
        self.feature_dim = feature_dim
        self.iterations = iterations
        self.theta = initial_theta
        self.max_train_samples = max_train_samples
        self.simulator = StatevectorSimulator()

        # Load and preprocess data
        self.data = pd.read_csv(csv_file)
        self.features = ['open_scaled', 'high_scaled', 'low_scaled', 'close_scaled',
                         'volume_scaled', 'ma5_scaled', 'ma20_scaled', 'rsi_scaled', 'volatility_scaled']
        self.X = self.data[self.features[:feature_dim]].values
        self.y = self.data['regime'].values

        # Ensure balanced training data
        unique_classes = np.unique(self.y)
        if len(unique_classes) < 2:
            raise ValueError(
                f"Dataset contains only {len(unique_classes)} class(es): {unique_classes}. At least 2 classes required.")

        # Sample balanced data
        samples_per_class = max_train_samples // 2
        X_balanced = []
        y_balanced = []
        for cls in unique_classes[:2]:
            cls_indices = np.where(self.y == cls)[0]
            if len(cls_indices) < samples_per_class:
                print(
                    f"Warning: Class {cls} has only {len(cls_indices)} samples, using all available")
                selected_indices = cls_indices
            else:
                selected_indices = np.random.choice(
                    cls_indices, samples_per_class, replace=False)
            X_balanced.append(self.X[selected_indices])
            y_balanced.append(self.y[selected_indices])

        self.X = np.vstack(X_balanced)
        self.y = np.hstack(y_balanced)
        print(
            f"Selected {len(self.X)} training samples ({samples_per_class} per class)")

        # Scale features
        self.scaler = StandardScaler()
        self.X_scaled = self.scaler.fit_transform(self.X)

        # Initialize quantum kernel
        self.feature_map = ZZFeatureMap(
            feature_dimension=feature_dim, reps=3, entanglement='linear')
        self.svm = None
        self.kernel_matrix = None

        print(
            f"QSVM Configuration:\nNumber of qubits: {self.n_qubits}\nFeature dimension: {self.feature_dim}")
        print(
            f"Initial threshold: {self.theta:.4f}\nFeedback iterations: {self.iterations}")
        print(f"Training samples: {len(self.X)}, Classes: {np.unique(self.y)}")

    def compute_quantum_kernel(self, X1, X2):
        """
        Compute the quantum kernel matrix using the feature map.
        
        Parameters:
        - X1, X2: Data matrices for kernel computation.
        
        Returns:
        - kernel_matrix: Quantum kernel matrix.
        """
        kernel_matrix = np.zeros((len(X1), len(X2)))
        total = len(X1) * len(X2)
        count = 0
        for i, x1 in enumerate(X1):
            for j, x2 in enumerate(X2):
                # Create quantum circuit for kernel computation
                qc = QuantumCircuit(self.n_qubits)
                # Apply feature map for x1
                fm1 = self.feature_map.assign_parameters(x1)
                qc.compose(fm1, inplace=True)
                # Apply inverse feature map for x2
                fm2 = self.feature_map.assign_parameters(x2).inverse()
                qc.compose(fm2, inplace=True)
                # Simulate to get overlap
                state = Statevector.from_instruction(qc)
                kernel_matrix[i, j] = abs(state.data[0])**2
                count += 1
                if count % 100 == 0:
                    print(
                        f"Computed {count}/{total} kernel entries ({count/total*100:.1f}%)")
        return kernel_matrix

    def train(self):
        """
        Train the QSVM using a quantum kernel and classical SVM.
        """
        print("Training QSVM...")
        self.kernel_matrix = self.compute_quantum_kernel(
            self.X_scaled, self.X_scaled)
        self.svm = SVC(kernel='precomputed')
        self.svm.fit(self.kernel_matrix, self.y)
        print("QSVM training completed.")

    def compute_scores(self, state_indices, X_test):
        """
        Compute QSVM scores for given state indices.
        
        Parameters:
        - state_indices: List of indices representing quantum states.
        - X_test: Test data corresponding to state indices.
        
        Returns:
        - scores: QSVM decision function scores.
        """
        if self.svm is None:
            raise ValueError("QSVM must be trained before computing scores.")
        kernel_matrix = self.compute_quantum_kernel(X_test, self.X_scaled)
        scores = self.svm.decision_function(kernel_matrix)
        return scores

    def dynamic_oracle(self, qc, qubits, ancilla, state_indices, X_test):
        scores = self.compute_scores(state_indices, X_test)
        marked_count = 0
        print("Debug: QSVM scores for states:")
        for idx, score in zip(state_indices, scores):
            print(
                f"  State {format(idx, f'0{self.n_qubits}b')}: score = {score:.4f}")
            if score > self.theta:
                binary = format(idx, f'0{self.n_qubits}b')
                for i, bit in enumerate(binary):
                    if bit == '0':
                        qc.x(qubits[i])
                qc.mcx(qubits, ancilla[0], mode='noancilla')
                qc.z(ancilla[0])
                qc.mcx(qubits, ancilla[0], mode='noancilla')
                for i, bit in enumerate(binary):
                    if bit == '0':
                        qc.x(qubits[i])
                marked_count += 1
        print(
            f"Debug: Dynamic QSVM oracle applied with threshold {self.theta:.4f}, marked {marked_count} states")
    def feedback_loop(self, segc, qc, qubits, ancilla, state_indices, X_test):
        """
        Implement quantum-classical feedback loop to refine the threshold.
        
        Parameters:
        - segc: SEGCAlgorithm instance.
        - qc: Quantum circuit.
        - qubits: Quantum register.
        - ancilla: Ancilla register.
        - state_indices: Indices of states in the coarse subspace.
        - X_test: Test data for QSVM scoring.
        
        Returns:
        - final_state: Final quantum state after feedback iterations.
        """
        for i in range(self.iterations):
            print(f"Feedback iteration {i+1}/{self.iterations}")
            # Apply dynamic oracle
            self.dynamic_oracle(qc, qubits, ancilla, state_indices, X_test)
            # Apply partial diffuser from SEGC
            segc.partial_diffuser(qc, qubits, ancilla)
            # Simulate and analyze state
            state = Statevector.from_instruction(qc)
            segc.analyze_state(state, f"After feedback iteration {i+1}")
            # Compute target probability
            target_idx_0 = int('0' + segc.target, 2)
            target_idx_1 = int('1' + segc.target, 2)
            target_prob = abs(
                state.data[target_idx_0])**2 + abs(state.data[target_idx_1])**2
            print(f"Debug: Target probability = {target_prob:.4f}")
            # Adjust threshold towards target prob 0.7
            self.theta = min(1.0, self.theta + 0.1 * (0.7 - target_prob))
            print(f"Updated threshold: {self.theta:.4f}")
        return state
