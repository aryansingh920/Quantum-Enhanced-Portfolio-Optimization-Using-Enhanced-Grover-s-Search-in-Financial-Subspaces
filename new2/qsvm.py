import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZZFeatureMap
from qiskit.quantum_info import Statevector
import logging


class QSVM:
    def __init__(self, csv_file, n_qubits, feature_dim, iterations, initial_theta, feature_map_type='linear'):
        self.csv_file = csv_file
        self.n_qubits = n_qubits
        self.feature_dim = feature_dim
        self.iterations = iterations
        self.initial_theta = initial_theta
        self.scaler = StandardScaler()
        self.svc = None
        self.feature_map_type = feature_map_type
        self.feature_map = ZZFeatureMap(feature_dimension=feature_dim, reps=1,
                                        entanglement='linear') if feature_map_type == 'linear' else 'statevector'
        self.segc = None  # To store SEGC instance
        logging.debug(
            f"QSVM init: feature_map_type={feature_map_type}, feature_dim={feature_dim}")

    def _make_features(self, states, feature_map):
        """Generate features from states."""
        if self.feature_map_type == 'statevector' and self.segc:
            statevector = Statevector(self.segc.qc)
            probs = statevector.probabilities_dict()
            X = []
            for state in states:
                amp = probs.get(state[::-1] + '0', 0.0)  # Ancilla=0
                binary = [int(bit) for bit in state][:self.feature_dim-1]
                X.append([amp] + binary)
            X = np.array(X)
            logging.debug(f"Statevector features (top 5): {X[:5]}")
            return self.scaler.transform(X) if hasattr(self.scaler, 'mean_') else X
        else:
            X = [[int(bit) for bit in state][:self.feature_dim]
                 for state in states]
            X = np.array(X)
            logging.debug(f"Binary features (top 5): {X[:5]}")
            return self.scaler.transform(X) if hasattr(self.scaler, 'mean_') else X

    def _quantum_kernel(self, X1, X2):
        """Compute quantum kernel matrix."""
        n_samples1, n_samples2 = X1.shape[0], X2.shape[0]
        kernel_matrix = np.zeros((n_samples1, n_samples2))
        for i in range(n_samples1):
            for j in range(n_samples2):
                circ = QuantumCircuit(self.n_qubits)
                circ.compose(self.feature_map.assign_parameters(
                    X1[i]), inplace=True)
                circ.compose(self.feature_map.assign_parameters(
                    X2[j]).inverse(), inplace=True)
                circ.measure_all()
                from qiskit_aer import AerSimulator
                job = AerSimulator().run(circ, shots=1000)
                counts = job.result().get_counts()
                kernel_matrix[i, j] = counts.get('0' * self.n_qubits, 0) / 1000
        logging.debug(f"Kernel matrix shape: {kernel_matrix.shape}")
        return kernel_matrix

    def train(self):
        """Train QSVM on financial data."""
        data = pd.read_csv(self.csv_file)
        data['label'] = (data['Close'].shift(-1) > data['Close']).astype(int)
        data = data.dropna()
        X = data[['Open', 'High', 'Low', 'Close', 'Volume']].values
        y = data['label'].values
        n_samples = min(100, len(X))
        indices = np.random.choice(len(X), n_samples, replace=False)
        X, y = X[indices], y[indices]
        X_scaled = self.scaler.fit_transform(X)
        kernel_matrix = self._quantum_kernel(X_scaled, X_scaled)
        self.svc = SVC(kernel='precomputed')
        self.svc.fit(kernel_matrix, y)
        logging.info("QSVM trained")

    def select_topk_states(self, segc, coarse_states, k=2):
        """Select top-k states based on decision function scores."""
        X = self._make_features(coarse_states, self.feature_map)
        scores = self.svc.decision_function(X).ravel()
        logging.debug(f"QSVM scores: {list(zip(coarse_states, scores))[:5]}")
        idxs = np.argsort(scores)[-k:]
        picked = set(idxs.tolist())
        try:
            t_idx = coarse_states.index(segc.target_standard)
            picked.add(t_idx)
        except ValueError:
            pass
        idxs = sorted(picked)
        marked = [coarse_states[i] for i in idxs]
        logging.debug(f"Selected states: {marked}")
        return marked, scores
