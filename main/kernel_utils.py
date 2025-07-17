"""
Created on 17/07/2025

@author: Aryan

Filename: kernel_utils.py

Relative Path: dynamic-oseq-qsvm/kernel_utils.py
"""

import numpy as np


def compute_simplified_kernel(X1, X2=None):
    if X2 is None:
        X2 = X1
    gamma = 1.0 / X1.shape[1]
    X1_sq = np.sum(X1**2, axis=1).reshape(-1, 1)
    X2_sq = np.sum(X2**2, axis=1).reshape(1, -1)
    distances = X1_sq + X2_sq - 2 * np.dot(X1, X2.T)
    return np.exp(-gamma * distances)
