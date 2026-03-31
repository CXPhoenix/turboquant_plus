"""Numpy-compatible wrappers for RotorQuantMSE and IsoQuantMSE.

These match TurboQuantMSE's interface:
    q = RotorQuantMSENp(d, bit_width, seed)
    indices, norms = q.quantize(x_np)     # x_np: (batch, d) or (d,)
    x_hat_np = q.dequantize(indices, norms)

Internally delegates to the torch-based RotorQuant/IsoQuant implementations.
"""

import numpy as np
import torch
from turboquant.rotorquant import RotorQuantMSE
from turboquant.isoquant import IsoQuantMSE


class RotorQuantMSENp:
    """Numpy wrapper around RotorQuantMSE (Clifford Cl(3,0) rotors)."""

    def __init__(self, d: int, bit_width: int, seed: int = 42):
        self.d = d
        self.bit_width = bit_width
        self._q = RotorQuantMSE(d=d, bits=bit_width, seed=seed, device="cpu")

    def quantize(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Quantize and return (indices_np, norms_np) for storage."""
        x_t = torch.from_numpy(x).float()
        x_hat, indices_dict = self._q(x_t)
        # Pack the dict into something we can pass back through dequantize
        # Store the raw dict on the instance keyed by id — hacky but compatible
        token = id(indices_dict)
        if not hasattr(self, '_cache'):
            self._cache = {}
        self._cache[token] = indices_dict
        norms = indices_dict['_norms'].numpy()
        return token, norms

    def dequantize(self, token, norms: np.ndarray) -> np.ndarray:
        """Reconstruct from cached indices dict."""
        indices_dict = self._cache.pop(token)
        x_hat = self._q.dequantize(indices_dict)
        return x_hat.numpy()


class IsoQuantMSENp:
    """Numpy wrapper around IsoQuantMSE (quaternion SO(4) rotations).

    Supports both 'fast' (3 DOF) and 'full' (6 DOF) modes.
    """

    def __init__(self, d: int, bit_width: int, seed: int = 42,
                 mode: str = 'full'):
        self.d = d
        self.bit_width = bit_width
        self.mode = mode
        self._q = IsoQuantMSE(d=d, bits=bit_width, seed=seed,
                               mode=mode, device="cpu")

    def quantize(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_t = torch.from_numpy(x).float()
        x_hat, indices_dict = self._q(x_t)
        token = id(indices_dict)
        if not hasattr(self, '_cache'):
            self._cache = {}
        self._cache[token] = indices_dict
        norms = indices_dict['_norms'].numpy()
        return token, norms

    def dequantize(self, token, norms: np.ndarray) -> np.ndarray:
        indices_dict = self._cache.pop(token)
        x_hat = self._q.dequantize(indices_dict)
        return x_hat.numpy()
