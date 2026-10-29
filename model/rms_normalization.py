import numpy as np
from typing import List


class Solution:
    def rms_norm(self, x: List[float], gamma: List[float], eps: float) -> List[float]:
        # Implement RMS Normalization (similar to LayerNorm but without mean centering or beta)
        x = np.array(x)
        gamma = np.array(gamma)
        n = x.size
        rms_x = np.sqrt(np.mean(x ** 2) + eps)
        # Normalize x, then scale by gamma
        x_hat = x/rms_x
        out = gamma * x_hat

        # Return result rounded to 4 decimal places as a list
        return np.round(out,4)
