import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        if len(weights) == 1:
            return np.round(x @ weights[0] + biases[0],5)
        if len(biases) >= len(weights):
            layers = len(weights)
        else:
            layers = len(biases)
        for i in range(layers-1):
            zi = x.T @ weights[i] + biases[i]
            hi = np.maximum(zi, 0)
            x = hi
        return np.round(hi @ weights[layers-1] + biases[layers-1], 5)
        


        
        
