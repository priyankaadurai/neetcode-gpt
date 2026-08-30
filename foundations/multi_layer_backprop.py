import numpy as np
from typing import List


class Solution:
    def forward_and_backward(self,
                              x: List[float],
                              W1: List[List[float]], b1: List[float],
                              W2: List[List[float]], b2: List[float],
                              y_true: List[float]) -> dict:
        # Architecture: x -> Linear(W1, b1) -> ReLU -> Linear(W2, b2) -> predictions
        # Loss: MSE = mean((predictions - y_true)^2)
        #
        # Return dict with keys:
        #   'loss':  float (MSE loss, rounded to 4 decimals)
        #   'dW1':   2D list (gradient w.r.t. W1, rounded to 4 decimals)
        #   'db1':   1D list (gradient w.r.t. b1, rounded to 4 decimals)
        #   'dW2':   2D list (gradient w.r.t. W2, rounded to 4 decimals)
        #   'db2':   1D list (gradient w.r.t. b2, rounded to 4 decimals)
        x = np.array(x)              
        W1 = np.array(W1)            
        b1 = np.array(b1)            
        W2 = np.array(W2)            
        b2 = np.array(b2)            
        y_true = np.array(y_true)     
        z1 = W1 @ x + b1
        a1 = np.maximum(z1,0)
        z2 = W2 @ a1 + b2
        y_hat = z2
        n = y_true.size
        loss = np.round((np.sum(((y_hat-y_true)**2)))/n, 4)
        dZ2 = 2/n*(y_hat-y_true)
        dW2 = np.round(np.outer(dZ2, a1), 4)
        db2 = np.round(dZ2, 4)
        dA1 = dZ2 @ W2
        mask = (z1 > 0)
        dZ1 = dA1 * mask
        dW1 = np.round(np.outer(dZ1,x), 4)
        db1 = np.round(dZ1, 4)

        return {
            'loss': float(loss),
            'dW1': dW1.tolist(),
            'db1': db1.tolist(),
            'dW2': dW2.tolist(),
            'db2': db2.tolist(),
}




        
        
