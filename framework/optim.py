"""
File:    optim.py 
  * 
  * Author:   Vladimir Surtaev
  * Date:     November 2021
  * 
  * Summary of File: 
  * 
  *   This file contains optimizers for neural network.
  *   (SGD at the moment)
"""

"""
Stochastic gradient descent optimizer
"""
def SGD(params, gradients, lr=1e-3):    
    for weights, gradient in zip(params, gradients):
        weights -= lr * gradient