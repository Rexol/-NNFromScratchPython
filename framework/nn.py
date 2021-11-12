"""
File:    nn.py 
  * 
  * Author:   Vladimir Surtaev
  * Date:     November 2021
  * 
  * Summary of File: 
  * 
  *   This file contains basic parts of neural network
  *     Linear layer
  *     Activation functions
  *     Regularization layers
  *     Criterions
"""

# NymPy only.
import numpy as np


"""
Parent class for all layers
"""
class Module():
    def __init__(self):
        self._train = True
    
    def forward(self, input):
        raise NotImplementedError

    def backward(self,input, grad_output):
        raise NotImplementedError
    
    def parameters(self):
        '''Returns list of it's parameters'''
        return []
    
    def grad_parameters(self):
        '''returns list of tensors-gradients for it's parameters'''
        return []
    
    def train(self):
        self._train = True
    
    def eval(self):
        self._train = False

"""
Sequential takes list of modules and execute them sequentially
"""
class Sequential(Module):
    def __init__ (self, *layers):
        super().__init__()
        self.layers = layers

    def forward(self, input):
        """
        Passes input through all layers
        """

        for layer in self.layers:
            input = layer.forward(input)

        self.output = input
        return self.output

    def backward(self, input, grad_output):
        """        
        Calculates gradient for it's parameters and 
        passes gradient for layer's input an passes it to next layer
        """
        
        for i in range(len(self.layers)-1, 0, -1):
            grad_output = self.layers[i].backward(self.layers[i-1].output, grad_output)
        
        grad_input = self.layers[0].backward(input, grad_output)
        
        return grad_input
      
    def parameters(self):
        'Concatenates all parameters in one list'
        res = []
        for l in self.layers:
            res += l.parameters()
        return res
    
    def grad_parameters(self):
        'Concatenates all gradients in one list'
        res = []
        for l in self.layers:
            res += l.grad_parameters()
        return res
    
    def train(self):
        for layer in self.layers:
            layer.train()
    
    def eval(self):
        for layer in self.layers:
            layer.eval()


#---LAYERS
"""
Linear layer realization.
"""
class Linear(Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        
        # Xavier initialization
        stdv = 1./np.sqrt(dim_in)
        self.W = np.random.uniform(-stdv, stdv, size=(dim_in, dim_out))
        self.b = np.random.uniform(-stdv, stdv, size=dim_out)
        
    def forward(self, input):
        self.output = np.dot(input, self.W) + self.b
        return self.output
    
    def backward(self, input, grad_output):
        self.grad_b = np.mean(grad_output, axis=0)
        
        #     in_dim x batch_size
        self.grad_W = np.dot(input.T, grad_output)
        #                 batch_size x out_dim
        
        grad_input = np.dot(grad_output, self.W.T)
        
        return grad_input
    
    def parameters(self):
        return [self.W, self.b]
    
    def grad_parameters(self):
        return [self.grad_W, self.grad_b]


#----ACTIVATION-FUNCTIONS
"""
ReLU activation function.
"""
class ReLU(Module):
    def __init__(self):
         super().__init__()
    
    def forward(self, input):
        self.output = np.maximum(input, 0)
        return self.output
    
    def backward(self, input, grad_output):
        grad_input = np.multiply(grad_output, input > 0)
        return grad_input

"""
Leaky ReLU activation fundtion
"""
class LeakyReLU(Module):
    def __init__(self, slope=0.03):
        super().__init__()
            
        self.slope = slope
        
    def forward(self, input):
        self.output = np.multiply(input, input > 0) + np.multiply(self.slope*input, input < 0)
        return self.output
    
    def backward(self, input, grad_output):
        grad_input = np.multiply(grad_output, input > 0) + np.multiply(self.slope*grad_output, input < 0)
        return grad_input

"""
Sigmoid activation function
"""
class Sigmoid(Module):
    def __init__(self, slope=0.03):
        super().__init__()

    def forward(self, input):
        self.output = 1/(1+np.exp(-input))
        return self.output
    
    def backward(self, input, grad_output):
        grad_input = np.multiply(self.output*(1-self.output), grad_output)
        return grad_input

"""
SoftMax activation function
"""
class SoftMax(Module):
    def __init__(self):
         super().__init__()
    
    def forward(self, input: np.ndarray) -> np.ndarray: 
        # N.B. If inputs are large, then
        # exponents would be much larger
        self.output = np.subtract(input, input.max(axis=1, keepdims=True))
        self.output = np.exp(self.output)
        self.output = np.divide(self.output, self.output.sum(axis=1, keepdims=True))
        
        return self.output
    
    def backward(self, input, grad_output):
        jacobian = np.einsum('ij,jk->ijk', self.output, np.eye(self.output.shape[-1])) \
        - np.einsum('ij,ik->ijk', self.output, self.output)
        grad_input = np.einsum('ij, ijk->ik', grad_output, jacobian)
        return grad_input


#----REGULARIZATION
"""
Dopout regularization
"""
class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        
        self.p = p
        self.mask = None
        
    def forward(self, input):
        if self._train:
            self.mask = (np.random.random(input.size) < (1 - self.p)).reshape(input.shape)
            self.output = input * self.mask / (1 - self.p) # To save Expected value.
        else:
            self.output = input
        return self.output
    
    def backward(self, input, grad_output):
        if self._train:
            grad_input = np.multiply(grad_output, self.mask / (1 - self.p))
        else:
            grad_input = grad_output
        return grad_input

"""
BatchNorm regularization
"""
class BatchNorm(Module):
    def __init__(self, num_features, eps=1e-05, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        self.moving_mean = 0 # For forward pass
        self.moving_var = 0
    
    def forward(self, input):
        if self._train:
            sample_mean = input.mean(axis=0)
            sample_var = input.var(axis=0)

            self.moving_mean = sample_mean * self.momentum + self.moving_mean * (1 - self.momentum)
            self.moving_var = sample_var * self.momentum + self.moving_var * (1 - self.momentum)
            
            self.std = np.sqrt(sample_var + self.eps)
            self.centered = input - sample_mean
            self.normalized = self.centered / self.std
            self.output = self.gamma * self.normalized + self.beta
            
        else:
            input_norm = (input - self.moving_mean) / np.sqrt(self.moving_var + self.eps)
            self.output = self.gamma * input_norm + self.beta
            
        return self.output
    
    def backward(self, input, grad_output):
        if self._train:
            N = grad_output.shape[0]
            
            self.grad_gamma = (grad_output * self.normalized).sum(axis=0)
            self.grad_beta = grad_output.sum(axis=0)
            
            #t = 1./np.sqrt(sample_var + self.eps) # = 1/self.std
            grad_input = ((self.gamma /self.std) / N) * (N * grad_output - np.sum(grad_output, axis=0)
                    - np.power(self.std, -2) * self.centered * np.sum(grad_output * self.centered, axis=0))

            '''grad_input_norm = grad_output * self.gamma
            grad_input = 1/N / self.std * (N * grad_input_norm - 
                            grad_input_norm.sum(axis=0) - 
                            self.normalized * (grad_input_norm * self.normalized).sum(axis=0))'''
        else:
            grad_input = grad_output

        return grad_input

    def parameters(self):
        return [self.gamma, self.beta]
    
    def grad_parameters(self):
        return [self.grad_gamma, self.grad_beta]


#----CRITERIONS
"""
Parent class for criterions
"""
class Criterion():        
    def forward(self, input, target):
        raise NotImplementedError

    def backward(self, input, target):
        raise NotImplementedError

"""
MSE criterion
"""
class MSE(Criterion):
    def forward(self, input, target):
        self.output = np.sum(np.power(input - target, 2)) / input.shape[0]
        return self.output
 
    def backward(self, input, target):
        self.grad_output  = (input - target) * 2 / input.shape[0]
        return self.grad_output

"""
CrossEntropy criterion
"""
class CrossEntropy(Criterion):
    def __init__(self):
        super().__init__()
        
    def forward(self, input, target):       
        eps = 1e-9
        input_clamp = np.clip(input, eps, 1 - eps) # To avoid log(0):
        self.output = -1 * np.sum(np.log(input_clamp) * target) / input.shape[0]
        
        return self.output

    def backward(self, input, target):
        eps = 1e-9
        input_clamp = np.clip(input, eps, 1 - eps)
                
        grad_input = -1 * target / input_clamp
        return grad_input