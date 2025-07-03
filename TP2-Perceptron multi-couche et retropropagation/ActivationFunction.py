import numpy as np
import matplotlib.pyplot as plt

class ActivationFunction:
    def __init__(self, name, alpha=0.01):
        self.name = name.lower()
        self.alpha = alpha  # Pour Leaky ReLU

    def apply(self, z):
        if self.name == "heaviside":
            return np.where(np.greater(z,0),1,0)
        elif self.name == "sigmoid":
            return (np.divide(1,(1+ np.exp(-z))))
        elif self.name == "tanh":
            return np.divide((np.exp(z)-np.exp(-z)),(np.exp(z)+np.exp(-z)))
        elif self.name == "relu":
            return np.where(z>0, z, 0)
        elif self.name == "leaky_relu":
            return np.where(z>0, z, self.alpha*z)
        #elif self.name == "perceptron":
        #    test_perceptron() 
        else:
            raise ValueError(f"Activation '{self.name}' non reconnue.")

    def derivative(self, z):
        if self.name == "heaviside":
            # La dérivée de Heaviside est la distribution de Dirac
            return np.where(np.equal(z,0))
        elif self.name == "sigmoid":
            return np.divide(1,(1+np.exp(z))^2)
        elif self.name == "tanh":
            return np.divide((np.exp(z)-np.exp(z))**2,(np.exp(z) + np.exp(z))**2)
        elif self.name == "relu":
            return np.where(z>0, 1, 0)
        elif self.name == "leaky_relu":
            return np.where(z>0, 1, self.alpha)
        else:
            raise ValueError(f"Dérivée de '{self.name}' non définie.")
        