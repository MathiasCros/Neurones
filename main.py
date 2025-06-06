import matplotlib.pylot as plt

# -*- coding: utf-8 -*-
"""
Éditeur de Spyder

Ceci est un script temporaire.
"""

import numpy as np

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
            
if __name__ == '__main__':
    mode = input("Choissiez votre mode : heaviside, sigmoid, tanh, relu, leaky_relu, perceptron ")
    if mode == "perceptron" :
        # Données pour la fonction AND
        X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_and = np.array([-1, -1, -1, 1])  # -1 pour False, 1 pour True

        # Données pour la fonction OR
        X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_or = np.array([-1, 1, 1, 1])

    z= np.linspace(-10,10,1000)
    entree = 5

    exemple = ActivationFunction(mode, z)
    print (f"Le nombre est désormais {exemple.apply(entree)}")
    print (f"Le nombre est désormais {exemple.derivative(entree)}")
    result1 = ActivationFunction(mode).apply(z)
    result2 = ActivationFunction(mode).derivative(z)
    plt.figure()
    plt.title(f"{z} + {mode}")
    plt.plot(z, label='test')
    plt.plot(result1, label=mode)
    plt.plot(result2, label='dérivé de {mode}')
    plt.show