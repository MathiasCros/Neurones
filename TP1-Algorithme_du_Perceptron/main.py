import matplotlib.pyplot as plt
from perceptron_module import PerceptronSimple

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
        elif self.name == "perceptron":
            test_perceptron() 
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
        



def test_perceptron():
    # Données d'apprentissage AND
    X = np.array([[0, 0],[0, 1],[1, 0],[1, 1]])
    y = np.array([-1, -1, -1, 1])  # pour fonction AND

    
    model = PerceptronSimple(learning_rate=0.1)
    model.fit(X, y, max_epochs=20)

    predictions = model.predict(X)

    # Affichage des résultats
    print("Prédictions du perceptron :")
    for i, x in enumerate(X):
        print(f"Entrée : {x}, Prédit : {predictions[i]}, Réel : {y[i]}")

    # Score final
    print(f"Exactitude : {model.score(X, y) * 100:.2f}%")

            
if __name__ == '__main__':
    mode = input("Choissiez votre mode : heaviside, sigmoid, tanh, relu, leaky_relu, perceptron, perceptron4")
    if mode == "perceptron" or mode =="perceptron4" :
        # Données AND
        X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_and = np.array([0, 0, 0, 1])  # 0 pour False, 1 pour True, car elle fait parti du dommaine d'activation des fonctions.

        # OR
        X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_or = np.array([0, 1, 1, 1]) # 0 pour False, 1 pour True, car elle fait parti du dommaine d'activation des fonctions.

        # XOR
        X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y_xor = np.array([0, 1, 1, 0]) # 0 pour False, 1 pour True, car elle fait parti du dommaine d'activation des fonctions.

        model = PerceptronSimple(learning_rate=0.1)
        model.fit(X_and, y_and, max_epochs=20)
        predictions = model.predict(X_and)

        for i, x in enumerate(X_and):
            print(f"debut : {x}, Predic : {predictions[i]}, Sortie : {y_and[i]}")
        print(f" {model.score(X_and, y_and) * 100:.2f}%")
    
    else:
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

def generer_donnees_separables(n_points=100, noise=0.1):
    """
    Génère deux classes de points linéairement séparables
    """
    np.random.seed(42)
    # TODO: Générer deux nuages de points séparables
    n = n_points // 2
    
    x1 = np.random.normal(2, noise, (n, 2))
    y1 = [1 for _ in range(n)]
    x2 = np.random.normal(-2, noise, (n, 2))
    y2 = [-1 for _ in range(n)]

    x = list(x1) + list(x2)
    y = y1 + y2
    x = np.array(x)
    y = np.array(y)
    # Classe 1: points autour de (2, 2)
    # Classe 2: points autour de (-2, -2)
    return x, y

def visualiser_donnes(X, y, w=None, b=None, title="Données"):
    """
    Visualise les données et optionnellement la droite de séparation
    """
    plt.figure(figsize=(8, 6))
    # Afficher les points
    mask_pos = (y == 1)
    plt.scatter(X[mask_pos, 0], X[mask_pos, 1], c='blue', marker='+', s=100, label='Classe +1')
    plt.scatter(X[~mask_pos, 0], X[~mask_pos, 1], c='red', marker='*', s=100, label='Classe -1')
    # Afficher la droite de séparation si fournie
    if w is not None and b is not None:
        # TODO: Tracer la droite w·x + b = 0
        x_vals = np.linspace(X[:, 0].min(), X[:, 0].max(), 100)
        y_vals = -(w[0] * x_vals + b) / w[1]
        plt.plot(x_vals, y_vals, 'k--', label='Droite de décision')
    
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.legend()
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()
