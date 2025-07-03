import numpy as np
import matplotlib.pyplot as plt
from ActivationFunction import ActivationFunction

class CoucheNeurones:
    def __init__(self, n_input, n_neurons, activation='sigmoid', learning_rate=0.01):
        """
        Initialise une couche de neurones

        Parameters:
        - n_input: nombre d'entrées
        - n_neurons: nombre de neurones dans cette couche
        - activation: fonction d'activation ('sigmoid', 'tanh', 'relu')
        - learning_rate: taux d'apprentissage
        """
        self.n_input = n_input
        self.n_neurons = n_neurons
        self.activation_name = activation
        self.learning_rate = learning_rate

        # Initialisation Xavier/Glorot pour éviter l'explosion/disparition des gradients
        limit = np.sqrt(6 / (n_input + n_neurons))
        self.weights = np.random.uniform(-limit, limit, (n_neurons, n_input))
        self.bias = np.zeros((n_neurons, 1))

        # Variables pour stocker les valeurs lors de la propagation
        self.last_input = None
        self.last_z = None
        self.last_activation = None

        # Import de la fonction d'activation du TP précédent
        from ActivationFunction.py import ActivationFunction
        self.activation_func = ActivationFunction(activation)

    
    def forward(self, X):
        """
        Propagation avant
        X: matrice d'entrée (n_features, n_samples)
        """
        # TODO: Implémenter la propagation avant
        # Stocker les valeurs intermédiaires pour la rétropropagations
        self.last_input = X
        self.last_z = np.dot(self.weights, X) + self.bias  # Combinaison linéaire
        self.last_activation = self.activation_func.forward(self.last_z)  # Après fonction d'activation
        
        return self.last_activation

    def backward(self, gradient_from_next_layer):
        """
        Rétropropagation
        gradient_from_next_layer: gradient venant de la couche suivante
        """
        # np.dot => W(-1)a(etape-1)
        # +b si y a un biais
        # .dot => * matriciel
        
        # TODO: Calculer les gradients par rapport aux poids et biais
        # TODO: Calculer le gradient à propager vers la couche précédente

        # Gradient par rapport à la fonction d'activation
        grad_activation = gradient_from_next_layer * ActivationFunction.derivative(self.last_z)

        # Gradient par rapport aux poids
        grad_weights = np.dot(grad_activation, self.last_input.T) / self.last_input.shape[1]  

        # Gradient par rapport aux biais  
        grad_bias = np.sum(grad_activation, axis=1, keepdims=True) / self.last_input.shape[1]

        # Gradient à propager vers la couche précédente
        grad_input = np.dot(self.weights.T, grad_activation)

        # Mise à jour des paramètres
        self.weights -= self.learning_rate * grad_weights
        self.bias -= self.learning_rate * grad_bias

        return grad_input
    
        
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
