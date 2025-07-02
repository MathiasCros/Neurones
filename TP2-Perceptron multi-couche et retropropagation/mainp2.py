import numpy as np

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
        from activation_functions import ActivationFunction
        self.activation_func = ActivationFunction(activation)

    def forward(self, X):
        """
        Propagation avant
        X: matrice d'entrée (n_features, n_samples)
        """
        # TODO: Implémenter la propagation avant
        # Stocker les valeurs intermédiaires pour la rétropropagation
        self.last_input = X
        self.last_z = np.dot(self.weights, X) + self.bias  # Combinaison linéaire
        self.last_activation = self.activation_func.forward(self.last_z)  # Après fonction d'activation
        
        return self.last_activation

    def backward(self, gradient_from_next_layer):
        """
        Rétropropagation
        gradient_from_next_layer: gradient venant de la couche suivante
        """
        # TODO: Calculer les gradients par rapport aux poids et biais
        # TODO: Calculer le gradient à propager vers la couche précédente

        # Gradient par rapport à la fonction d'activation
        grad_activation = gradient_from_next_layer * self.activation_func.derivative(self.last_z)

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