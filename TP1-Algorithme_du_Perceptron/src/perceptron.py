import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class PerceptronSimple:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None

    def fit(self, X, y, max_epochs=100):
        """
        Entraîne le perceptron
        X: matrice des entrées (n_samples, n_features)
        y: vecteur des sorties désirées (n_samples,)
        """
        # Initialisation les poids et le biais
        self.weights = np.random.randn(X.shape[1])
        self.bias = 0.0

        for e in tqdm(max_epochs):
            for b in range(X.shape[0]):
                x = X[b]
                y_true = y[b]
                # TODO: Implémenter l'algorithme d'apprentissage
                # Possible d'optimiser d'aventage numpy
                activation = np.dot(self.weights, x) + self.bias #pondéré + bias
                if activation >= 0: y_pred = 1
                else:y_pred = 0

                erreur = y_true - y_pred
                
                self.weights += self.learning_rate * x * erreur
                self.bias += self.learning_rate * erreur

    def predict(self, X):
        """Prédit les sorties pour les entrées X"""
            
        y_pred = np.zeros(X.shape[0])
        for b in range(X.shape[0]):
        # TODO: Calculer les prédictions
            x = X[b] # (n_features,)
            act = np.dot(self.weights, x) + self.bias #pondéré + bias
            if act >= 0: y_pred[b] = 1
            else: y_pred[b] = 0

            

        return y_pred

    def score(self, X, y):
        """Calcule l'accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)
