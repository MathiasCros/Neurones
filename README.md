Pourquoi la fonction de Heaviside pose-t-elle problème pour l'apprentissage par gradient ?

    Pour l'apprentissage par gradient, faut que la fonction soit dérivable, ce que n'exisite pas pour Heaviside

Dans quels cas utiliser sigmoid vs tanh ?

    Gigmoid sert à calculer des probabilité ayant une intervalle de 0 à 1.
    Tahn est un modele d'apprentissage vers 0 qui sert de vecteur d'apprentisage quand les modeles linéaires ne peuvent pas le faire

Pourquoi ReLU est-elle si populaire dans les réseaux profonds ?

    ReLU a un coup de calcul en moins, ayant besoin de faire une comparaison uniquement (<0), il permet aussi d'atténuer le problème de la disparition du gradient.

Quel est l'avantage du Leaky ReLU ?

    Il n'aura jamais dée neurones purement égal à 0 du à son coefficient alpha. Cela évite de phénomene de neuronnes mort.