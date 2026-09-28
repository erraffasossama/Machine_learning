import numpy as np
from collections import Counter

class KNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = np.array(X)
        self.y_train = np.array(y)

    def euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def predict(self, X):

        predictions = []

        for sample in X:

            distances = []

            # Calcul des distances
            for train_sample in self.X_train:

                distance = self.euclidean_distance(sample, train_sample)
                distances.append(distance)

            # Trier
            k_indices = np.argsort(distances)[:self.k]

            # Classes des voisins
            k_labels = self.y_train[k_indices]

            # Vote majoritaire
            most_common = Counter(k_labels).most_common(1)

            predictions.append(most_common[0][0])

        return np.array(predictions)


import numpy as np

X_train = np.array([
    [2,4],
    [4,6],
    [6,8],
    [8,10],
    [10,12]
])

y_train = np.array([
    0,
    0,
    1,
    1,
    1
])

knn = KNN(k=3)

knn.fit(X_train, y_train)

X_test = np.array([
    [5,7],
    [9,11]
])

prediction = knn.predict(X_test)

print(prediction)









# Importer la librairie numpy pour les opérations mathématiques
import numpy as np

# Définition de la classe KNN
class KNN:
    # Initialisation de la classe avec k (nombre de voisins)
    def __init__(self, k=3):
        self.k = k  # k = nombre de voisins à considérer

    # Méthode "fit" : on stocke juste les données d'entraînement
    def fit(self, X, y):
        self.X_train = X  # X = features (données d'entrée)
        self.y_train = y  # y = labels (classes)

    # Méthode "predict" : prédit les classes pour de nouvelles données
    def predict(self, X):
        # Pour chaque point de X, on appelle la méthode _predict
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)  # Retourne les prédictions sous forme de tableau

    # Méthode privée _predict : prédit la classe d'un seul point x
    def _predict(self, x):
        # Calculer la distance euclidienne entre x et tous les points d'entraînement
        distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]

        # Trouver les indices des k plus proches voisins
        k_indices = np.argsort(distances)[:self.k]

        # Récupérer les labels de ces k voisins
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # Vote majoritaire : retourne la classe la plus fréquente
        return np.argmax(np.bincount(k_nearest_labels))

# Exemple d'utilisation
X_train = np.array([[1,2], [2,3], [3,1], [6,6], [7,8]])  # 5 points, 2 features
y_train = np.array([0,0,0,1,1])  # 2 classes (0 et 1)
knn = KNN(k=3)  # On crée un KNN avec k=3
knn.fit(X_train, y_train)  # On "entraine" le modèle
X_test = np.array([[3,2], [6,7]])  # 2 points à prédire
print(knn.predict(X_test))  # Prédictions : [0 1]