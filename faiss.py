import numpy as np
from scipy.spatial import distance

class SimpleKNN:
    def __init__(self, data_vectors):
        print("sdfassadf")
        self.data = np.array(data_vectors)  # Speichert die Vektoren

    def search(self, query_vector, k=5):
        """Findet die k-nächsten Nachbarn zu einem Abfragevektor."""
        distances = np.array([distance.euclidean(query_vector, vec) for vec in self.data])
        nearest_indices = distances.argsort()[:k]  # k-kleinste Distanzen auswählen
        return nearest_indices, distances[nearest_indices]

# Beispiel-Datenbank mit zufälligen 3D-Vektoren
data_vectors = np.random.rand(1000, 3)

# Initialisiere die KNN-Suche
knn = SimpleKNN(data_vectors)

# Beispiel-Anfrage: Suche nächsten Vektor zu [0.1, 0.2, 0.3]
query = np.array([0.1, 0.2, 0.3])
indices, distances = knn.search(query, k=5)

print("Indices der nächsten Nachbarn:", indices)
print("Distanzen:", distances)
