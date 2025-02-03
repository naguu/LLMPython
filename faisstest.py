import numpy as np
import faiss
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer

# Modell für die Vektorisierung laden
model = SentenceTransformer('all-MiniLM-L6-v2')

# Liste für gespeicherte Sätze
sentences = []
embeddings = []

def add_sentence(sentence):
    """Wandelt einen Satz in einen Vektor um und speichert ihn."""
    vector = model.encode([sentence])[0]
    sentences.append(sentence)
    embeddings.append(vector)
    print(f'✅ Satz gespeichert: "{sentence}"')


def create_faiss_index():
    """Erstellt einen FAISS-Index aus den gespeicherten Vektoren."""
    if not embeddings:
        print("⚠️ Keine Sätze vorhanden!")
        return None
    
    dimension = len(embeddings[0])
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    print(f'✅ FAISS-Index mit {len(embeddings)} Sätzen erstellt!')
    return index


def search_similar(query, index, top_k=3):
    """Findet die ähnlichsten Sätze zu einer neuen Eingabe."""
    query_vector = model.encode([query])
    distances, indices = index.search(np.array(query_vector), k=top_k)
    
    print("\n🔍 Ähnliche Sätze gefunden:")
    for i, idx in enumerate(indices[0]):
        print(f'({i+1}) {sentences[idx]}  [Distanz: {distances[0][i]:.4f}]')
    
    return indices[0]


def plot_embeddings():
    """Visualisiert die Vektoren in 2D mittels PCA."""
    if len(embeddings) < 2:
        print("⚠️ Mindestens zwei Sätze erforderlich für Visualisierung!")
        return
    
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(np.array(embeddings))
    
    plt.figure(figsize=(8, 6))
    for i, txt in enumerate(sentences):
        plt.scatter(reduced_vectors[i, 0], reduced_vectors[i, 1], label=txt)
        plt.annotate(txt, (reduced_vectors[i, 0], reduced_vectors[i, 1]))
    
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.title("FAISS-Vektor-Darstellung")
    plt.legend()
    plt.show()

# Interaktive Eingabe
while True:
    print("\n1️⃣ Satz speichern | 2️⃣ Ähnliche Sätze finden | 3️⃣ Visualisieren | 4️⃣ Beenden")
    choice = input("➡️ Deine Wahl: ")
    
    if choice == "1":
        text = input("Gib einen Satz ein: ")
        add_sentence(text)
    elif choice == "2":
        if not embeddings:
            print("⚠️ Keine Sätze gespeichert!")
            continue
        faiss_index = create_faiss_index()
        query = input("Gib einen Suchsatz ein: ")
        search_similar(query, faiss_index)
    elif choice == "3":
        plot_embeddings()
    elif choice == "4":
        print("👋 Programm beendet!")
        break
    else:
        print("⚠️ Ungültige Eingabe!")
