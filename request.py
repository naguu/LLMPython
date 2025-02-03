import sqlite3
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch



# **2️⃣ Daten aus der SQLite-Datenbank laden**
def load_data():
    connection = sqlite3.connect('db/database.db')
    df = pd.read_sql_query("SELECT * FROM documents", connection)
    connection.close()
    
    # Falls die Datenbank leer ist, Fehler ausgeben
    if df.empty:
        print("⚠️ Fehler: Die Datenbank enthält keine Daten!")
        exit(1)

    print("✅ Datenbank geladen!")
    print(df.head())  # Zeigt die ersten 5 Einträge zur Kontrolle
    return df

# **3️⃣ Daten in Vektoren umwandeln**
def vectorize_data(df):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Überprüfen, ob df existiert
    if df is None or df.empty:
        print("⚠️ Fehler: Die Datenbank enthält keine Daten!")
        exit(1)

    # Store data as a list of dictionaries
    data = [
        {"title": row["title"], "content": row["content"]}
        for _, row in df.iterrows()
    ]

    # Vektorisierung der Texte
    text_data = [f"Title: {doc['title']}, Content: {doc['content']}" for doc in data]
    embeddings = model.encode(text_data)

    print("✅ Daten erfolgreich vektorisiert!")
    return data, embeddings, model

# **4️⃣ FAISS Vektorsuche initialisieren**
def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    print(f"✅ FAISS Index mit {index.ntotal} Vektoren erstellt!")
    return index

# **5️⃣ Anfrage verarbeiten und relevante Daten abrufen**
def retrieve_relevant_data(query, model, index, data, top_k=5):
    query_embedding = model.encode([query])
    distances, indices = index.search(np.array(query_embedding), k=top_k)

    relevant_data = []
    for i, idx in enumerate(indices[0]):
        if distances[0][i] < 1.5:  # Nur relevante Ergebnisse unter einem Distanzwert nehmen
            relevant_data.append(f"Title: {data[idx]['title']}\nContent: {data[idx]['content']}")

    retrieved_text = "\n\n".join(relevant_data)
    return retrieved_text

# **6️⃣ Antwort mit LLM generieren**
def generate_response(query, retrieved_text, model_llm, tokenizer, device):
    messages = [
        {"role": "system", "content": "Beantworte die Frage basierend auf dem gegebenen Kontext."},
        {"role": "user", "content": f"Kontext:\n{retrieved_text}\n\nFrage: {query}"}
    ]

    input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    output = model_llm.generate(
        inputs.input_ids,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )

    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print("\n🤖 Generierte Antwort:")
    print(response)

# **💡 Hauptprogramm**
if __name__ == "__main__":

    print(torch.cuda.is_available())  # Should print: True
    print(torch.cuda.device_count())  # Should print: 1
    print(torch.cuda.get_device_name(0))  # Should print: "NVIDIA GeForce RTX 2080 Super"

    df = load_data()  # Lade Daten aus der Datenbank
    data, embeddings, model = vectorize_data(df)  # Vektorisiere Daten
    index = create_faiss_index(np.array(embeddings))  # Erstelle FAISS-Index

    print("\n💬 Gib eine Frage ein oder tippe 'exit', um zu beenden.\n")

    # **Modell nur einmal laden!**
    # Wähle ein kleineres Modell
    model_name = "deepseek-ai/deepseek-llm-7b-chat"

    # Prüfe, ob CUDA verfügbar ist
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model_llm = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    trust_remote_code=True
    )



    while True:
        query = input("❓ Frage: ").strip()

        if query.lower() == "exit":
            print("\n👋 Programm beendet. Bis zum nächsten Mal!\n")
            break  # Schleife beenden

        retrieved_text = retrieve_relevant_data(query, model, index, data)

        if not retrieved_text.strip():
            print("\n⚠️ Keine passenden Informationen gefunden.\n")
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # **Hier Modell weiterverwenden**
            generate_response(query, retrieved_text, model_llm, tokenizer, device)  