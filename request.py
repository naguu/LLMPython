import sqlite3
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from transformers import AutoModelForCausalLM, LlamaTokenizer
import torch



# **2️⃣ Daten aus der SQLite-Datenbank laden**
def load_data():
    connection = sqlite3.connect('database.db')
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
def retrieve_relevant_data(query, model, index, data, top_k=2):
    query_embedding = model.encode([query])

    # Suche nach den nächsten Vektoren
    distances, indices = index.search(np.array(query_embedding), k=top_k)

    # Ähnliche Daten ausgeben
    relevant_data = []
    for idx in indices[0]:  # idx refers to index positions in `data`
        relevant_data.append(f"Title: {data[idx]['title']}\nContent: {data[idx]['content']}")

    # Join all relevant documents into one block of text
    retrieved_text = "\n\n".join(relevant_data)

    print("\n🔍 Ähnliche Daten gefunden:")
    print(retrieved_text)

    return retrieved_text

# **6️⃣ Antwort mit LLM generieren**
def generate_response(query, retrieved_text, model_llm, tokenizer, device):
    # Generiere Input-Text
    input_text = f"{retrieved_text}\n\nFrage: {query}\nAntwort:"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)

    # Setze `attention_mask` explizit
    attention_mask = torch.ones_like(input_ids)

    # Stelle sicher, dass `pad_token_id` gesetzt ist
    model_llm.config.pad_token_id = tokenizer.eos_token_id

    # Generiere Antwort mit optimierten Einstellungen
    output = model_llm.generate(
        input_ids,
        attention_mask=attention_mask,  # Füge `attention_mask` hinzu
        max_new_tokens=100
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
    model_name = "HuggingFaceH4/zephyr-7b-alpha"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = LlamaTokenizer.from_pretrained(model_name, use_fast=False)
    model_llm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

    while True:
        query = input("❓ Frage: ").strip()

        if query.lower() == "exit":
            print("\n👋 Programm beendet. Bis zum nächsten Mal!\n")
            break  # Schleife beenden

        retrieved_text = retrieve_relevant_data(query, model, index, data)

        if not retrieved_text.strip():
            print("\n⚠️ Keine passenden Informationen gefunden.\n")
        else:
            # **Hier Modell weiterverwenden**
            generate_response(query, retrieved_text, model_llm, tokenizer, device)  