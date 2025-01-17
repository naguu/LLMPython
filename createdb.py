import sqlite3

# Verbindung zur SQLite-Datenbank
connection = sqlite3.connect("database.db")
cursor = connection.cursor()

# Tabelle für lange Texte erstellen
cursor.execute('''
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT,
    content TEXT
)
''')

# Beispielhafte lange Texte einfügen
text_data = [
    ("Dokument 1", "Das ist ein sehr langer Text über das Wetter in der Schweiz. Es regnet oft in Zürich."),
    ("Dokument 2", "Python ist eine Programmiersprache, die oft für Machine Learning verwendet wird."),
    ("Dokument 3", "In der Geschichte der Informatik war Alan Turing eine wichtige Person."),
]

cursor.executemany("INSERT INTO documents (title, content) VALUES (?, ?)", text_data)

connection.commit()
connection.close()

print("✅ Datenbank mit langen Texten erstellt!")
