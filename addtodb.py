import sqlite3

# Verbindung zur SQLite-Datenbank
connection = sqlite3.connect("db/database.db")
cursor = connection.cursor()

cur = cursor.execute("SELECT MAX(title) FROM documents")

maxdok = cur.fetchone()[0][9::]
maxdok = int(maxdok)
maxdock = maxdok + 1
maxdock = str(maxdock)

print(maxdock)

text = input("Enter text: ")


cursor.execute("INSERT INTO documents (title, content) VALUES (?, ?)", ("Dokument: " + maxdock, text))


connection.commit()
connection.close()

print("✅ text added to database!")

