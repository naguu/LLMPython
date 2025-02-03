import sqlite3
import json

# Connect to the SQLite database
connection = sqlite3.connect("db/database.db")
cursor = connection.cursor()

# Execute the SQL query
cursor.execute("SELECT * FROM documents")

# Fetch all rows (this was missing in your code)
rows = cursor.fetchall()

# Get column names dynamically
column_names = [description[0] for description in cursor.description]

# Convert rows into a list of dictionaries (Easier version)
documents = []
for row in rows:
    document = dict(zip(column_names, row))  # Convert each row (tuple) to a dictionary
    documents.append(document)  # Add to the list

# Print the JSON output
print(json.dumps(documents, indent=4, ensure_ascii=False))

# Close the database connection
connection.close()
