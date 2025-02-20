import json
import sqlite3
import os
import glob

# Define paths
db_path = os.path.join("db", "import.db")
json_folder = "json"

# Ensure the database directory exists
os.makedirs(os.path.dirname(db_path), exist_ok=True)

# Remove existing database
if os.path.exists(db_path):
    os.remove(db_path)

# Connect to SQLite database
conn = sqlite3.connect(db_path)
c = conn.cursor()

# Create tables
c.execute('''
CREATE TABLE IF NOT EXISTS affairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    affair_canton_name TEXT,
    affair_canton_abbreviation TEXT,
    affair_canton_wikidata_id TEXT,
    affair_politmonitor_export_date TEXT,
    affair_politmonitor_url TEXT,
    affair_politmonitor_id INTEGER,
    affair_key TEXT,
    affair_external_id TEXT,
    affair_external_url TEXT,
    affair_date_start TEXT,
    affair_closed BOOLEAN,
    affair_state_name_de TEXT,
    affair_state_name_fr TEXT,
    affair_state_name_it TEXT,
    affair_title_de TEXT,
    affair_title_fr TEXT,
    affair_title_it TEXT,
    affair_author_name TEXT,
    affair_author_wikidata_id TEXT
)
''')

c.execute('''
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    affair_id INTEGER,
    event_latest BOOLEAN,
    event_order INTEGER,
    event_date TEXT,
    event_text_de TEXT,
    event_text_fr TEXT,
    event_text_it TEXT,
    event_source TEXT,
    FOREIGN KEY (affair_id) REFERENCES affairs (id)
)
''')

c.execute('''
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    affair_id INTEGER,
    doc_politmonitor_id INTEGER,
    doc_date TEXT,
    doc_name TEXT,
    doc_type TEXT,
    doc_link TEXT,
    doc_category TEXT,
    doc_language TEXT,
    doc_content TEXT,
    FOREIGN KEY (affair_id) REFERENCES affairs (id)
)
''')

# Process all JSON files
json_files = glob.glob(os.path.join(json_folder, "*.json"))

for json_file in json_files:
    with open(json_file, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    affairs = data.get("data", [])
    
    for affair in affairs:
        c.execute('''
        INSERT INTO affairs (
            affair_canton_name, affair_canton_abbreviation, affair_canton_wikidata_id,
            affair_politmonitor_export_date, affair_politmonitor_url, affair_politmonitor_id,
            affair_key, affair_external_id, affair_external_url,
            affair_date_start, affair_closed, affair_state_name_de,
            affair_state_name_fr, affair_state_name_it, affair_title_de,
            affair_title_fr, affair_title_it, affair_author_name, affair_author_wikidata_id
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            affair["affair_canton_name"], affair["affair_canton_abbreviation"], affair["affair_canton_wikidata_id"],
            affair["affair_politmonitor_export_date"], affair["affair_politmonitor_url"], affair["affair_politmonitor_id"],
            affair["affair_key"], affair["affair_external_id"], affair["affair_external_url"],
            affair["affair_date_start"], affair["affair_closed"], affair["affair_state_name_de"],
            affair["affair_state_name_fr"], affair["affair_state_name_it"], affair["affair_title_de"],
            affair["affair_title_fr"], affair["affair_title_it"], affair["affair_author_name"], affair["affair_author_wikidata_id"]
        ))
        affair_id = c.lastrowid
        
        # Insert events
        for event in affair.get("affair_events", []):
            c.execute('''
            INSERT INTO events (affair_id, event_latest, event_order, event_date, event_text_de, event_text_fr, event_text_it, event_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                affair_id, event["event_latest"], event["event_order"], event["event_date"],
                event["event_text_de"], event["event_text_fr"], event["event_text_it"], event["event_source"]
            ))
        
        # Insert documents
        for doc in affair.get("affair_documents", []):
            c.execute('''
            INSERT INTO documents (affair_id, doc_politmonitor_id, doc_date, doc_name, doc_type, doc_link, doc_category, doc_language, doc_content)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                affair_id, doc["doc_politmonitor_id"], doc["doc_date"], doc["doc_name"], doc["doc_type"],
                doc["doc_link"], doc["doc_category"], doc["doc_language"], doc.get("doc_content")
            ))

# Commit changes and close connection
conn.commit()
conn.close()

print("All JSON files successfully imported into database.")
