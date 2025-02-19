import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# Base URL
BASE_URL = "https://prod.politmonitor.ch/export/consultations/"

# Target directory
TARGET_DIR = "json"

# Ensure the directory exists
os.makedirs(TARGET_DIR, exist_ok=True)

# Get the HTML content
response = requests.get(BASE_URL)
if response.status_code != 200:
    print(f"Failed to access {BASE_URL}")
    exit(1)

# Parse HTML to find all JSON links
soup = BeautifulSoup(response.text, "html.parser")
links = soup.find_all("a")

# Download each JSON file
for link in links:
    href = link.get("href")
    if href and href.endswith(".json"):  # Ensure it's a JSON file
        file_url = urljoin(BASE_URL, href)
        file_path = os.path.join(TARGET_DIR, href)

        print(f"Downloading {file_url} to {file_path}...")

        # Download the file
        file_response = requests.get(file_url, stream=True)
        if file_response.status_code == 200:
            with open(file_path, "wb") as f:
                for chunk in file_response.iter_content(chunk_size=1024):
                    f.write(chunk)
            print(f"✅ {href} downloaded successfully.")
        else:
            print(f"❌ Failed to download {href}")

print("🎉 All downloads completed!")
