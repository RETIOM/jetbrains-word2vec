import os
import requests
import zipfile

dataset_url = (
    "https://www.kaggle.com/api/v1/datasets/download/vadimkurochkin/wikitext-103"
)
dataset_dir = "./data"
zip_path = os.path.join(dataset_dir, "wikitext-103.zip")

os.makedirs(dataset_dir, exist_ok=True)

print("Downloading dataset...")
with requests.get(dataset_url, stream=True) as r:
    r.raise_for_status()
    with open(zip_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
print("Downloaded zip file to:", zip_path)

print("Extracting files...")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(dataset_dir)
print("Extracted dataset to:", dataset_dir)

os.remove(zip_path)
print("Removed zip file:", zip_path)
