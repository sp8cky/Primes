import json
import os
import csv
from datetime import datetime


DATA_DIR = "data"

def get_timestamped_filename(basename: str, ext: str = "json"):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{basename}.{ext}"

def save_json(data, filename):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def load_json(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r") as f:
        return json.load(f)

def export_to_csv(data, filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)