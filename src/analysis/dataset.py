import os, json, csv
from datetime import datetime

# creates data directory relative to the src directory
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data")

# get timestamped filename for saving results
def get_timestamped_filename(basename: str, ext: str = "json"):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{basename}.{ext}"

# save data to a JSON file
def save_json(data, filename):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, filename)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

# load data from a JSON file
def load_json(filename):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "r") as f:
        return json.load(f)

# export datasets to a CSV file
def export_to_csv(datasets, filename):
    path = os.path.join(DATA_DIR, filename)
    all_rows = []
    for method, results in datasets.items():
        for row in results:
            row_copy = row.copy()
            row_copy["Methode"] = method
            all_rows.append(row_copy)
    if not all_rows:
        return
    with open(path, mode="w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
        writer.writeheader()
        writer.writerows(all_rows)