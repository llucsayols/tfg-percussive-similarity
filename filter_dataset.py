import json
import pandas as pd
from pathlib import Path

BASE_DIR = Path("/home/llucsayols/similarity-tool")
DATA_DIR = BASE_DIR / "data"
WAV_DIR  = Path("/mnt/c/TFG/Dataset/FSD50K.DEV_AUDIO")

SPECIFIC_LABELS = [
    'Bass_drum', 'Snare_drum', 'Hi-hat', 'Cymbal', 'Crash_cymbal',
    'Cowbell', 'Tambourine', 'Clapping', 'Mallet_percussion', 'Gong'
]

dev_labels = pd.read_csv(DATA_DIR / "dev.csv")

def has_specific_label(labels_str):
    labels = [l.strip() for l in str(labels_str).split(",")]
    return any(l in SPECIFIC_LABELS for l in labels)

specific = dev_labels[dev_labels["labels"].apply(has_specific_label)]

print(f"Sons amb etiquetes específiques: {len(specific)}")
print(f"\nDistribució per etiqueta:")
from collections import Counter
label_counts = Counter()
for labels_str in specific["labels"]:
    for l in labels_str.split(","):
        l = l.strip()
        if l in SPECIFIC_LABELS:
            label_counts[l] += 1
for label, count in label_counts.most_common():
    print(f"  {label}: {count}")

all_ids = set(specific["fname"].astype(str).tolist())
wav_available = {sid for sid in all_ids if (WAV_DIR / f"{sid}.wav").exists()}

print(f"\nTotal IDs filtrats:   {len(all_ids)}")
print(f"WAVs disponibles:     {len(wav_available)}")
print(f"WAVs no disponibles:  {len(all_ids) - len(wav_available)}")

output_path = DATA_DIR / "available_ids.json"
with open(output_path, "w") as f:
    json.dump(list(wav_available), f)

print(f"\n✓ Guardat: {output_path}")
print(f"  Total IDs guardats: {len(wav_available)}")