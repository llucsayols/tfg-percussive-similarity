"""
build_test_set.py
Genera el conjunt de test per a l'avaluació semàntica (Precision@10).

Selecciona 5 sons per categoria del conjunt d'avaluació de FSD50K (eval set),
aplicant el mateix filtre d'onset_rate <= 3 que es va aplicar al dev set
per assegurar que les queries són one-shots i no loops.

Requereix haver executat prèviament extract_features_eval.py.
Guarda els IDs seleccionats a evaluation/results/test_queries.json.
"""

import json
import random
import pandas as pd
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR          = Path("/home/llucsayols/similarity-tool")
EVAL_CSV          = Path("/mnt/c/TFG/Dataset/eval.csv")
EVAL_WAV_DIR      = Path("/mnt/c/TFG/Dataset/FSD50K.EVAL_AUDIO")
EVAL_FEATURES_DIR = BASE_DIR / "evaluation" / "features_eval"
RESULTS_DIR       = BASE_DIR / "evaluation" / "results"

# ── Paràmetres ────────────────────────────────────────────────────────────
N_QUERIES_PER_CAT = 5
RANDOM_SEED       = 42
ONSET_RATE_MAX    = 3.0

SPECIFIC_LABELS = [
    'Bass_drum', 'Snare_drum', 'Hi-hat', 'Cymbal', 'Crash_cymbal',
    'Cowbell', 'Tambourine', 'Clapping', 'Mallet_percussion', 'Gong'
]

# ── Carrega metadades del eval set ────────────────────────────────────────
print("Carregant eval.csv...")
df = pd.read_csv(EVAL_CSV)
df['fname'] = df['fname'].astype(str)
print(f"Total sons a eval set: {len(df)}")

# ── Carrega onset_rates dels sons processats ──────────────────────────────
print("Carregant onset_rates...")
onset_rates = {}
for json_path in EVAL_FEATURES_DIR.glob("*.json"):
    with open(json_path) as f:
        data = json.load(f)
    rate = data.get("rhythm.onset_rate", None)
    if rate is not None:
        onset_rates[json_path.stem] = float(rate)

print(f"Sons amb onset_rate calculat: {len(onset_rates)}")
filtered = {sid: r for sid, r in onset_rates.items() if r <= ONSET_RATE_MAX} #només sons amb onset_rate <= 3 (one-shots, la gran majoria))
print(f"Sons amb onset_rate <= {ONSET_RATE_MAX}: {len(filtered)}")

# ── Selecciona 5 sons per categoria ──────────────────────────────────────
random.seed(RANDOM_SEED)
test_queries = {}

print(f"\nSeleccionant {N_QUERIES_PER_CAT} sons per categoria (onset_rate <= {ONSET_RATE_MAX}):")
print("-" * 55)

for label in SPECIFIC_LABELS:
    candidates = []
    for _, row in df.iterrows():
        sound_id = row['fname']
        labels = [l.strip() for l in str(row['labels']).split(',')]
        #mirem que els sons que fem servir per evaluar tinguin el label i que tinguin onset_rate <= 3 (one-shots)
        if label in labels and sound_id in filtered: 
            wav_path = EVAL_WAV_DIR / f"{sound_id}.wav"
            if wav_path.exists():
                candidates.append(sound_id)

    if len(candidates) == 0:
        print(f"  {label}: CAP so disponible!")
        test_queries[label] = []
        continue

    selected = random.sample(candidates, min(N_QUERIES_PER_CAT, len(candidates)))
    test_queries[label] = selected
    print(f"  {label:<25}: {len(candidates)} disponibles → {len(selected)} seleccionats")

# ── Guarda el test set ────────────────────────────────────────────────────
output_path = RESULTS_DIR / "test_queries.json"
with open(output_path, "w") as f:
    json.dump(test_queries, f, indent=2)

total = sum(len(v) for v in test_queries.values())
print(f"\n✓ Test set guardat a {output_path}")
print(f"  Total queries: {total}")
print(f"  Filtre onset_rate: <= {ONSET_RATE_MAX}")
print(f"  Seed: {RANDOM_SEED} (reproduïble)")