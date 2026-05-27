import json
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler

BASE_DIR     = Path("/home/llucsayols/similarity-tool")
DATA_DIR     = BASE_DIR / "data"
FEATURES_DIR = DATA_DIR / "features_wav"
VERSION_DIR  = BASE_DIR / "versions" / "v0_music_extractor_complet"

# Carrega IDs disponibles
with open(DATA_DIR / "available_ids.json") as f:
    available = set(json.load(f))

# Analitza les keys d'un JSON per determinar escalars i vectorials
json_files = list(FEATURES_DIR.glob("*.json"))
with open(json_files[0]) as f:
    sample = json.load(f)

numeric_scalar = []
numeric_vector = []
for key, val in sample.items():
    if isinstance(val, list):
        numeric_vector.append((key, len(val)))
    elif isinstance(val, (int, float)):
        numeric_scalar.append(key)

# Exclou descriptors problemàtics (neteja estadística)
fixed_vector_keys = [
    (k, d) for k, d in numeric_vector
    if k != 'rhythm.beats_position'
    and 'gfcc.icov' not in k
    and 'gfcc.cov' not in k
    and 'mfcc.cov' not in k
    and 'mfcc.icov' not in k  
]

numeric_scalar = [
    k for k in numeric_scalar
    if not k.startswith('metadata.')
    and 'spectral_spread' not in k
]

print(f"Descriptors escalars:   {len(numeric_scalar)}")
print(f"Descriptors vectorials: {len(fixed_vector_keys)}")

def flatten(val):
    if isinstance(val, list):
        result = []
        for v in val:
            result.extend(flatten(v))
        return result
    else:
        return [float(val)]

def json_to_vector(json_path, scalar_keys, vector_keys):
    with open(json_path) as f:
        data = json.load(f)
    vector = []
    for key in scalar_keys:
        vector.append(float(data.get(key, 0.0)))
    for key, _ in vector_keys:
        vals = data.get(key, [])
        vector.extend(flatten(vals))
    return np.array(vector, dtype=np.float32)

# Filtre onset_rate ≤ 3
onset_rates = {}
for json_path in FEATURES_DIR.glob("*.json"):
    sound_id = json_path.stem
    with open(json_path) as f:
        data = json.load(f)
    onset_rate = data.get("rhythm.onset_rate", None)
    if onset_rate is not None:
        onset_rates[sound_id] = float(onset_rate)

available_filtered = set(sid for sid, rate in onset_rates.items() if rate <= 3.0)
print(f"\nSons abans del filtre onset_rate: {len(onset_rates)}")
print(f"Sons després del filtre (≤3):     {len(available_filtered)}")

# Construeix la matriu de vectors
print("\nConstruint vectors...")
sound_ids_list = []
vectors = []

for json_path in sorted(FEATURES_DIR.glob("*.json")):
    sound_id = json_path.stem
    if sound_id not in available_filtered:
        continue
    vec = json_to_vector(json_path, numeric_scalar, fixed_vector_keys)
    sound_ids_list.append(sound_id)
    vectors.append(vec)

vectors = np.array(vectors, dtype=np.float32)
print(f"✓ Matriu construïda: {vectors.shape}")

# Normalització i scaler
scaler = StandardScaler()
vectors_scaled = scaler.fit_transform(vectors)
print(f"✓ Normalitzat: {vectors_scaled.shape}")

# Guarda a la carpeta de la versió
np.save(VERSION_DIR / "vectors_scaled.npy", vectors_scaled)
with open(DATA_DIR / "sound_ids_list.json", "w") as f:
    json.dump(sound_ids_list, f)
with open(VERSION_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open(VERSION_DIR / "scalar_keys.json", "w") as f:
    json.dump(numeric_scalar, f)
with open(VERSION_DIR / "vector_keys.json", "w") as f:
    json.dump(fixed_vector_keys, f)

print(f"\n✓ Tot guardat a {VERSION_DIR}")
print(f"  Sons indexats: {len(sound_ids_list)}")
print(f"  Dimensions:    {vectors_scaled.shape[1]}")