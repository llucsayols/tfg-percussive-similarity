import json
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

BASE_DIR     = Path("/home/llucsayols/similarity-tool")
DATA_DIR     = BASE_DIR / "data"
FEATURES_DIR = DATA_DIR / "features_wav"
VERSION_DIR  = BASE_DIR / "versions" / "v1_variance_threshold"

# ── Reutilitza les keys i el filtre de v0 ─────────────────────────────────
V0_DIR = BASE_DIR / "versions" / "v0_music_extractor_complet"
with open(V0_DIR / "scalar_keys.json") as f:
    numeric_scalar = json.load(f)
with open(V0_DIR / "vector_keys.json") as f:
    fixed_vector_keys = json.load(f)
with open(DATA_DIR / "sound_ids_list.json") as f:
    sound_ids_list = json.load(f)

# ── Funcions ──────────────────────────────────────────────────────────────
def flatten(val):
    if isinstance(val, list):
        result = []
        for v in val:
            result.extend(flatten(v))
        return result
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

# ── Construeix la matriu (mateixos sons que v0) ───────────────────────────
print("Construint vectors...")
vectors = []
for sound_id in sound_ids_list:
    json_path = FEATURES_DIR / f"{sound_id}.json"
    vec = json_to_vector(json_path, numeric_scalar, fixed_vector_keys)
    vectors.append(vec)

vectors = np.array(vectors, dtype=np.float32)
print(f"✓ Matriu original: {vectors.shape}")

# ── Normalització Z-score ─────────────────────────────────────────────────
scaler = StandardScaler()
vectors_norm = scaler.fit_transform(vectors)

# ── VarianceThreshold ─────────────────────────────────────────────────────
# Elimina descriptors amb variància < threshold després de normalitzar
# Amb dades normalitzades, threshold=0.01 elimina les dimensions quasi constants
selector = VarianceThreshold(threshold=0.01)
vectors_selected = selector.fit_transform(vectors_norm)

n_original  = vectors_norm.shape[1]
n_selected  = vectors_selected.shape[1]
n_removed   = n_original - n_selected

print(f"\nDimensions originals:  {n_original}")
print(f"Dimensions seleccionades: {n_selected}")
print(f"Dimensions eliminades: {n_removed}")

# ── Guarda ────────────────────────────────────────────────────────────────
np.save(VERSION_DIR / "vectors_scaled.npy", vectors_selected)
with open(VERSION_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open(VERSION_DIR / "selector.pkl", "wb") as f:
    pickle.dump(selector, f)

print(f"\n✓ Tot guardat a {VERSION_DIR}")
print(f"  Sons indexats: {len(sound_ids_list)}")
print(f"  Dimensions finals: {vectors_selected.shape[1]}")

# ── Anàlisi de variàncies ─────────────────────────────────────────────────
variances = selector.variances_
mask = selector.get_support()

# Construeix la llista de noms de dimensions
all_keys = list(numeric_scalar)
for key, _ in fixed_vector_keys:
    json_path = FEATURES_DIR / f"{sound_ids_list[0]}.json"
    with open(json_path) as f:
        data = json.load(f)
    vals = data.get(key, [])
    flat = flatten(vals)
    all_keys.extend([f"{key}[{i}]" for i in range(len(flat))])

print(f"\n── 10 dimensions amb MENYS variància (eliminades o quasi):")
sorted_idx = np.argsort(variances)
for idx in sorted_idx[:10]:
    estat = "ELIMINADA" if not mask[idx] else "conservada"
    name = all_keys[idx] if idx < len(all_keys) else f"dim_{idx}"
    print(f"  {name}: var={variances[idx]:.6f} [{estat}]")

print(f"\n── 10 dimensions amb MÉS variància (més discriminatives):")
for idx in sorted_idx[-10:][::-1]:
    name = all_keys[idx] if idx < len(all_keys) else f"dim_{idx}"
    print(f"  {name}: var={variances[idx]:.4f}")