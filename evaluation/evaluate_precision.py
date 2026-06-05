"""
evaluate_precision.py
Avaluació semàntica: Precision@10 i Mean Average Precision (MAP)

Per a cada versió (v1, v2, v3), extreu el vector de cada query del test set
(sons del FSD50K eval set, independents de l'índex) i calcula:
- Precision@10: quants dels 10 resultats retornats pertanyen a la mateixa categoria
- MAP: mitjana de l'Average Precision, que té en compte l'ordre dels resultats

Requereix haver executat prèviament:
    - build_test_set.py  → evaluation/results/test_queries.json
    - build_index_v*.py  → versions/v*/vectors_scaled.npy
"""

import json
import numpy as np
import pickle
from pathlib import Path
from scipy.spatial.distance import cdist
import pandas as pd
import essentia.standard as es

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR     = Path("/home/llucsayols/similarity-tool")
DATA_DIR     = BASE_DIR / "data"
EVAL_WAV_DIR = Path("/mnt/c/TFG/Dataset/FSD50K.EVAL_AUDIO")
EVAL_CSV     = Path("/mnt/c/TFG/Dataset/FSD50K.ground_truth/eval.csv")
RESULTS_DIR  = BASE_DIR / "evaluation" / "results"

# ── Paràmetres ────────────────────────────────────────────────────────────
K = 10

SPECIFIC_LABELS = [
    'Bass_drum', 'Snare_drum', 'Hi-hat', 'Cymbal', 'Crash_cymbal',
    'Cowbell', 'Tambourine', 'Clapping', 'Mallet_percussion', 'Gong'
]

VERSIONS = {
    "v1": BASE_DIR / "versions" / "v1_variance_threshold",
    "v2": BASE_DIR / "versions" / "v2_personalized",
    "v3": BASE_DIR / "versions" / "v3_clap",
}

# ── Carrega etiquetes de l'índex (dev set) ────────────────────────────────
print("Carregant etiquetes del dev set...")
df_dev = pd.read_csv(DATA_DIR / "dev.csv")
df_dev["fname"] = df_dev["fname"].astype(str)

with open(DATA_DIR / "sound_ids_list.json") as f:
    sound_ids_list = json.load(f)

def get_categories_dev(sound_id):
    row = df_dev[df_dev["fname"] == str(sound_id)]
    if row.empty:
        return []
    labels_str = row["labels"].values[0]
    return [l.strip() for l in str(labels_str).split(",") if l.strip() in SPECIFIC_LABELS]

sound_categories = {sid: get_categories_dev(sid) for sid in sound_ids_list}
print(f"Sons a l'índex: {len(sound_ids_list)}")

# ── Carrega test queries ──────────────────────────────────────────────────
with open(RESULTS_DIR / "test_queries.json") as f:
    test_queries = json.load(f)

total_queries = sum(len(v) for v in test_queries.values())
print(f"Queries del test set: {total_queries}")

# ── Helpers per extreure features ─────────────────────────────────────────
def flatten(val):
    if isinstance(val, list):
        result = []
        for v in val:
            result.extend(flatten(v))
        return result
    return [float(val)]

def extract_music_extractor_features(wav_path):
    features, _ = es.MusicExtractor(
        lowlevelStats=['mean', 'stdev'],
        rhythmStats=['mean', 'stdev'],
        tonalStats=['mean', 'stdev']
    )(str(wav_path))
    feature_dict = {}
    for key in features.descriptorNames():
        val = features[key]
        try:
            feature_dict[key] = val.tolist() if hasattr(val, 'tolist') else float(val)
        except (ValueError, TypeError):
            feature_dict[key] = str(val)
    return feature_dict

def build_vector(feature_dict, scalar_keys, vector_keys):
    vector = []
    for key in scalar_keys:
        vector.append(float(feature_dict.get(key, 0.0)))
    for key, _ in vector_keys:
        vals = feature_dict.get(key, [])
        vector.extend(flatten(vals))
    return np.array(vector, dtype=np.float32).reshape(1, -1)

def extract_clap_embedding(wav_path):
    import librosa, torch
    from transformers import ClapModel, ClapProcessor
    if not hasattr(extract_clap_embedding, "model"):
        print("  Carregant model CLAP...")
        extract_clap_embedding.model     = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        extract_clap_embedding.processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        extract_clap_embedding.model.eval()
    model     = extract_clap_embedding.model
    processor = extract_clap_embedding.processor
    audio, sr = librosa.load(str(wav_path), sr=48000, mono=True)
    inputs = processor(audio=audio, sampling_rate=48000, return_tensors="pt")
    with torch.no_grad():
        result = model.get_audio_features(**inputs)
        embedding = result.pooler_output
    return embedding.cpu().numpy().reshape(1, -1)

# ── Càlcul de Precision@K i Average Precision ─────────────────────────────
def compute_precision_at_k(top_k_ids, query_category, k=K):
    """Precision@K: proporció dels k resultats que tenen la categoria del query."""
    hits = sum(1 for sid in top_k_ids if query_category in sound_categories.get(sid, []))
    return hits / k

def compute_average_precision(top_k_ids, query_category):
    """
    Average Precision: mitjana de les precisions en cada posició on hi ha un hit.
    Premia tenir els hits al principi del rànquing.
    """
    hits = 0
    precision_sum = 0.0
    for i, sid in enumerate(top_k_ids, start=1):
        if query_category in sound_categories.get(sid, []):
            hits += 1
            precision_at_i = hits / i
            precision_sum += precision_at_i
    if hits == 0:
        return 0.0
    return precision_sum / hits

# ── Funció principal d'avaluació ──────────────────────────────────────────
def evaluate_version(version_name, version_dir):
    print(f"\n{'='*60}")
    print(f"Versió: {version_name}")
    print(f"{'='*60}")

    vectors = np.load(version_dir / "vectors_scaled.npy")
    with open(version_dir / "scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    selector = None
    scalar_keys = None
    vector_keys = None

    if version_name in ("v1", "v2"):
        v0_dir = BASE_DIR / "versions" / "v0_music_extractor_complet"
        if version_name == "v2":
            with open(version_dir / "scalar_keys.json") as f:
                scalar_keys = json.load(f)
            with open(version_dir / "vector_keys.json") as f:
                vector_keys = json.load(f)
        else:
            with open(v0_dir / "scalar_keys.json") as f:
                scalar_keys = json.load(f)
            with open(v0_dir / "vector_keys.json") as f:
                vector_keys = json.load(f)
            with open(version_dir / "selector.pkl", "rb") as f:
                selector = pickle.load(f)

    precision_per_cat = {}
    map_per_cat       = {}
    all_precisions    = []
    all_aps           = []

    print(f"  {'Categoria':<25} {'P@10':<8} {'AP':<8}")
    print(f"  {'-'*25} {'-'*8} {'-'*8}")

    for cat in SPECIFIC_LABELS:
        queries = test_queries.get(cat, [])
        cat_precisions = []
        cat_aps        = []

        for query_id in queries:
            wav_path = EVAL_WAV_DIR / f"{query_id}.wav"
            if not wav_path.exists():
                continue

            try:
                if version_name == "v3":
                    query_vec = extract_clap_embedding(wav_path)
                    query_vec_scaled = scaler.transform(query_vec)
                else:
                    feature_dict = extract_music_extractor_features(wav_path)
                    query_vec = build_vector(feature_dict, scalar_keys, vector_keys)
                    query_vec_scaled = scaler.transform(query_vec)
                    if selector is not None:
                        query_vec_scaled = selector.transform(query_vec_scaled)
            except Exception as e:
                print(f"  Error query {query_id}: {e}")
                continue

            distances = cdist(query_vec_scaled, vectors, metric="cosine")[0]
            top_k_idx = np.argsort(distances)[:K]
            top_k_ids = [sound_ids_list[i] for i in top_k_idx]

            p_at_k = compute_precision_at_k(top_k_ids, cat, k=K)
            ap     = compute_average_precision(top_k_ids, cat)

            cat_precisions.append(p_at_k)
            cat_aps.append(ap)

        mean_p  = np.mean(cat_precisions) if cat_precisions else 0.0
        mean_ap = np.mean(cat_aps)        if cat_aps        else 0.0
        precision_per_cat[cat] = round(mean_p, 4)
        map_per_cat[cat]       = round(mean_ap, 4)
        all_precisions.extend(cat_precisions)
        all_aps.extend(cat_aps)

        print(f"  {cat:<25} {mean_p:<8.3f} {mean_ap:<8.3f}")

    mean_p_all  = np.mean(all_precisions) if all_precisions else 0.0
    mean_ap_all = np.mean(all_aps)        if all_aps        else 0.0
    print(f"  {'-'*25} {'-'*8} {'-'*8}")
    print(f"  {'MITJANA GLOBAL':<25} {mean_p_all:<8.3f} {mean_ap_all:<8.3f}")

    return {
        "precision_per_cat": precision_per_cat,
        "map_per_cat":       map_per_cat,
        "precision_mean":    round(mean_p_all, 4),
        "map":               round(mean_ap_all, 4),
    }

# ── Execució ──────────────────────────────────────────────────────────────
print("\n" + "="*60)
print("AVALUACIÓ SEMÀNTICA — Precision@10 i MAP")
print("="*60)

all_results = {}
for version_name, version_dir in VERSIONS.items():
    if not (version_dir / "vectors_scaled.npy").exists():
        print(f"\nVersió {version_name}: fitxers no trobats, saltant...")
        continue
    all_results[version_name] = evaluate_version(version_name, version_dir)

# ── Taula resum: Precision@10 ─────────────────────────────────────────────
print("\n\n" + "="*60)
print("TAULA RESUM — Precision@10")
print("="*60)

versions_available = list(all_results.keys())
header = f"{'Categoria':<25}" + "".join(f"  {v:<8}" for v in versions_available)
print(header)
print("-" * len(header))

for cat in SPECIFIC_LABELS:
    row = f"{cat:<25}"
    for v in versions_available:
        val = all_results[v]["precision_per_cat"].get(cat, 0.0)
        row += f"  {val:<8.3f}"
    print(row)

print("-" * len(header))
mean_row = f"{'MITJANA GLOBAL':<25}"
for v in versions_available:
    mean_row += f"  {all_results[v]['precision_mean']:<8.3f}"
print(mean_row)

# ── Taula resum: MAP ──────────────────────────────────────────────────────
print("\n" + "="*60)
print("TAULA RESUM — Mean Average Precision (MAP)")
print("="*60)

print(header)
print("-" * len(header))

for cat in SPECIFIC_LABELS:
    row = f"{cat:<25}"
    for v in versions_available:
        val = all_results[v]["map_per_cat"].get(cat, 0.0)
        row += f"  {val:<8.3f}"
    print(row)

print("-" * len(header))
mean_row = f"{'MAP GLOBAL':<25}"
for v in versions_available:
    mean_row += f"  {all_results[v]['map']:<8.3f}"
print(mean_row)

# ── Guarda resultats ──────────────────────────────────────────────────────
output_path = RESULTS_DIR / "evaluation_results.json"
with open(output_path, "w") as f:
    json.dump(all_results, f, indent=2)
print(f"\n✓ Resultats guardats a {output_path}")