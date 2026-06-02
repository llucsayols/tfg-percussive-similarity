"""
extract_features_eval.py
Extreu features amb Music Extractor dels WAVs del conjunt d'avaluació (eval set)
que pertanyen a les categories percussives seleccionades.
Necessari per calcular l'onset_rate i filtrar loops al build_test_set.py.
"""

import json
import pandas as pd
from pathlib import Path
import essentia.standard as es

# ── Paths ─────────────────────────────────────────────────────────────────
BASE_DIR          = Path("/home/llucsayols/similarity-tool")
EVAL_WAV_DIR      = Path("/mnt/c/TFG/Dataset/FSD50K.EVAL_AUDIO")
EVAL_CSV          = Path("/mnt/c/TFG/Dataset/eval.csv")
EVAL_FEATURES_DIR = BASE_DIR / "evaluation" / "features_eval"
EVAL_FEATURES_DIR.mkdir(parents=True, exist_ok=True)

SPECIFIC_LABELS = [
    'Bass_drum', 'Snare_drum', 'Hi-hat', 'Cymbal', 'Crash_cymbal',
    'Cowbell', 'Tambourine', 'Clapping', 'Mallet_percussion', 'Gong'
]

# ── Filtra sons percussius del eval set ───────────────────────────────────
print("Carregant eval.csv...")
df = pd.read_csv(EVAL_CSV)
df['fname'] = df['fname'].astype(str)

def has_specific_label(labels_str):
    labels = [l.strip() for l in str(labels_str).split(",")]
    return any(l in SPECIFIC_LABELS for l in labels)

percussive = df[df["labels"].apply(has_specific_label)]
wav_available = [
    row['fname'] for _, row in percussive.iterrows()
    if (EVAL_WAV_DIR / f"{row['fname']}.wav").exists()
]

print(f"Sons percussius al eval set: {len(percussive)}")
print(f"WAVs disponibles:            {len(wav_available)}")

# ── Extracció de features ─────────────────────────────────────────────────
def extract_features(audio_path):
    try:
        features, _ = es.MusicExtractor(
            lowlevelStats=['mean', 'stdev'],
            rhythmStats=['mean', 'stdev'],
            tonalStats=['mean', 'stdev']
        )(str(audio_path))
        return features
    except Exception:
        return None

def features_to_dict(features):
    feature_dict = {}
    for key in features.descriptorNames():
        val = features[key]
        try:
            if hasattr(val, 'tolist'):
                feature_dict[key] = val.tolist()
            else:
                feature_dict[key] = float(val)
        except (ValueError, TypeError):
            feature_dict[key] = str(val)
    return feature_dict

ok, skipped, errors = 0, 0, 0

for i, sound_id in enumerate(wav_available):
    json_path = EVAL_FEATURES_DIR / f"{sound_id}.json"

    if json_path.exists():
        skipped += 1
        continue

    wav_path = EVAL_WAV_DIR / f"{sound_id}.wav"
    features = extract_features(wav_path)

    if features is None:
        errors += 1
        continue

    feature_dict = features_to_dict(features)
    with open(json_path, "w") as f:
        json.dump(feature_dict, f)
    ok += 1

    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(wav_available)} | ok: {ok} | skipped: {skipped} | errors: {errors}")

print(f"\n✓ Completat!")
print(f"  OK:      {ok}")
print(f"  Skipped: {skipped}")
print(f"  Errors:  {errors}")