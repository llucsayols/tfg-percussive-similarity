import json
import numpy as np
from pathlib import Path
import essentia.standard as es
from sklearn.preprocessing import StandardScaler
import pickle

BASE_DIR     = Path("/home/llucsayols/similarity-tool")
WAV_DIR      = Path("/mnt/c/TFG/Dataset/FSD50K.DEV_AUDIO")
DATA_DIR     = BASE_DIR / "data"
FEATURES_DIR = DATA_DIR / "features_wav"

# Carrega IDs disponibles
with open(DATA_DIR / "available_ids.json") as f:
    available = set(json.load(f))

# Comprova quants WAVs tenim
wav_available = [sid for sid in available if (WAV_DIR / f"{sid}.wav").exists()]
print(f"WAVs disponibles: {len(wav_available)}")

def extract_features(audio_path):
    try:
        features, _ = es.MusicExtractor(
            lowlevelStats=['mean', 'stdev'],
            rhythmStats=['mean', 'stdev'],
            tonalStats=['mean', 'stdev']
        )(str(audio_path))
        return features
    except Exception as e:
        print(f"Error: {audio_path} → {e}")
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

# Processa tots els WAVs
FEATURES_DIR.mkdir(exist_ok=True)

ok, skipped, errors = 0, 0, 0
for i, sound_id in enumerate(wav_available):
    json_path = FEATURES_DIR / f"{sound_id}.json"

    if json_path.exists():
        skipped += 1
        continue

    wav_path = WAV_DIR / f"{sound_id}.wav"
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