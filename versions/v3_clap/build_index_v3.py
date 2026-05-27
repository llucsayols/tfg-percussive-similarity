import json
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import torch
from transformers import ClapModel, ClapProcessor
import soundfile as sf
import librosa

BASE_DIR     = Path("/home/llucsayols/similarity-tool")
DATA_DIR     = BASE_DIR / "data"
VERSION_DIR  = BASE_DIR / "versions" / "v3_clap"
WAV_DIR      = Path("/mnt/c/TFG/Dataset/FSD50K.DEV_AUDIO")

# ── Carrega model CLAP ────────────────────────────────────────────────────
print("Carregant model CLAP...")
model     = ClapModel.from_pretrained("laion/clap-htsat-unfused")
processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Model carregat ({device})")
print(f"Embedding size: {model.config.projection_dim}")

# ── Carrega llista de sons ────────────────────────────────────────────────
with open(DATA_DIR / "sound_ids_list.json") as f:
    sound_ids_list = json.load(f)

print(f"\nSons a processar: {len(sound_ids_list)}")

# ── Extreu embeddings ─────────────────────────────────────────────────────
def extract_clap_embedding(wav_path):
    audio_array, sr = librosa.load(str(wav_path), sr=48000, mono=True)

    inputs = processor(
        audio=audio_array,
        sampling_rate=48000,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        result = model.get_audio_features(**inputs)
        embedding = result.pooler_output

    return embedding.cpu().numpy().flatten()

print("\nExtraient embeddings CLAP...")
embeddings = []
errors     = 0
skipped    = 0

for i, sound_id in enumerate(sound_ids_list):
    wav_path = WAV_DIR / f"{sound_id}.wav"
    if not wav_path.exists():
        skipped += 1
        embeddings.append(np.zeros(model.config.projection_dim))
        continue
    try:
        emb = extract_clap_embedding(wav_path)
        embeddings.append(emb)
    except Exception as e:
        print(f"Error {sound_id}: {e}")
        errors += 1
        embeddings.append(np.zeros(model.config.projection_dim))

    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{len(sound_ids_list)} | errors: {errors} | skipped: {skipped}")

embeddings = np.array(embeddings, dtype=np.float32)
print(f"\n✓ Embeddings extrets: {embeddings.shape}")

# ── Normalització ─────────────────────────────────────────────────────────
scaler = StandardScaler()
embeddings_scaled = scaler.fit_transform(embeddings)
print(f"✓ Normalitzat: {embeddings_scaled.shape}")

# ── Guarda ────────────────────────────────────────────────────────────────
np.save(VERSION_DIR / "vectors_scaled.npy", embeddings_scaled)
with open(VERSION_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print(f"\n✓ Tot guardat a {VERSION_DIR}")
print(f"  Sons indexats: {len(sound_ids_list)}")
print(f"  Dimensions:    {embeddings_scaled.shape[1]}")
print(f"  Errors:        {errors}")
print(f"  Skipped:       {skipped}")