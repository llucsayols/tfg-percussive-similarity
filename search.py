import json
import numpy as np
import pickle
from pathlib import Path
from scipy.spatial.distance import cdist
import essentia.standard as es

BASE_DIR = Path("/home/llucsayols/similarity-tool")
DATA_DIR = BASE_DIR / "data"

# ── Configuració de versions ──────────────────────────────────────────────
VERSIONS = {
    "v0": {
        "name":   "Music Extractor complet",
        "dir":    BASE_DIR / "versions" / "v0_music_extractor_complet",
        "type":   "music_extractor",
    },
    "v1": {
        "name":   "VarianceThreshold automàtic",
        "dir":    BASE_DIR / "versions" / "v1_variance_threshold",
        "type":   "music_extractor_selected",
    },
    "v2": {
        "name":   "Selecció manual per domini",
        "dir":    BASE_DIR / "versions" / "v2_personalized",
        "type":   "music_extractor",
    },
    "v3": {
        "name":   "CLAP (deep learning)",
        "dir":    BASE_DIR / "versions" / "v3_clap",
        "type":   "clap",
    },
}

# ── Llista de sons compartida ─────────────────────────────────────────────
with open(DATA_DIR / "sound_ids_list.json") as f:
    sound_ids_list = json.load(f)


# ── Cache d'índexs carregats (per no recarregar a cada cerca) ────────────
_loaded_indexes = {}

def _load_index(version):
    """Carrega els fitxers d'una versió i els emmagatzema a cache."""
    if version in _loaded_indexes:
        return _loaded_indexes[version]

    if version not in VERSIONS:
        raise ValueError(f"Versió desconeguda: {version}. Opcions: {list(VERSIONS.keys())}")

    cfg = VERSIONS[version]
    version_dir = cfg["dir"]

    data = {
        "type":    cfg["type"],
        "vectors": np.load(version_dir / "vectors_scaled.npy"),
    }

    with open(version_dir / "scaler.pkl", "rb") as f:
        data["scaler"] = pickle.load(f)

    # Versions basades en Music Extractor necessiten les keys per construir el vector query
    if cfg["type"] in ("music_extractor", "music_extractor_selected"):
        with open(VERSIONS["v0"]["dir"] / "scalar_keys.json") as f:
            base_scalar = json.load(f)
        with open(VERSIONS["v0"]["dir"] / "vector_keys.json") as f:
            base_vector = json.load(f)
        data["base_scalar_keys"] = base_scalar
        data["base_vector_keys"] = base_vector

    # v1 té un selector addicional (VarianceThreshold)
    if cfg["type"] == "music_extractor_selected":
        with open(version_dir / "selector.pkl", "rb") as f:
            data["selector"] = pickle.load(f)

    # v2 té les seves pròpies keys de selecció manual
    if version == "v2":
        with open(version_dir / "scalar_keys.json") as f:
            data["scalar_keys"] = json.load(f)
        with open(version_dir / "vector_keys.json") as f:
            data["vector_keys"] = json.load(f)

    _loaded_indexes[version] = data
    return data


# ── Helpers ───────────────────────────────────────────────────────────────
def flatten(val):
    if isinstance(val, list):
        result = []
        for v in val:
            result.extend(flatten(v))
        return result
    return [float(val)]


def _extract_music_extractor_features(audio_path):
    """Extreu features amb Music Extractor i retorna el diccionari."""
    features, _ = es.MusicExtractor(
        lowlevelStats=['mean', 'stdev'],
        rhythmStats=['mean', 'stdev'],
        tonalStats=['mean', 'stdev']
    )(str(audio_path))

    feature_dict = {}
    for key in features.descriptorNames():
        val = features[key]
        try:
            feature_dict[key] = val.tolist() if hasattr(val, 'tolist') else float(val)
        except (ValueError, TypeError):
            feature_dict[key] = str(val)
    return feature_dict


def _build_vector_from_features(feature_dict, scalar_keys, vector_keys):
    """Construeix un vector a partir de les features i les keys especificades."""
    vector = []
    for key in scalar_keys:
        vector.append(float(feature_dict.get(key, 0.0)))
    for key, _ in vector_keys:
        vals = feature_dict.get(key, [])
        vector.extend(flatten(vals))
    return np.array(vector, dtype=np.float32).reshape(1, -1)


def _audio_to_vector_clap(audio_path, data):
    """Extreu embedding CLAP de l'àudio."""
    # Import dins de la funció per no carregar CLAP si no s'usa
    import librosa
    import torch
    from transformers import ClapModel, ClapProcessor

    if not hasattr(_audio_to_vector_clap, "model"):
        print("Carregant model CLAP...")
        _audio_to_vector_clap.model     = ClapModel.from_pretrained("laion/clap-htsat-unfused")
        _audio_to_vector_clap.processor = ClapProcessor.from_pretrained("laion/clap-htsat-unfused")
        _audio_to_vector_clap.model.eval()

    model     = _audio_to_vector_clap.model
    processor = _audio_to_vector_clap.processor

    audio_array, sr = librosa.load(str(audio_path), sr=48000, mono=True)
    inputs = processor(audio=audio_array, sampling_rate=48000, return_tensors="pt")

    with torch.no_grad():
        result = model.get_audio_features(**inputs)
        embedding = result.pooler_output

    vec = embedding.cpu().numpy().reshape(1, -1)
    vec_scaled = data["scaler"].transform(vec)
    return vec_scaled


def _audio_to_vector(audio_path, version):
    """Genera el vector de l'àudio query segons la versió."""
    data = _load_index(version)

    if data["type"] == "clap":
        return _audio_to_vector_clap(audio_path, data)

    # Music Extractor pipeline
    feature_dict = _extract_music_extractor_features(audio_path)

    if version == "v2":
        # v2 té keys pròpies de selecció manual
        vec = _build_vector_from_features(
            feature_dict,
            data["scalar_keys"],
            data["vector_keys"]
        )
    else:
        # v0 i v1 usen les keys de v0 com a base
        vec = _build_vector_from_features(
            feature_dict,
            data["base_scalar_keys"],
            data["base_vector_keys"]
        )

    # Normalitza amb el scaler corresponent
    vec_scaled = data["scaler"].transform(vec)

    # v1 aplica selecció de VarianceThreshold després de normalitzar
    if version == "v1":
        vec_scaled = data["selector"].transform(vec_scaled)

    return vec_scaled


# ── API pública ───────────────────────────────────────────────────────────
def find_similar(audio_path, top_k=5, version="v0"):
    """Donada la ruta d'un àudio, retorna els top_k sons més similars
    segons la versió de representació especificada (v0, v1, v2, v3)."""
    data = _load_index(version)
    query_vec = _audio_to_vector(audio_path, version)
    distances = cdist(query_vec, data["vectors"], metric="cosine")[0]
    sorted_idx = np.argsort(distances)
    return [
        (sound_ids_list[i], float(distances[i]))
        for i in sorted_idx[:top_k]
    ]


def get_versions():
    """Retorna informació de les versions disponibles per a la interfície."""
    info = {}
    for vid, cfg in VERSIONS.items():
        try:
            vecs = np.load(cfg["dir"] / "vectors_scaled.npy")
            info[vid] = {
                "name": cfg["name"],
                "dims": vecs.shape[1],
            }
        except Exception:
            info[vid] = {
                "name": cfg["name"] + " (no disponible)",
                "dims": 0,
            }
    return info