import json
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler

BASE_DIR     = Path("/home/llucsayols/similarity-tool")
DATA_DIR     = BASE_DIR / "data"
FEATURES_DIR = DATA_DIR / "features_wav"
VERSION_DIR  = BASE_DIR / "versions" / "v2_personalized"

# ── Descriptors seleccionats manualment ───────────────────────────────────
# Criteri: rellevància perceptiva per a sons percussius curts
# Basada en documentació Essentia + coneixement de domini
# v2.1: afegits erbbands, dissonance i barkbands vectorials
#        per millorar la discriminació de cymbals i crashes

SCALAR_KEYS = [
    # Loudness i dinàmica
    'lowlevel.average_loudness',
    'lowlevel.dynamic_complexity',
    'lowlevel.loudness_ebu128.integrated',
    'lowlevel.loudness_ebu128.loudness_range',
    'lowlevel.loudness_ebu128.momentary.mean',
    'lowlevel.loudness_ebu128.momentary.stdev',
    'lowlevel.loudness_ebu128.short_term.mean',
    'lowlevel.loudness_ebu128.short_term.stdev',
    'lowlevel.spectral_rms.mean',
    'lowlevel.spectral_rms.stdev',
    'lowlevel.spectral_energy.mean',
    'lowlevel.spectral_energy.stdev',

    # Energia per bandes de freqüència
    'lowlevel.spectral_energyband_low.mean',
    'lowlevel.spectral_energyband_low.stdev',
    'lowlevel.spectral_energyband_middle_low.mean',
    'lowlevel.spectral_energyband_middle_low.stdev',
    'lowlevel.spectral_energyband_middle_high.mean',
    'lowlevel.spectral_energyband_middle_high.stdev',
    'lowlevel.spectral_energyband_high.mean',
    'lowlevel.spectral_energyband_high.stdev',

    # Forma espectral — captura el timbre
    'lowlevel.spectral_centroid.mean',
    'lowlevel.spectral_centroid.stdev',
    'lowlevel.spectral_rolloff.mean',
    'lowlevel.spectral_rolloff.stdev',
    'lowlevel.spectral_flux.mean',
    'lowlevel.spectral_flux.stdev',
    'lowlevel.spectral_entropy.mean',
    'lowlevel.spectral_entropy.stdev',
    'lowlevel.spectral_complexity.mean',
    'lowlevel.spectral_complexity.stdev',
    'lowlevel.spectral_strongpeak.mean',
    'lowlevel.spectral_strongpeak.stdev',
    'lowlevel.spectral_decrease.mean',
    'lowlevel.spectral_decrease.stdev',
    'lowlevel.spectral_skewness.mean',
    'lowlevel.spectral_skewness.stdev',
    'lowlevel.spectral_kurtosis.mean',
    'lowlevel.spectral_kurtosis.stdev',

    # Contingut en altes freqüències
    'lowlevel.hfc.mean',
    'lowlevel.hfc.stdev',
    'lowlevel.zerocrossingrate.mean',
    'lowlevel.zerocrossingrate.stdev',

    # Silenci
    'lowlevel.silence_rate_20dB.mean',
    'lowlevel.silence_rate_20dB.stdev',
    'lowlevel.silence_rate_30dB.mean',
    'lowlevel.silence_rate_30dB.stdev',
    'lowlevel.silence_rate_60dB.mean',
    'lowlevel.silence_rate_60dB.stdev',

    # Estadístiques de bandes mel
    'lowlevel.melbands_crest.mean',
    'lowlevel.melbands_crest.stdev',
    'lowlevel.melbands_flatness_db.mean',
    'lowlevel.melbands_flatness_db.stdev',
    'lowlevel.melbands_kurtosis.mean',
    'lowlevel.melbands_kurtosis.stdev',
    'lowlevel.melbands_skewness.mean',
    'lowlevel.melbands_skewness.stdev',
    'lowlevel.melbands_spread.mean',
    'lowlevel.melbands_spread.stdev',

    # Estadístiques de bandes Bark
    'lowlevel.barkbands_crest.mean',
    'lowlevel.barkbands_crest.stdev',
    'lowlevel.barkbands_flatness_db.mean',
    'lowlevel.barkbands_flatness_db.stdev',
    'lowlevel.barkbands_kurtosis.mean',
    'lowlevel.barkbands_kurtosis.stdev',
    'lowlevel.barkbands_skewness.mean',
    'lowlevel.barkbands_skewness.stdev',
    'lowlevel.barkbands_spread.mean',
    'lowlevel.barkbands_spread.stdev',

    # [MOD 2] Dissonància — captura inharmonicitat dels cymbals i crashes
    # Afegit en v2.1 després d'observar limitacions en la cerca de cymbals.
    # Sons molt inarmònics (cymbal, crash) tenen dissonància alta.
    # Sons semi-tonals (cowbell, gong) tenen dissonància baixa.
    'lowlevel.dissonance.mean',
    'lowlevel.dissonance.stdev',

    # Pitch salience
    'lowlevel.pitch_salience.mean',
    'lowlevel.pitch_salience.stdev',

    # Onset rate
    'rhythm.onset_rate',

    # HPCP entropy i crest
    'tonal.hpcp_entropy.mean',
    'tonal.hpcp_entropy.stdev',
    'tonal.hpcp_crest.mean',
    'tonal.hpcp_crest.stdev',
]

VECTOR_KEYS = [
    # MFCC — representació comprimida del timbre (13 coeficients)
    ('lowlevel.mfcc.mean', 13),
    # GFCC — timbre robust al soroll, complementari al MFCC
    ('lowlevel.gfcc.mean', 13),
    # Melbands128 — distribució espectral rica en escala mel (128 bandes)
    ('lowlevel.melbands128.mean', 128),
    ('lowlevel.melbands128.stdev', 128),
    # [MOD 1] ERBbands — millor resolució freqüencial alta per a cymbals
    # Afegit en v2.1: les bandes ERB imiten la còclea amb més precisió
    # a les freqüències altes on els cymbals concentren la seva energia.
    ('lowlevel.erbbands.mean', 40),
    ('lowlevel.erbbands.stdev', 40),
    # [MOD 3] Barkbands — cobertura espectral perceptiva complementària
    # Afegit en v2.1: les bandes Bark aporten major resolució a les
    # freqüències baixes (útil per kicks) complementant les ERBbands.
    ('lowlevel.barkbands.mean', 27),
    ('lowlevel.barkbands.stdev', 27),
    # Spectral contrast — diferència entre pics i valls espectrals per bandes
    ('lowlevel.spectral_contrast_coeffs.mean', 6),
    ('lowlevel.spectral_contrast_coeffs.stdev', 6),
    ('lowlevel.spectral_contrast_valleys.mean', 6),
    ('lowlevel.spectral_contrast_valleys.stdev', 6),
    # HPCP — distribució d'energia per classes de to
    ('tonal.hpcp.mean', 36),
    ('tonal.hpcp.stdev', 36),
]

print(f"Descriptors escalars seleccionats: {len(SCALAR_KEYS)}")
print(f"Descriptors vectorials seleccionats: {len(VECTOR_KEYS)}")
total_dims = len(SCALAR_KEYS) + sum(d for _, d in VECTOR_KEYS)
print(f"Total dimensions esperades: {total_dims}")

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

# ── Carrega llista de sons ────────────────────────────────────────────────
with open(DATA_DIR / "sound_ids_list.json") as f:
    sound_ids_list = json.load(f)

# ── Construeix la matriu ──────────────────────────────────────────────────
print("\nConstruint vectors...")
vectors = []
errors  = 0
for sound_id in sound_ids_list:
    json_path = FEATURES_DIR / f"{sound_id}.json"
    try:
        vec = json_to_vector(json_path, SCALAR_KEYS, VECTOR_KEYS)
        vectors.append(vec)
    except Exception as e:
        print(f"Error {sound_id}: {e}")
        errors += 1

vectors = np.array(vectors, dtype=np.float32)
print(f"✓ Matriu construïda: {vectors.shape}")
if errors:
    print(f"  Errors: {errors}")

# ── Normalització Z-score ─────────────────────────────────────────────────
scaler = StandardScaler()
vectors_scaled = scaler.fit_transform(vectors)
print(f"✓ Normalitzat: {vectors_scaled.shape}")

# ── Guarda ────────────────────────────────────────────────────────────────
np.save(VERSION_DIR / "vectors_scaled.npy", vectors_scaled)
with open(VERSION_DIR / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open(VERSION_DIR / "scalar_keys.json", "w") as f:
    json.dump(SCALAR_KEYS, f, indent=2)
with open(VERSION_DIR / "vector_keys.json", "w") as f:
    json.dump(VECTOR_KEYS, f, indent=2)

print(f"\n✓ Tot guardat a {VERSION_DIR}")
print(f"  Sons indexats: {len(sound_ids_list)}")
print(f"  Dimensions:    {vectors_scaled.shape[1]}")