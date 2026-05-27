# Comparació de Representacions Vectorials d'Àudio per a la Cerca per Similitud en Samples Percussius

## Implementació d'una Eina per a Productors Musicals

Sistema que permet a productors musicals trobar sons percussius curts similars a un so de referència, comparant tres mètodes de representació vectorial d'àudio.

Treball Fi de Grau — Enginyeria Audiovisual Computacional, Universitat Pompeu Fabra  
Autor: Lluc Sayols Hidalgo  
Tutor: Xavier Serra (Music Technology Group, UPF)

---

## Requisits del sistema

- **Windows 10/11** amb WSL2 (Ubuntu)
- **Python 3.11** (instal·lat dins de WSL)
- **ffmpeg** (instal·lat dins de WSL)
- Dataset **FSD50K** descomprimit localment

---

## Instal·lació

### 1. Clona el repositori

```bash
git clone https://github.com/llucsayols/tfg-percussive-similarity-tool.git
cd tfg-percussive-similarity-tool
```

### 2. Crea i activa l'entorn virtual

Es recomana usar el nom `venv` o qualsevol altre nom estàndard:

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Instal·la les dependències

```bash
pip install essentia flask pydub scipy scikit-learn numpy pandas requests librosa transformers torch torchaudio soundfile faiss-cpu
```

### 4. Instal·la ffmpeg

```bash
sudo apt update
sudo apt install ffmpeg -y
```

---

## Preparació del dataset

### 1. Descarrega FSD50K

Descarrega el dataset FSD50K des de [Zenodo](https://zenodo.org/record/4060432) i descomprimeix els WAVs de desenvolupament en qualsevol carpeta del teu sistema.

### 2. Actualitza el path dels WAVs

Als scripts `filter_dataset.py`, `extract_features.py` i `app_flask.py`, actualitza la variable `WAV_DIR` perquè apunti a la carpeta on tens els WAVs de FSD50K:

```python
WAV_DIR = Path("/ruta/al/teu/FSD50K.DEV_AUDIO")
```

---

## Pipeline d'execució

Executa els scripts en aquest ordre:

### Pas 1 — Filtra els sons percussius del dataset

```bash
python filter_dataset.py
```

Genera `data/available_ids.json` amb els IDs dels sons percussius filtrats.

### Pas 2 — Extreu les features amb Music Extractor

```bash
python extract_features.py
```

Extreu les features de cada WAV i les guarda com a JSONs a `data/features_wav/`. Pot trigar 30-60 minuts.

### Pas 3 — Construeix els índexs de les versions

Executa cada versió per separat:

```bash
# v0 — Music Extractor complet
python versions/v0_music_extractor_complet/build_index_v0.py

# v1 — Selecció automàtica (VarianceThreshold)
python versions/v1_variance_threshold/build_index_v1.py

# v2 — Selecció manual per domini
python versions/v2_personalized/build_index_v2.py

# v3 — CLAP (deep learning) — pot trigar 30-60 minuts
python versions/v3_clap/build_index_v3.py
```

---

## Execució de la interfície

```bash
python app_flask.py
```

Obre el navegador a `http://127.0.0.1:5000`

---

## Estructura del repositori

```
similarity-tool/
├── data/
│   ├── features_wav/          # JSONs amb features (generats per extract_features.py)
│   ├── available_ids.json     # IDs filtrats (generat per filter_dataset.py)
│   ├── dev.csv                # Metadades FSD50K
│   └── sound_ids_list.json    # IDs indexats en ordre
├── versions/
│   ├── v0_music_extractor_complet/   # Music Extractor complet (1022 dims)
│   ├── v1_variance_threshold/        # VarianceThreshold automàtic (834 dims)
│   ├── v2_personalized/              # Selecció manual (589 dims)
│   └── v3_clap/                      # CLAP deep learning (512 dims)
├── static/
│   ├── app.js                 # Lògica JavaScript (WaveSurfer.js)
│   └── styles.css             # Estils de la interfície
├── templates/
│   └── index.html             # Interfície web
├── app_flask.py               # Servidor Flask (backend)
├── search.py                  # Mòdul de cerca per similitud
├── extract_features.py        # Extracció de features amb Music Extractor
└── filter_dataset.py          # Filtratge de sons percussius del dataset
```

---

## Notes

- Els fitxers de dades grans (`features_wav/`, `vectors_scaled.npy`, `scaler.pkl`) no s'inclouen al repositori per limitacions de mida. Cal generar-los localment seguint la pipeline d'execució.
- El model CLAP (v3) es descarrega automàticament de HuggingFace la primera vegada que s'executa `build_index_v3.py`.
- La interfície requereix connexió a Internet per carregar les imatges de forma d'ona de Freesound.
