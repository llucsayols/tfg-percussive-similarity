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
- **Essentia** (Music Technology Group, UPF) — llibreria principal d'extracció de features
- Dataset **FSD50K** descomprimit localment

---

## Instal·lació

### 1. Clona el repositori

```bash
git clone https://github.com/llucsayols/tfg-percussive-similarity-tool.git
cd tfg-percussive-similarity-tool
```

### 2. Crea i activa l'entorn virtual

```bash
python3.11 -m venv venv
source venv/bin/activate
```

### 3. Instal·la les dependències

```bash
pip install -r requirements.txt
```

### 4. Instal·la ffmpeg

```bash
sudo apt update
sudo apt install ffmpeg -y
```

---

## Preparació del dataset

### 1. Descarrega FSD50K

Descarrega el dataset FSD50K des de [Zenodo](https://zenodo.org/record/4060432) i descomprimeix els WAVs de desenvolupament i avaluació.

### 2. Actualitza els paths dels WAVs

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

```bash
python versions/v0_music_extractor_complet/build_index_v0.py
python versions/v1_variance_threshold/build_index_v1.py
python versions/v2_personalized/build_index_v2.py
python versions/v3_clap/build_index_v3.py   # pot trigar 30-60 minuts
```

---

## Execució de la interfície

```bash
python app_flask.py
```

Obre el navegador a `http://127.0.0.1:5000`

---

## Avaluació

### Construir el test set (sons del FSD50K eval set)

```bash
python evaluation/extract_features_eval.py
python evaluation/build_test_set.py
```

### Executar l'avaluació semàntica (Precision@10 i MAP)

```bash
python evaluation/evaluate_precision.py
```

---

## Estructura del repositori

```
similarity-tool/
├── data/
│   ├── features_wav/          # JSONs amb features per cada so
│   ├── available_ids.json     # IDs filtrats
│   ├── dev.csv                # Metadades FSD50K
│   └── sound_ids_list.json    # 2526 IDs indexats
├── evaluation/
│   ├── features_eval/         # Features del conjunt d'avaluació
│   ├── results/               # Resultats (P@10, MAP)
│   ├── build_test_set.py      # Generació del test set
│   ├── evaluate_precision.py  # Avaluació semàntica
│   └── extract_features_eval.py
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
├── app_flask.py               # Servidor Flask
├── search.py                  # Mòdul de cerca per similitud
├── extract_features.py        # Extracció de features
├── filter_dataset.py          # Filtratge del dataset
└── requirements.txt           # Dependències del projecte
```

---

## Notes

- Els fitxers de dades grans (`features_wav/`, `vectors_scaled.npy`, `scaler.pkl`) no s'inclouen al repositori per limitacions de mida. Cal generar-los localment seguint la pipeline d'execució.
- El model CLAP (v3) es descarrega automàticament de HuggingFace la primera vegada que s'executa `build_index_v3.py`.
- La interfície requereix connexió a Internet per obtenir les metadades dels sons de Freesound.