import os
import json
import tempfile
import io
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_file
from pydub import AudioSegment
from pydub.utils import which
import pydub
pydub.AudioSegment.converter = which("ffmpeg")
from search import find_similar, get_versions
import pandas as pd
import requests

app = Flask(__name__)

BASE_DIR = Path("/home/llucsayols/similarity-tool")
WAV_DIR  = Path("/mnt/c/TFG/Dataset/FSD50K.DEV_AUDIO")
DATA_DIR = BASE_DIR / "data"

# Carrega etiquetes
df_labels = pd.read_csv(DATA_DIR / "dev.csv")

CATEGORY_COLORS = {
    "Bass_drum":         "#ff4d4d",
    "Snare_drum":        "#4d9fff",
    "Hi-hat":            "#4dff91",
    "Cymbal":            "#ff9f4d",
    "Crash_cymbal":      "#c04dff",
    "Cowbell":           "#ff4dcc",
    "Tambourine":        "#4dfff0",
    "Clapping":          "#ffdd4d",
    "Mallet_percussion": "#4daaff",
    "Gong":              "#ff7a4d",
}

def get_categories(sound_id):
    row = df_labels[df_labels["fname"].astype(str) == str(sound_id)]
    if row.empty:
        return []
    labels_str = row["labels"].values[0]
    return [cat for cat in CATEGORY_COLORS if cat in labels_str]

@app.route("/")
def index():
    with open(DATA_DIR / "sound_ids_list.json") as f:
        ids = json.load(f)
    versions = get_versions()
    return render_template("index.html", n_sons=len(ids), versions=versions)

@app.route("/versions")
def versions_info():
    return jsonify(get_versions())

@app.route("/search", methods=["POST"])
def search():
    audio_file = request.files.get("audio")
    trim_start = request.form.get("trim_start", type=float)
    trim_end   = request.form.get("trim_end",   type=float)
    top_k      = request.form.get("top_k", 10,  type=int)
    version    = request.form.get("version", "v0")

    if not audio_file:
        return jsonify({"error": "No s'ha rebut cap fitxer"}), 400

    suffix = Path(audio_file.filename).suffix.lower() or ".wav"
    audio_bytes = audio_file.read()

    # Converteix sempre a WAV per assegurar compatibilitat
    try:
        import subprocess
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix or '.webm') as tmp_in:
            tmp_in.write(audio_bytes)
            tmp_in_path = tmp_in.name

        tmp_wav_path = tmp_in_path + '.wav'
        result = subprocess.run(
            ['ffmpeg', '-y', '-i', tmp_in_path, '-ar', '44100', '-ac', '1', tmp_wav_path],
            capture_output=True, text=True
        )
        os.unlink(tmp_in_path)

        if result.returncode != 0:
            return jsonify({"error": f"Error conversió: {result.stderr}"}), 400

        with open(tmp_wav_path, 'rb') as f:
            audio_bytes = f.read()
        os.unlink(tmp_wav_path)
        suffix = '.wav'

    except Exception as e:
        return jsonify({"error": f"Error: {e}"}), 400

    # Retalla si cal
    if trim_start is not None and trim_end is not None and trim_end > trim_start:
        fmt = suffix.replace(".", "")
        seg = AudioSegment.from_file(io.BytesIO(audio_bytes), format=fmt)
        seg = seg[int(trim_start * 1000):int(trim_end * 1000)]
        buf = io.BytesIO()
        seg.export(buf, format="wav")
        audio_bytes = buf.getvalue()
        suffix = ".wav"

    # Guarda temporalment i cerca
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        results = find_similar(tmp_path, top_k=top_k, version=version)
    finally:
        os.unlink(tmp_path)

    response = []
    for sound_id, dist in results:
        cats = get_categories(sound_id)
        wav_exists = (WAV_DIR / f"{sound_id}.wav").exists()
        response.append({
            "id":         sound_id,
            "dist":       round(dist, 4),
            "sim":        round((1 - dist) * 100, 1),
            "cats":       cats,
            "colors":     {c: CATEGORY_COLORS[c] for c in cats},
            "wav_exists": wav_exists,
        })

    return jsonify(response)

@app.route("/audio/<sound_id>")
def serve_audio(sound_id):
    wav_path = WAV_DIR / f"{sound_id}.wav"
    if wav_path.exists():
        return send_file(str(wav_path), mimetype="audio/wav")
    return "Not found", 404

@app.route("/freesound_info/<sound_id>")
def freesound_info(sound_id):
    try:
        r = requests.get(
            f"https://freesound.org/apiv2/sounds/{sound_id}/",
            headers={"Authorization": "Token NsPt7dcZQfrivKdkHTyT1s2HO654coATfQE9pEui"},
            timeout=8
        )
        if r.status_code == 200:
            return jsonify(r.json())
    except Exception:
        pass
    return jsonify({"error": "not found"})

if __name__ == "__main__":
    app.run(debug=True, port=5000)