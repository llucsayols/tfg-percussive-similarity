// ── Estat global ──────────────────────────────────────────────────────────
let currentFile    = null;
let currentBlob    = null;
let trimStart      = null;
let trimEnd        = null;
let wavesurfer     = null;
let activeRegion   = null;
let mediaRecorder  = null;
let recordedChunks = [];
let isRecording    = false;
let currentVersion = 'v0';   // versió activa

const CATEGORY_COLORS = {
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
};

const CATEGORY_LABELS = {
    "Bass_drum":         "Bass Drum",
    "Snare_drum":        "Snare",
    "Hi-hat":            "Hi-hat",
    "Cymbal":            "Cymbal",
    "Crash_cymbal":      "Crash",
    "Cowbell":           "Cowbell",
    "Tambourine":        "Tambourine",
    "Clapping":          "Clapping",
    "Mallet_percussion": "Mallet",
    "Gong":              "Gong",
};

// ── Selector de versió ────────────────────────────────────────────────────
function selectVersion(vid, btn) {
    currentVersion = vid;
    document.querySelectorAll('.version-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');

    // Actualitza el stat de dimensions
    fetch('/versions')
        .then(r => r.json())
        .then(versions => {
            const info = versions[vid];
            if (info) {
                document.getElementById('dims-val').textContent = info.dims;
            }
        });
}

// ── Tabs ──────────────────────────────────────────────────────────────────
function switchTab(name, btn) {
    document.querySelectorAll('.input-mode-tabs .tab').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById('tab-' + name).classList.add('active');
}

// ── Drag & drop / file input ──────────────────────────────────────────────
function handleDrop(e) {
    e.preventDefault();
    document.getElementById('dropzone').classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
}

function handleFile(file) {
    if (!file) return;
    currentFile = file;
    currentBlob = null;
    trimStart = null;
    trimEnd   = null;

    const url = URL.createObjectURL(file);
    document.getElementById('preview-audio').src = url;
    document.getElementById('preview-bar').style.display = 'flex';
    document.getElementById('search-btn').style.display  = 'block';
    initWaveSurfer(url);
}

// ── WaveSurfer ────────────────────────────────────────────────────────────
function initWaveSurfer(url) {
    const container = document.getElementById('waveform-container');
    container.style.display = 'block';

    if (wavesurfer) wavesurfer.destroy();
    activeRegion = null;
    trimStart    = null;
    trimEnd      = null;

    wavesurfer = WaveSurfer.create({
        container: '#waveform',
        waveColor: '#333',
        progressColor: '#e8ff47',
        height: 80,
        scrollParent: true,
        plugins: [
            WaveSurfer.regions.create({
                regionsMinLength: 0.1,
                dragSelection: { slop: 5, color: 'rgba(232,255,71,0.15)' }
            })
        ],
    });

    wavesurfer.load(url);

    wavesurfer.on('ready', () => {
        const dur = wavesurfer.getDuration();
        setWaveInfo(null, null, dur);
        const end = Math.min(2.0, dur);
        wavesurfer.addRegion({
            start: 0, end: end,
            color: 'rgba(232,255,71,0.15)',
            drag: true, resize: true,
        });
        trimStart = 0;
        trimEnd   = end;
    });

    wavesurfer.on('region-created', region => {
        Object.values(wavesurfer.regions.list).forEach(r => {
            if (r.id !== region.id) r.remove();
        });
        activeRegion = region;
        trimStart = region.start;
        trimEnd   = region.end;
        setWaveInfo(region.start, region.end, wavesurfer.getDuration());
    });

    wavesurfer.on('region-updated', region => {
        activeRegion = region;
        trimStart = region.start;
        trimEnd   = region.end;
        setWaveInfo(region.start, region.end, wavesurfer.getDuration());
    });

    wavesurfer.on('region-click', (region, e) => {
        e.stopPropagation();
        region.play();
    });
}

function setWaveInfo(start, end, total) {
    const el = document.getElementById('wave-info');
    if (start === null) {
        el.innerHTML = `Durada total: <span>${total.toFixed(2)}s</span> — arrossega per seleccionar un fragment`;
    } else {
        const dur = (end - start).toFixed(2);
        el.innerHTML = `Selecció: <span>${start.toFixed(2)}s → ${end.toFixed(2)}s</span> (${dur}s)`;
    }
}

function playRegion() {
    if (activeRegion) activeRegion.play();
    else if (wavesurfer) wavesurfer.play();
}

// ── Gravació ──────────────────────────────────────────────────────────────
async function toggleRecord() {
    if (!isRecording) {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder  = new MediaRecorder(stream);
        recordedChunks = [];
        isRecording    = true;

        document.getElementById('rec-btn').classList.add('recording');
        document.getElementById('rec-btn').textContent = '⏹';
        document.getElementById('rec-status').textContent = 'Gravant...';

        mediaRecorder.ondataavailable = e => recordedChunks.push(e.data);
        mediaRecorder.onstop = () => {
            currentBlob = new Blob(recordedChunks, { type: 'audio/webm' });
            currentFile = null;
            trimStart   = null;
            trimEnd     = null;

            const url = URL.createObjectURL(currentBlob);
            document.getElementById('rec-audio').style.display = 'none';
            document.getElementById('preview-audio').src = url;
            document.getElementById('preview-bar').style.display = 'flex';
            document.getElementById('search-btn').style.display  = 'block';
            document.getElementById('waveform-container').style.display = 'block';
            initWaveSurfer(url);

            document.getElementById('rec-btn').classList.remove('recording');
            document.getElementById('rec-btn').textContent = '●';
            document.getElementById('rec-status').textContent = 'Gravació llesta — selecciona un fragment si vols';

            stream.getTracks().forEach(t => t.stop());
            isRecording = false;
        };

        mediaRecorder.start();
    } else {
        mediaRecorder.stop();
    }
}

// ── Cerca ─────────────────────────────────────────────────────────────────
async function doSearch() {
    const topK     = document.getElementById('topk').value;
    const showDist = document.getElementById('dist-toggle').classList.contains('on');

    const formData = new FormData();
    formData.append('top_k', topK);
    formData.append('version', currentVersion);

    if (currentFile) {
        formData.append('audio', currentFile, currentFile.name);
    } else if (currentBlob) {
        formData.append('audio', currentBlob, 'gravacio.wav');
    } else {
        return;
    }

    if (trimStart !== null && trimEnd !== null) {
        formData.append('trim_start', trimStart);
        formData.append('trim_end',   trimEnd);
    }

    document.getElementById('spinner').style.display = 'block';
    document.getElementById('results-section').innerHTML = '';

    try {
        const response = await fetch('/search', { method: 'POST', body: formData });
        const results  = await response.json();
        renderResults(results, showDist);

        const name = currentFile ? currentFile.name : 'gravació.wav';
        addHistory(name, results.length, currentVersion);
    } catch (err) {
        document.getElementById('results-section').innerHTML =
            `<div class="empty-state"><div class="empty-state-label">Error</div><br>${err.message}</div>`;
    } finally {
        document.getElementById('spinner').style.display = 'none';
    }
}

// ── Renderitza resultats ──────────────────────────────────────────────────
function renderResults(results, showDist) {
    const section = document.getElementById('results-section');
    const maxDist = Math.max(...results.map(r => r.dist));

    let html = `<div class="results-header">Top ${results.length} similars — ${currentVersion.toUpperCase()}</div>`;
    html += `<div class="results-grid">`;

    results.forEach((r, i) => {
        const bar  = Math.round((1 - r.dist / Math.max(maxDist, 0.001)) * 100);
        const cats = (r.cats || []).map(c => {
            const color = CATEGORY_COLORS[c] || '#aaa';
            const label = CATEGORY_LABELS[c] || c;
            return `<span class="cat-badge" style="color:${color};border-color:${color};background:${color}22">${label}</span>`;
        }).join('');

        const distHtml = showDist
            ? `<div class="result-dist">dist: ${r.dist} · similitud: ${r.sim}%</div>`
            : '';

        html += `
        <div class="result-card">
            <div id="wf-${r.id}" style="background:#0a0a0a;height:60px;"></div>
            <div class="result-body">
                <div class="result-rank">#${i + 1}</div>
                <div class="result-id">${r.id}</div>
                <div class="result-cats">${cats}</div>
                ${distHtml}
                <div class="bar-bg"><div class="bar-fill" style="width:${bar}%"></div></div>
            </div>
            <div class="result-audio">
                <audio controls src="/audio/${r.id}"></audio>
            </div>
            <div class="result-meta" id="meta-${r.id}">
                <div style="font-family:monospace;font-size:0.65rem;color:#666;">Carregant info...</div>
            </div>
        </div>`;
    });

    html += `</div>`;
    section.innerHTML = html;

    results.forEach(r => {
        renderMiniWave(r.id);
        loadFreesoundInfo(r.id);
    });
}

function renderMiniWave(soundId) {
    const ws = WaveSurfer.create({
        container: `#wf-${soundId}`,
        waveColor: '#333',
        progressColor: '#e8ff47',
        height: 60,
        interact: false,
    });
    ws.load(`/audio/${soundId}`);
}

async function loadFreesoundInfo(soundId) {
    try {
        const r    = await fetch(`/freesound_info/${soundId}`);
        const info = await r.json();
        if (!info || info.error) return;

        const dur   = info.duration ? info.duration.toFixed(2) : '—';
        const sr    = info.samplerate || '—';
        const bd    = info.bitdepth || '—';
        const ch    = info.channels === 1 ? 'Mono' : 'Stereo';
        const size  = info.filesize ? (info.filesize / 1024).toFixed(0) + ' KB' : '—';
        const tags  = (info.tags || []).slice(0, 4).map(t => `<span class="tag">${t}</span>`).join('');
        const user  = info.username || '';
        const fsUrl = `https://freesound.org/people/${user}/sounds/${soundId}/`;

        document.getElementById(`meta-${soundId}`).innerHTML = `
            <div style="font-family:monospace;font-size:0.62rem;color:#666;margin-bottom:0.4rem;">
                ${dur}s · ${sr}Hz · ${bd}bit · ${ch} · ${size}
            </div>
            <div class="result-tags">${tags}</div>
            <a href="${fsUrl}" target="_blank" class="result-link">↗ Veure a Freesound</a>
        `;
    } catch {}
}

// ── Historial ─────────────────────────────────────────────────────────────
function addHistory(name, n, version) {
    const list = document.getElementById('history-list');
    const item = document.createElement('div');
    item.className = 'history-item';
    item.textContent = `[${version}] ${name} (${n})`;
    list.prepend(item);
    while (list.children.length > 8) list.removeChild(list.lastChild);
}