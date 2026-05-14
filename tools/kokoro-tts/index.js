/**
 * Kokoro TTS — Pure Frontend Tool
 * Uses kokoro-js (Transformers.js) with onnx-community/Kokoro-82M-v1.0-ONNX
 * Apache 2.0 licensed model weights
 */

import { KokoroTTS } from 'https://esm.sh/kokoro-js@1.2.1';

// ─── Configuration ────────────────────────────────────────────────────────
const MODEL_ID = 'onnx-community/Kokoro-82M-v1.0-ONNX';
const SAMPLE_RATE = 24000;

// ─── Global State ─────────────────────────────────────────────────────────
let tts = null;
let currentAudioUrl = null;
const fileMap = {};      // url/key → { loaded, total, done }
const fileKeys = {};     // url/key → sanitized DOM id

// ─── DOM Refs ─────────────────────────────────────────────────────────────
const statusBadge   = document.getElementById('statusBadge');
const statusMsg     = document.getElementById('statusMsg');
const backendBadge  = document.getElementById('backendBadge');
const progressWrap  = document.getElementById('progressWrap');
const progressFill  = document.getElementById('progressFill');
const progressLabel = document.getElementById('progressLabel');
const progressMeta  = document.getElementById('progressMeta');
const fileList      = document.getElementById('fileList');
const generateBtn   = document.getElementById('generateBtn');
const resultsArea   = document.getElementById('resultsArea');
const cacheSizeEl   = document.getElementById('cacheSize');

// ─── Utility ──────────────────────────────────────────────────────────────
function fmtBytes(bytes) {
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function setStatus(type, msg) {
  statusBadge.className = `status-badge ${type}`;
  statusBadge.textContent = type.toUpperCase();
  statusMsg.textContent = msg;
}

function sanitizeKey(str) {
  return 'f_' + str.replace(/[^a-zA-Z0-9]/g, '_');
}

function addFileDot(key, label) {
  const domId = `dot-${sanitizeKey(key)}`;
  if (document.getElementById(domId)) return;
  fileKeys[key] = domId;
  const item = document.createElement('div');
  item.className = 'model-item';
  item.innerHTML = `<span class="dot loading" id="${domId}"></span>${label}`;
  fileList.appendChild(item);
}

function setDot(key, state) {
  const domId = fileKeys[key] || `dot-${sanitizeKey(key)}`;
  const el = document.getElementById(domId);
  if (el) el.className = `dot ${state}`;
}

// ─── Progress Callback ────────────────────────────────────────────────────
function onProgress(info) {
  const { status, file, loaded, total } = info;
  if (!file) return;

  const shortName = file.split('/').pop() || file;

  if (status === 'initiate' || status === 'download') {
    if (!fileMap[file]) {
      fileMap[file] = { loaded: 0, total: 0, done: false };
      addFileDot(file, shortName);
    }
    progressWrap.classList.remove('hidden');
  }

  if (status === 'progress') {
    if (!fileMap[file]) {
      fileMap[file] = { loaded: 0, total: 0, done: false };
      addFileDot(file, shortName);
      progressWrap.classList.remove('hidden');
    }
    fileMap[file].loaded = loaded || 0;
    fileMap[file].total  = total  || 0;

    const allFiles    = Object.values(fileMap);
    const totalLoaded = allFiles.reduce((s, f) => s + (f.loaded || 0), 0);
    const totalBytes  = allFiles.reduce((s, f) => s + (f.total  || 0), 0);

    if (totalBytes > 0) {
      const pct = Math.min(100, (totalLoaded / totalBytes) * 100).toFixed(1);
      progressFill.style.width  = pct + '%';
      progressLabel.textContent = `Downloading ${shortName}…`;
      progressMeta.textContent  = `${fmtBytes(totalLoaded)} / ${fmtBytes(totalBytes)}`;
    }

    setStatus('loading', `Downloading ${shortName}…`);
  }

  if (status === 'done') {
    if (!fileMap[file]) {
      fileMap[file] = { loaded: 0, total: 0, done: false };
      addFileDot(file, shortName);
    }
    fileMap[file].done = true;
    setDot(file, 'done');
  }
}

// ─── Initialize TTS ───────────────────────────────────────────────────────
async function init() {
  // Detect WebGPU
  let device = 'wasm';
  let dtype  = 'q8';

  if (typeof navigator !== 'undefined' && navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) { device = 'webgpu'; dtype = 'fp32'; }
    } catch (_) { /* fallback to wasm */ }
  }

  backendBadge.textContent = device === 'webgpu' ? 'WebGPU' : 'WebAssembly';
  backendBadge.className   = `backend-badge ${device === 'webgpu' ? 'webgpu' : 'wasm'}`;
  backendBadge.classList.remove('hidden');

  setStatus('loading', `Loading Kokoro-82M (${dtype} · ${device})…`);

  try {
    tts = await KokoroTTS.from_pretrained(MODEL_ID, {
      dtype,
      device,
      progress_callback: onProgress,
    });

    progressFill.style.width = '100%';
    progressWrap.classList.add('hidden');
    setStatus('success', `Ready — ${device === 'webgpu' ? 'WebGPU' : 'WebAssembly'} · ${dtype}`);
    generateBtn.disabled = false;
    updateCacheSize();

  } catch (err) {
    console.error('Model load error:', err);

    // Retry with WASM if WebGPU failed
    if (device === 'webgpu') {
      setStatus('loading', 'WebGPU failed — retrying with WebAssembly…');
      backendBadge.textContent = 'WebAssembly';
      backendBadge.className   = 'backend-badge wasm';
      try {
        tts = await KokoroTTS.from_pretrained(MODEL_ID, {
          dtype: 'q8',
          device: 'wasm',
          progress_callback: onProgress,
        });
        progressFill.style.width = '100%';
        progressWrap.classList.add('hidden');
        setStatus('success', 'Ready — WebAssembly · q8');
        generateBtn.disabled = false;
        updateCacheSize();
        return;
      } catch (err2) {
        console.error('WASM fallback error:', err2);
        setStatus('error', `Load failed: ${err2.message}`);
        Object.keys(fileMap).forEach(k => { if (!fileMap[k].done) setDot(k, 'error'); });
        return;
      }
    }

    setStatus('error', `Load failed: ${err.message}`);
    Object.keys(fileMap).forEach(k => { if (!fileMap[k].done) setDot(k, 'error'); });
  }
}

// ─── Generate Speech ──────────────────────────────────────────────────────
generateBtn.addEventListener('click', async () => {
  if (!tts) return;

  const text  = document.getElementById('textInput').value.trim();
  if (!text) { alert('Please enter some text.'); return; }

  const voice = document.getElementById('voiceSel').value;
  const speed = Math.max(0.5, Math.min(2.0, parseFloat(document.getElementById('speedInput').value) || 1.0));

  generateBtn.disabled     = true;
  generateBtn.textContent  = '⏳ Generating…';
  setStatus('loading', `Generating with ${voice}…`);

  try {
    const t0    = Date.now();
    const audio = await tts.generate(text, { voice, speed });
    const ms    = Date.now() - t0;

    const wavBlob  = float32ToWav(audio.audio, audio.sampling_rate ?? SAMPLE_RATE);
    const durSec   = (audio.audio.length / (audio.sampling_rate ?? SAMPLE_RATE)).toFixed(1);
    const genSec   = (ms / 1000).toFixed(1);
    const rtf      = (ms / 1000 / parseFloat(durSec)).toFixed(2);

    displayResult(wavBlob, text, voice, speed, durSec, genSec, rtf);
    setStatus('success', `Done — ${durSec}s audio generated in ${genSec}s (RTF ${rtf})`);

  } catch (err) {
    console.error('Generation error:', err);
    setStatus('error', `Generation failed: ${err.message}`);
  } finally {
    generateBtn.disabled    = false;
    generateBtn.textContent = '🗣️ Generate Speech';
  }
});

// ─── Display Result ───────────────────────────────────────────────────────
function displayResult(blob, text, voice, speed, durSec, genSec, rtf) {
  if (currentAudioUrl) URL.revokeObjectURL(currentAudioUrl);
  currentAudioUrl = URL.createObjectURL(blob);

  const preview = text.length > 80 ? text.slice(0, 80) + '…' : text;

  resultsArea.innerHTML = `
    <div class="result-item">
      <div class="result-meta">
        <span>🗣️ ${voice}</span>
        <span>⏱ ${durSec}s</span>
        <span>🚀 ${genSec}s gen</span>
        <span>📊 RTF ${rtf}</span>
        <span>⚡ ×${speed}</span>
      </div>
      <div style="font-size:.82rem;color:var(--text-muted);margin-bottom:.25rem;word-break:break-word">"${escapeHtml(preview)}"</div>
      <audio controls src="${currentAudioUrl}"></audio>
      <div class="result-actions">
        <a href="${currentAudioUrl}" download="kokoro-${voice}-${Date.now()}.wav" class="btn btn-ghost">⬇ Download WAV</a>
      </div>
    </div>`;
}

function escapeHtml(str) {
  return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

// ─── Float32 → WAV ────────────────────────────────────────────────────────
function float32ToWav(samples, sampleRate) {
  const numChannels  = 1;
  const bitsPerSample = 16;
  const dataSize     = samples.length * numChannels * (bitsPerSample / 8);
  const buffer       = new ArrayBuffer(44 + dataSize);
  const view         = new DataView(buffer);

  function writeStr(offset, str) {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  }

  // RIFF header
  writeStr(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeStr(8, 'WAVE');

  // fmt chunk
  writeStr(12, 'fmt ');
  view.setUint32(16, 16, true);           // chunk size
  view.setUint16(20, 1, true);            // PCM format
  view.setUint16(22, numChannels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * numChannels * (bitsPerSample / 8), true);
  view.setUint16(32, numChannels * (bitsPerSample / 8), true);
  view.setUint16(34, bitsPerSample, true);

  // data chunk
  writeStr(36, 'data');
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
    offset += 2;
  }

  return new Blob([buffer], { type: 'audio/wav' });
}

// ─── Cache Size ───────────────────────────────────────────────────────────
async function updateCacheSize() {
  if (!('storage' in navigator && 'estimate' in navigator.storage)) return;
  try {
    const { usage } = await navigator.storage.estimate();
    if (usage) cacheSizeEl.textContent = `Cache: ~${fmtBytes(usage)}`;
  } catch (_) { /* ignore */ }
}

// ─── Clear Cache ──────────────────────────────────────────────────────────
window._kokoroClearCache = async function () {
  if (!confirm('Clear all cached model & voice data? The model will need to be re-downloaded on next visit.')) return;
  if ('caches' in window) {
    const keys = await caches.keys();
    await Promise.all(keys.map(k => caches.delete(k)));
  }
  cacheSizeEl.textContent = 'Cache cleared.';
  alert('Cache cleared. Reload the page to re-download the model.');
};

// ─── Boot ─────────────────────────────────────────────────────────────────
init().catch(err => {
  console.error(err);
  setStatus('error', `Startup error: ${err.message}`);
});
