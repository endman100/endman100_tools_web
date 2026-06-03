import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.webgpu.min.mjs';
import { AutoTokenizer, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2';

const HF_BASE = 'https://huggingface.co/endman100/Irodori-TTS-600M-v3-VoiceDesign-onnx/resolve/main';
const TOKENIZER_ID = 'llm-jp/llm-jp-3-150m';
const DB_NAME = 'irodori-voicedesign-v3-onnx-cache';
const STORE_NAME = 'files';
const CACHE_NAME = 'irodori-voicedesign-v3-onnx-files';
const OPFS_DIR = 'irodori-voicedesign-v3-onnx-files-v1';
const SAMPLE_RATE = 48000;
const HOP_LENGTH = 1920;
const LATENT_DIM = 32;
const HEAD_DIM = 64;
const DOWNLOAD_CHUNK_SIZE = 32 * 1024 * 1024;

const MODEL_DEFS = [
  { id: 'text', label: 'Text Encoder', file: 'text_encoder.onnx', size: '338 MB' },
  { id: 'caption', label: 'Caption Encoder', file: 'caption_encoder.onnx', size: '338 MB' },
  { id: 'speaker', label: 'Speaker Encoder', file: 'speaker_encoder.onnx', size: '242 MB' },
  { id: 'dit', label: 'VoiceDesign Diffusion Transformer Step', file: 'dit_step.onnx', size: '1.8 GB' },
  { id: 'decoder', label: 'DACVAE Decoder', file: 'dacvae_decoder.onnx', size: '283 MB' },
];

const OPTIONAL_MODEL_DEFS = {
  encoder: { id: 'encoder', label: 'DACVAE Encoder', file: 'dacvae_encoder.onnx', size: '109 MB' },
  duration: { id: 'duration', label: 'Duration Predictor', file: 'duration_predictor.onnx', size: '68 MB' },
};

let db = null;
let tokenizer = null;
let sessions = null;
let currentAudioUrl = null;
let backendMode = localStorage.getItem('irodori-backend-mode') || 'gpu';
const preloadedBuffers = new Map();
const downloadPromises = new Map();

const refs = {
  statusBadge: document.getElementById('statusBadge'),
  statusMsg: document.getElementById('statusMsg'),
  progressWrap: document.getElementById('progressWrap'),
  progressFill: document.getElementById('progressFill'),
  progressLabel: document.getElementById('progressLabel'),
  progressMeta: document.getElementById('progressMeta'),
  generateBtn: document.getElementById('generateBtn'),
  resetBtn: document.getElementById('resetBtn'),
  resultsArea: document.getElementById('resultsArea'),
  logCard: document.getElementById('logCard'),
  runLog: document.getElementById('runLog'),
  dotRuntime: document.getElementById('dot-runtime'),
  dotModels: document.getElementById('dot-models'),
  dotOutput: document.getElementById('dot-output'),
  backendBadge: document.getElementById('backendBadge'),
  backendToggle: document.getElementById('backendToggle'),
  cacheClearBtn: document.getElementById('cacheClearBtn'),
  textInput: document.getElementById('textInput'),
  captionInput: document.getElementById('captionInput'),
  referenceAudio: document.getElementById('referenceAudio'),
  numStepsInput: document.getElementById('numStepsInput'),
  numCandidatesInput: document.getElementById('numCandidatesInput'),
  seedInput: document.getElementById('seedInput'),
  secondsInput: document.getElementById('secondsInput'),
  durationScaleInput: document.getElementById('durationScaleInput'),
  maxRefSecondsInput: document.getElementById('maxRefSecondsInput'),
  refNormalizeDbInput: document.getElementById('refNormalizeDbInput'),
  refEnsureMaxInput: document.getElementById('refEnsureMaxInput'),
  tScheduleModeSel: document.getElementById('tScheduleModeSel'),
  swayCoeffInput: document.getElementById('swayCoeffInput'),
  cfgModeSel: document.getElementById('cfgModeSel'),
  cfgScaleTextInput: document.getElementById('cfgScaleTextInput'),
  cfgScaleCaptionInput: document.getElementById('cfgScaleCaptionInput'),
  cfgScaleSpeakerInput: document.getElementById('cfgScaleSpeakerInput'),
  cfgOverrideInput: document.getElementById('cfgOverrideInput'),
  cfgMinTInput: document.getElementById('cfgMinTInput'),
  cfgMaxTInput: document.getElementById('cfgMaxTInput'),
  contextKvCacheInput: document.getElementById('contextKvCacheInput'),
  maxTextLenInput: document.getElementById('maxTextLenInput'),
  maxCaptionLenInput: document.getElementById('maxCaptionLenInput'),
  truncationFactorInput: document.getElementById('truncationFactorInput'),
  rescaleKInput: document.getElementById('rescaleKInput'),
  rescaleSigmaInput: document.getElementById('rescaleSigmaInput'),
  speakerKvScaleInput: document.getElementById('speakerKvScaleInput'),
  speakerKvMinTInput: document.getElementById('speakerKvMinTInput'),
  speakerKvMaxLayersInput: document.getElementById('speakerKvMaxLayersInput'),
  speakerUncondModeSel: document.getElementById('speakerUncondModeSel'),
  trimTailInput: document.getElementById('trimTailInput'),
  tailWindowSizeInput: document.getElementById('tailWindowSizeInput'),
  tailStdThresholdInput: document.getElementById('tailStdThresholdInput'),
  tailMeanThresholdInput: document.getElementById('tailMeanThresholdInput'),
};

const DEFAULTS = {
  numSteps: '40',
  numCandidates: '1',
  seed: '',
  seconds: '',
  durationScale: '1.0',
  maxRefSeconds: '30',
  refNormalizeDb: '-16',
  refEnsureMax: true,
  tScheduleMode: 'linear',
  swayCoeff: '-1.0',
  cfgMode: 'independent',
  cfgScaleText: '3.0',
  cfgScaleCaption: '4.0',
  cfgScaleSpeaker: '5.0',
  cfgOverride: '',
  cfgMinT: '0.5',
  cfgMaxT: '1',
  contextKvCache: true,
  maxTextLen: '',
  maxCaptionLen: '',
  truncationFactor: '',
  rescaleK: '',
  rescaleSigma: '',
  speakerKvScale: '',
  speakerKvMinT: '0.9',
  speakerKvMaxLayers: '',
  speakerUncondMode: 'mask',
  trimTail: true,
  tailWindowSize: '20',
  tailStdThreshold: '0.05',
  tailMeanThreshold: '0.1',
};

function setStatus(type, message) {
  refs.statusBadge.className = `status-badge ${type}`;
  refs.statusBadge.textContent = type.toUpperCase();
  refs.statusMsg.textContent = message;
}

function setDot(dot, state) {
  if (dot) dot.className = `dot ${state}`;
}

function fmtBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '';
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function clampNumber(input, fallback, min, max) {
  const value = Number.parseFloat(input.value);
  if (!Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, value));
}

function clampInteger(input, fallback, min, max) {
  const value = Number.parseInt(input.value, 10);
  if (!Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, value));
}

function optionalNumber(input, min = -Infinity, max = Infinity) {
  const raw = input.value.trim().toLowerCase();
  if (!raw || ['none', 'null', 'off', 'disable', 'disabled'].includes(raw)) return null;
  const value = Number.parseFloat(raw);
  if (!Number.isFinite(value)) throw new Error(`${input.labels?.[0]?.textContent || 'Optional number'} must be finite.`);
  return Math.min(max, Math.max(min, value));
}

function optionalInteger(input, min = -Infinity, max = Infinity) {
  const raw = input.value.trim().toLowerCase();
  if (!raw || ['none', 'null', 'off', 'disable', 'disabled'].includes(raw)) return null;
  const value = Number.parseInt(raw, 10);
  if (!Number.isFinite(value)) throw new Error(`${input.labels?.[0]?.textContent || 'Optional integer'} must be an integer.`);
  return Math.min(max, Math.max(min, value));
}

function hasSessionInput(session, name) {
  return Array.isArray(session.inputNames) && session.inputNames.includes(name);
}

async function runSession(session, feeds) {
  if (!Array.isArray(session.inputNames) || session.inputNames.length === 0) {
    return session.run(feeds);
  }
  const filtered = {};
  for (const name of session.inputNames) {
    if (feeds[name] !== undefined && feeds[name] !== null) filtered[name] = feeds[name];
  }
  const missing = session.inputNames.filter(name => filtered[name] === undefined);
  if (missing.length) {
    throw new Error(`ONNX input missing for ${session.inputNames.join(', ')}: ${missing.join(', ')}`);
  }
  return session.run(filtered);
}

function pickOutput(outputs, names) {
  for (const name of names) {
    if (outputs[name]) return outputs[name];
  }
  const first = Object.values(outputs)[0];
  if (!first) throw new Error(`ONNX output missing. Expected one of: ${names.join(', ')}`);
  return first;
}

function resetParams() {
  refs.referenceAudio.value = '';
  refs.numStepsInput.value = DEFAULTS.numSteps;
  refs.numCandidatesInput.value = DEFAULTS.numCandidates;
  refs.seedInput.value = DEFAULTS.seed;
  refs.secondsInput.value = DEFAULTS.seconds;
  refs.durationScaleInput.value = DEFAULTS.durationScale;
  refs.maxRefSecondsInput.value = DEFAULTS.maxRefSeconds;
  refs.refNormalizeDbInput.value = DEFAULTS.refNormalizeDb;
  refs.refEnsureMaxInput.checked = DEFAULTS.refEnsureMax;
  refs.tScheduleModeSel.value = DEFAULTS.tScheduleMode;
  refs.swayCoeffInput.value = DEFAULTS.swayCoeff;
  refs.cfgModeSel.value = DEFAULTS.cfgMode;
  refs.cfgScaleTextInput.value = DEFAULTS.cfgScaleText;
  refs.cfgScaleCaptionInput.value = DEFAULTS.cfgScaleCaption;
  refs.cfgScaleSpeakerInput.value = DEFAULTS.cfgScaleSpeaker;
  refs.cfgOverrideInput.value = DEFAULTS.cfgOverride;
  refs.cfgMinTInput.value = DEFAULTS.cfgMinT;
  refs.cfgMaxTInput.value = DEFAULTS.cfgMaxT;
  refs.contextKvCacheInput.checked = DEFAULTS.contextKvCache;
  refs.maxTextLenInput.value = DEFAULTS.maxTextLen;
  refs.maxCaptionLenInput.value = DEFAULTS.maxCaptionLen;
  refs.truncationFactorInput.value = DEFAULTS.truncationFactor;
  refs.rescaleKInput.value = DEFAULTS.rescaleK;
  refs.rescaleSigmaInput.value = DEFAULTS.rescaleSigma;
  refs.speakerKvScaleInput.value = DEFAULTS.speakerKvScale;
  refs.speakerKvMinTInput.value = DEFAULTS.speakerKvMinT;
  refs.speakerKvMaxLayersInput.value = DEFAULTS.speakerKvMaxLayers;
  refs.speakerUncondModeSel.value = DEFAULTS.speakerUncondMode;
  refs.trimTailInput.checked = DEFAULTS.trimTail;
  refs.tailWindowSizeInput.value = DEFAULTS.tailWindowSize;
  refs.tailStdThresholdInput.value = DEFAULTS.tailStdThreshold;
  refs.tailMeanThresholdInput.value = DEFAULTS.tailMeanThreshold;
}

function activeBackendLabel() {
  return backendMode === 'gpu' && navigator.gpu ? 'GPU (WebGPU)' : 'CPU (WASM)';
}

function updateBackendUI() {
  if (backendMode === 'gpu' && !navigator.gpu) backendMode = 'cpu';
  refs.backendBadge.textContent = activeBackendLabel();
  refs.backendToggle.disabled = !navigator.gpu || refs.generateBtn.disabled;
  refs.backendToggle.textContent = backendMode === 'gpu' ? '切換至 CPU' : '切換至 GPU';
  refs.cacheClearBtn.disabled = refs.generateBtn.disabled;
  refs.swayCoeffInput.disabled = refs.tScheduleModeSel.value !== 'sway';
}

function switchBackend() {
  if (!navigator.gpu) return;
  backendMode = backendMode === 'gpu' ? 'cpu' : 'gpu';
  localStorage.setItem('irodori-backend-mode', backendMode);
  sessions = null;
  updateBackendUI();
  setDot(refs.dotOutput, 'pending');
  setStatus('success', `Backend switched to ${activeBackendLabel()} — click Generate`);
}

function deleteIndexedDB(name) {
  return new Promise(resolve => {
    try {
      const req = indexedDB.deleteDatabase(name);
      req.onsuccess = () => resolve();
      req.onerror = () => resolve();
      req.onblocked = () => resolve();
    } catch {
      resolve();
    }
  });
}

async function clearOpfsCache() {
  if (!navigator.storage?.getDirectory) return;
  try {
    const root = await navigator.storage.getDirectory();
    await root.removeEntry(OPFS_DIR, { recursive: true });
  } catch {
    // Missing OPFS cache is already a cleared state.
  }
}

async function clearModelCache() {
  if (refs.generateBtn.disabled) return;
  if (!window.confirm('清除已下載的 Irodori VoiceDesign ONNX 模型快取？清除後會立即重新下載。')) return;

  sessions = null;
  preloadedBuffers.clear();
  downloadPromises.clear();
  if (db) {
    db.close();
    db = null;
  }

  refs.generateBtn.disabled = true;
  updateBackendUI();
  setDot(refs.dotModels, 'loading');
  setDot(refs.dotOutput, 'pending');
  setStatus('loading', 'Clearing downloaded model cache…');
  refs.progressWrap.classList.remove('hidden');
  refs.progressLabel.textContent = 'Clearing OPFS / Cache API / IndexedDB';
  refs.progressMeta.textContent = '';
  refs.progressFill.style.width = '35%';

  try {
    await clearOpfsCache();
    if ('caches' in window) await caches.delete(CACHE_NAME);
    await deleteIndexedDB(DB_NAME);
    refs.progressFill.style.width = '100%';
    setStatus('loading', 'Cache cleared — downloading models again…');
    await preloadModelFiles();
  } catch (error) {
    console.error(error);
    refs.progressWrap.classList.add('hidden');
    setDot(refs.dotModels, 'error');
    setStatus('error', `Cache clear failed: ${error.message}`);
    refs.generateBtn.disabled = false;
    refs.generateBtn.textContent = '🎙️ Generate';
    updateBackendUI();
  }
}

function openDB() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = event => event.target.result.createObjectStore(STORE_NAME);
    req.onsuccess = event => resolve(event.target.result);
    req.onerror = event => reject(event.target.error);
  });
}

function dbGet(key) {
  return new Promise(resolve => {
    try {
      const req = db.transaction(STORE_NAME, 'readonly').objectStore(STORE_NAME).get(key);
      req.onsuccess = () => resolve(req.result ?? null);
      req.onerror = () => resolve(null);
    } catch {
      resolve(null);
    }
  });
}

function dbPut(key, value) {
  return new Promise((resolve, reject) => {
    try {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      tx.objectStore(STORE_NAME).put(value, key);
      tx.oncomplete = resolve;
      tx.onerror = () => reject(tx.error || new Error('IndexedDB write failed'));
      tx.onabort = () => reject(tx.error || new Error('IndexedDB write aborted'));
    } catch (error) {
      reject(error);
    }
  });
}

async function cacheMatch(url) {
  if (!('caches' in window)) return null;
  try {
    const cache = await caches.open(CACHE_NAME);
    return cache.match(url);
  } catch {
    return null;
  }
}

async function opfsDir(create = false) {
  if (!navigator.storage?.getDirectory) return null;
  const root = await navigator.storage.getDirectory();
  return root.getDirectoryHandle(OPFS_DIR, { create });
}

async function opfsFileHandle(def, create = false) {
  try {
    const dir = await opfsDir(create);
    if (!dir) return null;
    return dir.getFileHandle(def.file, { create });
  } catch {
    return null;
  }
}

async function opfsFile(def) {
  const handle = await opfsFileHandle(def, false);
  if (!handle) return null;
  try {
    const file = await handle.getFile();
    return file.size > 0 ? file : null;
  } catch {
    return null;
  }
}

async function removeOpfsFile(def) {
  try {
    const dir = await opfsDir(false);
    await dir?.removeEntry(def.file);
  } catch {
    // Ignore cleanup failures; fallback caches can still handle this run.
  }
}

async function cachedArrayBuffer(def) {
  const url = `${HF_BASE}/${def.file}`;
  if (preloadedBuffers.has(url)) return preloadedBuffers.get(url);
  const opfsCached = await opfsFile(def);
  if (opfsCached) return opfsCached.arrayBuffer();
  const legacy = await dbGet(url);
  if (legacy instanceof ArrayBuffer) return legacy;
  if (legacy instanceof Blob) return legacy.arrayBuffer();
  const cached = await cacheMatch(url);
  if (cached) return cached.arrayBuffer();
  return null;
}

async function isCached(def) {
  try {
    const url = `${HF_BASE}/${def.file}`;
    if (preloadedBuffers.has(url)) return true;
    if (await opfsFile(def)) return true;
    return false;
  } catch {
    return false;
  }
}

async function downloadToOpfs(def, url, total) {
  if (!Number.isFinite(total) || total <= 0) return false;
  const handle = await opfsFileHandle(def, true);
  if (!handle) return false;
  const writable = await handle.createWritable();
  let loaded = 0;
  try {
    for (let start = 0; start < total; start += DOWNLOAD_CHUNK_SIZE) {
      const end = Math.min(total - 1, start + DOWNLOAD_CHUNK_SIZE - 1);
      const response = await fetch(url, {
        headers: { Range: `bytes=${start}-${end}` },
        cache: 'no-store',
      });
      if (!(response.status === 206 || (start === 0 && response.ok && total <= DOWNLOAD_CHUNK_SIZE))) {
        throw new Error(`Range request failed for ${def.file}: HTTP ${response.status}`);
      }
      const chunk = new Uint8Array(await response.arrayBuffer());
      await writable.write({ type: 'write', position: start, data: chunk });
      loaded += chunk.byteLength;
      const pct = Math.min(100, (loaded / total) * 100);
      refs.progressLabel.textContent = `Downloading ${def.label}`;
      refs.progressMeta.textContent = `${fmtBytes(loaded)} / ${fmtBytes(total) || def.size}`;
      refs.progressFill.style.width = `${pct}%`;
      await new Promise(resolve => setTimeout(resolve, 0));
    }
    await writable.truncate(total);
    await writable.close();
    return true;
  } catch (error) {
    await writable.abort?.();
    throw error;
  }
}

async function downloadToCache(def) {
  const url = `${HF_BASE}/${def.file}`;
  if (await isCached(def)) return;

  let total = 0;
  try {
    const head = await fetch(url, { method: 'HEAD', cache: 'no-store' });
    total = Number.parseInt(head.headers.get('content-length') || '0', 10);
  } catch {
    total = 0;
  }

  refs.progressLabel.textContent = `Downloading ${def.label}`;
  refs.progressMeta.textContent = fmtBytes(total) || def.size;
  refs.progressFill.style.width = '12%';

  try {
    if (await downloadToOpfs(def, url, total)) return;
  } catch (error) {
    console.warn('OPFS cache failed; falling back for', def.file, error);
    await removeOpfsFile(def);
  }

  let response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status} downloading ${def.file}`);
  if (!total) total = Number.parseInt(response.headers.get('content-length') || '0', 10);

  if ('caches' in window) {
    const cache = await caches.open(CACHE_NAME);
    try {
      await cache.put(url, response.clone());
      refs.progressFill.style.width = '100%';
      return;
    } catch (error) {
      if (def.id === 'dit') throw error;
      console.warn('Cache API failed; falling back to IndexedDB for', def.file, error);
    }
  }

  if (!response.body || !Number.isFinite(total) || total > 512 * 1024 * 1024) {
    throw new Error(`Browser cache failed for ${def.file}. Free browser storage or restart the browser and try again.`);
  }

  {
    const buffer = await response.arrayBuffer();
    try {
      await dbPut(url, buffer);
    } catch (error) {
      console.warn('IndexedDB fallback failed; keeping in-memory buffer for this page load', def.file, error);
    }
    refs.progressLabel.textContent = `${def.label} downloaded`;
    refs.progressMeta.textContent = fmtBytes(buffer.byteLength) || def.size;
    refs.progressFill.style.width = '100%';
    return buffer;
  }
}

async function ensureCached(def) {
  const url = `${HF_BASE}/${def.file}`;
  if (await isCached(def)) {
    refs.progressLabel.textContent = `${def.label} loaded from cache`;
    refs.progressFill.style.width = '100%';
    refs.progressMeta.textContent = def.size;
    return;
  }
  if (!downloadPromises.has(url)) {
    downloadPromises.set(url, downloadToCache(def).then(buffer => {
      if (buffer) preloadedBuffers.set(url, buffer);
      return buffer;
    }).finally(() => downloadPromises.delete(url)));
  }
  return downloadPromises.get(url);
}

async function fetchCached(def) {
  const transientBuffer = await ensureCached(def);
  if (transientBuffer) return transientBuffer;
  const buffer = await cachedArrayBuffer(def);
  if (!buffer) throw new Error(`Model file was not cached: ${def.file}`);
  return buffer;
}

async function preloadModelFiles() {
  setStatus('loading', 'Downloading ONNX model files…');
  setDot(refs.dotRuntime, 'done');
  setDot(refs.dotModels, 'loading');
  refs.generateBtn.disabled = true;
  updateBackendUI();
  refs.generateBtn.textContent = '⏳ Downloading Models…';
  refs.progressWrap.classList.remove('hidden');
  refs.progressFill.style.width = '0%';
  try {
    db = db || await openDB();
    const allDefs = [...MODEL_DEFS, ...Object.values(OPTIONAL_MODEL_DEFS)];
    for (const def of allDefs) {
      try {
        await ensureCached(def);
      } catch (error) {
        throw new Error(`${def.file}: ${error.message}`);
      }
    }
    refs.progressWrap.classList.add('hidden');
    setDot(refs.dotModels, 'done');
    setStatus('success', 'Models cached — click Generate to create ONNX sessions');
  } catch (error) {
    console.error(error);
    setDot(refs.dotModels, 'error');
    setStatus('error', `Model preload failed: ${error.message}`);
  } finally {
    refs.generateBtn.disabled = false;
    refs.generateBtn.textContent = '🎙️ Generate';
    updateBackendUI();
  }
}

function providerOptions() {
  return {
    executionProviders: backendMode === 'gpu' && navigator.gpu ? ['webgpu', 'wasm'] : ['wasm'],
    graphOptimizationLevel: 'all',
  };
}

async function loadEngine() {
  if (sessions && tokenizer) return;
  setStatus('loading', 'Loading tokenizer and ONNX models…');
  updateBackendUI();
  setDot(refs.dotRuntime, 'loading');
  setDot(refs.dotModels, 'loading');
  refs.progressWrap.classList.remove('hidden');
  refs.progressFill.style.width = '0%';

  db = db || await openDB();
  env.allowLocalModels = false;
  tokenizer = tokenizer || await AutoTokenizer.from_pretrained(TOKENIZER_ID);
  setDot(refs.dotRuntime, 'done');

  const options = providerOptions();
  const loaded = {};
  for (const def of MODEL_DEFS) {
    const buffer = await fetchCached(def);
    refs.progressLabel.textContent = `Creating ${def.label} session`;
    refs.progressMeta.textContent = def.size;
    loaded[def.id] = await ort.InferenceSession.create(buffer, options);
  }

  sessions = loaded;
  refs.progressWrap.classList.add('hidden');
  setDot(refs.dotModels, 'done');
  setStatus('success', `Ready — ONNX Runtime Web (${options.executionProviders.join(' -> ')})`);
  updateBackendUI();
}

async function loadOptionalSession(id) {
  await loadEngine();
  if (sessions[id]) return sessions[id];
  const def = OPTIONAL_MODEL_DEFS[id];
  if (!def) throw new Error(`Unknown optional model: ${id}`);
  refs.progressWrap.classList.remove('hidden');
  setDot(refs.dotModels, 'loading');
  const buffer = await fetchCached(def);
  refs.progressLabel.textContent = `Creating ${def.label} session`;
  refs.progressMeta.textContent = def.size;
  sessions[id] = await ort.InferenceSession.create(buffer, providerOptions());
  setDot(refs.dotModels, 'done');
  return sessions[id];
}

function makeFreqsCis(seqLen, headDim = HEAD_DIM) {
  const half = Math.floor(headDim / 2);
  const data = new Float32Array(seqLen * half * 2);
  for (let pos = 0; pos < seqLen; pos++) {
    for (let i = 0; i < half; i++) {
      const freq = 1 / Math.pow(10000, (2 * i) / headDim);
      const angle = pos * freq;
      const base = (pos * half + i) * 2;
      data[base] = Math.cos(angle);
      data[base + 1] = Math.sin(angle);
    }
  }
  return new ort.Tensor('float32', data, [seqLen, half, 2]);
}

function boolTensorFromMask(maskTensor) {
  const source = Array.from(maskTensor.data);
  const data = new Uint8Array(source.length);
  for (let i = 0; i < source.length; i++) data[i] = Number(source[i]) !== 0 ? 1 : 0;
  return new ort.Tensor('bool', data, maskTensor.dims);
}

async function encodeConditionText(session, text, maxLength, stateNames) {
  const encoded = await tokenizer(text, {
    add_special_tokens: true,
    padding: true,
    truncation: true,
    max_length: maxLength,
  });
  const inputIds = new ort.Tensor('int64', BigInt64Array.from(encoded.input_ids.data), encoded.input_ids.dims);
  const attentionMask = boolTensorFromMask(encoded.attention_mask);
  const out = await runSession(session, {
    input_ids: inputIds,
    attention_mask: attentionMask,
    freqs_cis: makeFreqsCis(inputIds.dims[1]),
  });
  return { state: pickOutput(out, stateNames), mask: attentionMask };
}

async function encodeText(text, maxLength) {
  const encoded = await encodeConditionText(sessions.text, text, maxLength, ['text_state', 'hidden_state', 'last_hidden_state']);
  return { textState: encoded.state, textMask: encoded.mask };
}

async function encodeCaption(caption, maxLength) {
  const text = caption.trim();
  const encoded = await encodeConditionText(sessions.caption, text, maxLength, ['caption_state', 'text_state', 'hidden_state', 'last_hidden_state']);
  if (!text) {
    return { captionState: zeroLike(encoded.state), captionMask: zeroLike(encoded.mask), hasCaption: false };
  }
  return { captionState: encoded.state, captionMask: encoded.mask, hasCaption: true };
}

async function encodeNoReference() {
  const refLatent = new ort.Tensor('float32', new Float32Array(LATENT_DIM), [1, 1, LATENT_DIM]);
  const refMask = new ort.Tensor('bool', new Uint8Array([0]), [1, 1]);
  const out = await runSession(sessions.speaker, {
    ref_latent: refLatent,
    ref_mask: refMask,
    freqs_cis: makeFreqsCis(1),
  });
  return {
    speakerState: pickOutput(out, ['speaker_state', 'ref_state', 'hidden_state']),
    speakerMask: out.speaker_mask || refMask,
    hasSpeaker: false,
  };
}

async function audioFileToFloat32(file, maxRefSeconds, normalizeDb, ensureMax) {
  const arrayBuffer = await file.arrayBuffer();
  const AudioCtx = window.AudioContext || window.webkitAudioContext;
  if (!AudioCtx) throw new Error('This browser does not support Web Audio decoding.');
  const context = new AudioCtx({ sampleRate: SAMPLE_RATE });
  const audioBuffer = await context.decodeAudioData(arrayBuffer.slice(0));
  const length = audioBuffer.length;
  const channels = audioBuffer.numberOfChannels;
  const mono = new Float32Array(length);
  for (let channel = 0; channel < channels; channel++) {
    const data = audioBuffer.getChannelData(channel);
    for (let i = 0; i < length; i++) mono[i] += data[i] / channels;
  }

  let samples = mono;
  if (audioBuffer.sampleRate !== SAMPLE_RATE) {
    const offline = new OfflineAudioContext(1, Math.ceil(length * SAMPLE_RATE / audioBuffer.sampleRate), SAMPLE_RATE);
    const sourceBuffer = offline.createBuffer(1, length, audioBuffer.sampleRate);
    sourceBuffer.copyToChannel(mono, 0);
    const source = offline.createBufferSource();
    source.buffer = sourceBuffer;
    source.connect(offline.destination);
    source.start(0);
    const rendered = await offline.startRendering();
    samples = rendered.getChannelData(0).slice();
  }

  await context.close?.();
  if (normalizeDb !== null) {
    let sumSquares = 0;
    for (const sample of samples) sumSquares += sample * sample;
    const rms = Math.sqrt(sumSquares / Math.max(1, samples.length));
    if (rms > 1e-8) {
      const targetRms = Math.pow(10, normalizeDb / 20);
      const gain = targetRms / rms;
      for (let i = 0; i < samples.length; i++) samples[i] *= gain;
    }
  }

  if (ensureMax) {
    let peak = 0;
    for (const sample of samples) peak = Math.max(peak, Math.abs(sample));
    if (peak > 1) {
      for (let i = 0; i < samples.length; i++) samples[i] /= peak;
    }
  }

  const maxSamples = SAMPLE_RATE * maxRefSeconds;
  if (samples.length > maxSamples) samples = samples.slice(0, maxSamples);
  if (samples.length < HOP_LENGTH) throw new Error('Reference audio must be at least 0.04 seconds long.');
  const usableLength = Math.floor(samples.length / HOP_LENGTH) * HOP_LENGTH;
  return samples.slice(0, usableLength);
}

async function encodeReferenceAudio(file, maxRefSeconds, normalizeDb, ensureMax) {
  await loadOptionalSession('encoder');
  const samples = await audioFileToFloat32(file, maxRefSeconds, normalizeDb, ensureMax);
  refs.progressLabel.textContent = 'Encoding reference audio';
  refs.progressFill.style.width = '100%';
  const audioTensor = new ort.Tensor('float32', samples, [1, 1, samples.length]);
  const encoded = await runSession(sessions.encoder, { audio: audioTensor, wav: audioTensor });
  const refLatent = pickOutput(encoded, ['latent', 'ref_latent', 'z']);
  const refFrames = refLatent.dims[1];
  const refMask = new ort.Tensor('bool', new Uint8Array(refFrames).fill(1), [1, refFrames]);
  const out = await runSession(sessions.speaker, {
    ref_latent: refLatent,
    ref_mask: refMask,
    freqs_cis: makeFreqsCis(refFrames),
  });
  return {
    speakerState: pickOutput(out, ['speaker_state', 'ref_state', 'hidden_state']),
    speakerMask: out.speaker_mask || refMask,
    referenceSeconds: samples.length / SAMPLE_RATE,
    referenceName: file.name,
    hasSpeaker: true,
  };
}

function durationFeatures(text, textMask, maxTextLen, hasSpeaker) {
  const tokenCount = Array.from(textMask.data).reduce((sum, value) => sum + (value ? 1 : 0), 0);
  const chars = Array.from(text);
  const charCount = Math.max(chars.length, 1);
  const log1pCap = (count, cap) => Math.log1p(Math.min(Math.max(count, 0), cap)) / Math.log1p(cap);
  const kanaCount = chars.filter(ch => /[\u3040-\u309f\u30a0-\u30ff]/u.test(ch)).length;
  const kanjiCount = chars.filter(ch => /[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]/u.test(ch)).length;
  const alnumCount = chars.filter(ch => /^[A-Za-z0-9]$/.test(ch)).length;
  const emojiCount = (text.match(/\p{Extended_Pictographic}/gu) || []).length;
  const periodCount = (text.match(/[。.]/g) || []).length;
  const commaCount = (text.match(/[、,]/g) || []).length;
  const longVowelCount = (text.match(/ー/g) || []).length;
  const ellipsisCount = (text.match(/…/g) || []).length;
  const exclamationCount = (text.match(/[！!]/g) || []).length;
  const questionCount = (text.match(/[？?]/g) || []).length;
  return new ort.Tensor('float32', new Float32Array([
    tokenCount / Math.max(1, maxTextLen),
    Math.log1p(charCount) / Math.log1p(512),
    tokenCount / charCount,
    log1pCap(periodCount, 8),
    log1pCap(commaCount, 16),
    log1pCap(longVowelCount, 8),
    log1pCap(ellipsisCount, 8),
    log1pCap(exclamationCount, 8),
    log1pCap(questionCount, 8),
    log1pCap(emojiCount, 8),
    kanaCount / charCount,
    kanjiCount / charCount,
    alnumCount / charCount,
    hasSpeaker ? 1 : 0,
  ]), [1, 14]);
}

async function predictLatentSteps(text, textCond, captionCond, speakerCond, durationScale, maxTextLen) {
  await loadOptionalSession('duration');
  refs.progressLabel.textContent = 'Predicting duration';
  refs.progressFill.style.width = '100%';
  const out = await runSession(sessions.duration, {
    text_state: textCond.textState,
    text_mask: textCond.textMask,
    speaker_state: speakerCond.speakerState,
    speaker_mask: speakerCond.speakerMask,
    caption_state: captionCond.captionState,
    caption_mask: captionCond.captionMask,
    duration_features: durationFeatures(text, textCond.textMask, maxTextLen, speakerCond.hasSpeaker),
    has_speaker: new ort.Tensor('bool', new Uint8Array([speakerCond.hasSpeaker ? 1 : 0]), [1]),
    has_caption: new ort.Tensor('bool', new Uint8Array([captionCond.hasCaption ? 1 : 0]), [1]),
  });
  const logFrames = pickOutput(out, ['log_frames', 'pred_log_frames', 'duration_log_frames']);
  const frames = Math.max(1, Math.expm1(Number(logFrames.data[0]))) * durationScale;
  return Math.min(750, Math.max(1, Math.round(frames)));
}

function seededRandom(seed) {
  let state = (seed >>> 0) || 123456789;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

function randnArray(length, seed) {
  const rnd = seededRandom(seed);
  const data = new Float32Array(length);
  for (let i = 0; i < length; i += 2) {
    const u1 = Math.max(1e-7, rnd());
    const u2 = rnd();
    const mag = Math.sqrt(-2 * Math.log(u1));
    data[i] = mag * Math.cos(2 * Math.PI * u2);
    if (i + 1 < length) data[i + 1] = mag * Math.sin(2 * Math.PI * u2);
  }
  return data;
}

function zeroLike(tensor) {
  const Ctor = tensor.data.constructor;
  return new ort.Tensor(tensor.type, new Ctor(tensor.data.length), tensor.dims);
}

async function ditStep(x, t, textState, textMask, speakerState, speakerMask, captionState, captionMask) {
  const out = await runSession(sessions.dit, {
    x_t: x,
    t: new ort.Tensor('float32', new Float32Array([t]), [1]),
    text_state: textState,
    text_mask: textMask,
    speaker_state: speakerState,
    speaker_mask: speakerMask,
    caption_state: captionState,
    caption_mask: captionMask,
    freqs_cis: makeFreqsCis(x.dims[1]),
  });
  return pickOutput(out, ['v_pred', 'velocity', 'output']);
}

function makeSchedule(numSteps, mode, swayCoeff) {
  const values = [];
  for (let i = 0; i <= numSteps; i++) {
    let u = i / numSteps;
    if (mode === 'sway') {
      u = u + swayCoeff * (Math.cos(0.5 * Math.PI * u) + u - 1.0);
      u = Math.max(0, Math.min(1, u));
    }
    values.push((1 - u) * 0.999);
  }
  for (let i = 0; i < values.length - 1; i++) {
    if (!(values[i] > values[i + 1])) throw new Error('Time schedule must be strictly decreasing; adjust sway coeff or steps.');
  }
  return values;
}

function findFlatteningPoint(latent, windowSize, stdThreshold, meanThreshold) {
  const [, steps, dim] = latent.dims;
  const data = latent.data;
  const paddedSteps = steps + windowSize;
  for (let step = 0; step < paddedSteps - windowSize; step++) {
    let sum = 0;
    let sumSq = 0;
    let count = 0;
    for (let w = 0; w < windowSize; w++) {
      const frame = step + w;
      for (let d = 0; d < dim; d++) {
        const value = frame < steps ? data[frame * dim + d] : 0;
        sum += value;
        sumSq += value * value;
        count += 1;
      }
    }
    const mean = sum / count;
    const variance = Math.max(0, sumSq / count - mean * mean);
    if (Math.sqrt(variance) < stdThreshold && Math.abs(mean) < meanThreshold) return step;
  }
  return steps;
}

async function sampleLatent(args) {
  const {
    textState, textMask, speakerState, speakerMask, captionState, captionMask,
    latentSteps, numSteps, seed, cfgText, cfgCaption, cfgSpeaker, cfgMode,
    cfgMinT, cfgMaxT, truncationFactor, rescaleK, rescaleSigma, tScheduleMode, swayCoeff,
  } = args;
  const total = latentSteps * LATENT_DIM;
  let xData = randnArray(total, seed);
  if (truncationFactor !== null) {
    for (let i = 0; i < xData.length; i++) xData[i] *= truncationFactor;
  }
  let x = new ort.Tensor('float32', xData, [1, latentSteps, LATENT_DIM]);
  const textZero = zeroLike(textState);
  const textMaskZero = zeroLike(textMask);
  const speakerZero = zeroLike(speakerState);
  const speakerMaskZero = zeroLike(speakerMask);
  const captionZero = zeroLike(captionState);
  const captionMaskZero = zeroLike(captionMask);
  const schedule = makeSchedule(numSteps, tScheduleMode, swayCoeff);

  for (let i = 0; i < numSteps; i++) {
    const t = schedule[i];
    const tNext = schedule[i + 1];
    const dt = tNext - t;
    refs.progressLabel.textContent = `Sampling ${i + 1} / ${numSteps}`;
    refs.progressFill.style.width = `${((i + 1) / numSteps) * 100}%`;

    const cond = await ditStep(x, t, textState, textMask, speakerState, speakerMask, captionState, captionMask);
    const velocity = new Float32Array(cond.data);

    const active = [];
    if (cfgText > 0) active.push(['text', cfgText]);
    if (cfgSpeaker > 0) active.push(['speaker', cfgSpeaker]);
    if (cfgCaption > 0 && args.hasCaption) active.push(['caption', cfgCaption]);

    if (cfgMinT <= t && t <= cfgMaxT && active.length) {
      if (cfgMode === 'joint') {
        const scales = active.map(([, scale]) => scale);
        if (Math.max(...scales) - Math.min(...scales) > 1e-6) {
          throw new Error("joint CFG requires equal enabled text/caption/speaker scales, or set CFG Scale Override.");
        }
        const uncond = await ditStep(x, t, textZero, textMaskZero, speakerZero, speakerMaskZero, captionZero, captionMaskZero);
        for (let k = 0; k < velocity.length; k++) velocity[k] += scales[0] * (cond.data[k] - uncond.data[k]);
      } else {
        const selected = cfgMode === 'alternating' ? [active[i % active.length]] : active;
        for (const [name, scale] of selected) {
          const uncond = await ditStep(
            x,
            t,
            name === 'text' ? textZero : textState,
            name === 'text' ? textMaskZero : textMask,
            name === 'speaker' ? speakerZero : speakerState,
            name === 'speaker' ? speakerMaskZero : speakerMask,
            name === 'caption' ? captionZero : captionState,
            name === 'caption' ? captionMaskZero : captionMask,
          );
          for (let k = 0; k < velocity.length; k++) velocity[k] += scale * (cond.data[k] - uncond.data[k]);
        }
      }
    }

    if (rescaleK !== null && rescaleSigma !== null && t < 1) {
      const oneMinusT = 1 - t;
      const snr = (oneMinusT * oneMinusT) / Math.max(1e-7, t * t);
      const sigmaSq = rescaleSigma * rescaleSigma;
      const ratio = (snr * sigmaSq + 1) / (snr * sigmaSq / rescaleK + 1);
      for (let k = 0; k < velocity.length; k++) {
        velocity[k] = (ratio * (oneMinusT * velocity[k] + xData[k]) - xData[k]) / oneMinusT;
      }
    }

    for (let k = 0; k < xData.length; k++) xData[k] += velocity[k] * dt;
    x = new ort.Tensor('float32', xData, [1, latentSteps, LATENT_DIM]);
    await new Promise(resolve => setTimeout(resolve, 0));
  }
  return x;
}

async function decodeLatent(latent) {
  const out = await runSession(sessions.decoder, { latent, z: latent });
  return new Float32Array(pickOutput(out, ['audio', 'wav', 'waveform']).data);
}

function float32ToWav(samples, sampleRate) {
  const dataSize = samples.length * 2;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);
  const write = (offset, str) => { for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i)); };
  write(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  write(8, 'WAVE');
  write(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);
  write(36, 'data');
  view.setUint32(40, dataSize, true);
  let offset = 44;
  for (const sample of samples) {
    const s = Math.max(-1, Math.min(1, sample));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true);
    offset += 2;
  }
  return new Blob([buffer], { type: 'audio/wav' });
}

function escapeHtml(value) {
  return String(value).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function displayAudio(items) {
  if (currentAudioUrl) URL.revokeObjectURL(currentAudioUrl);
  const urls = items.map(item => ({ ...item, url: URL.createObjectURL(item.blob) }));
  currentAudioUrl = urls[0]?.url || null;
  refs.resultsArea.innerHTML = `
    <div class="results-list">
      ${urls.map((item, index) => `
        <div class="result-item">
          <div class="result-meta">
            <span>candidate ${index + 1}</span>
            <span>${item.seconds.toFixed(2)}s target</span>
            <span>${item.steps} steps</span>
            <span>${item.schedule}</span>
            <span>seed ${item.seed}</span>
            <span>${item.caption ? 'caption' : 'text-only'}</span>
            <span>${item.referenceName ? 'ref audio' : 'no-ref'}</span>
          </div>
          <audio controls preload="metadata" src="${item.url}"></audio>
          <div class="result-actions">
            <a href="${item.url}" download="irodori-voicedesign-${Date.now()}-${index + 1}.wav" class="btn btn-ghost">⬇ Download WAV</a>
          </div>
        </div>`).join('')}
    </div>`;
}

async function generateSpeech() {
  const text = refs.textInput.value.trim();
  if (!text) {
    alert('Please enter Japanese text.');
    return;
  }
  const caption = refs.captionInput.value.trim();
  refs.generateBtn.disabled = true;
  updateBackendUI();
  refs.generateBtn.textContent = '⏳ Loading / Generating…';
  refs.logCard.classList.remove('hidden');
  refs.runLog.textContent = '';
  refs.resultsArea.innerHTML = '<div class="results-placeholder"><span class="icon">⏳</span><p>Downloading cached ONNX models and generating locally in the browser.</p></div>';
  setDot(refs.dotOutput, 'pending');
  refs.progressWrap.classList.remove('hidden');

  const started = performance.now();
  try {
    await loadEngine();
    const numSteps = clampInteger(refs.numStepsInput, 4, 1, 120);
    const numCandidates = clampInteger(refs.numCandidatesInput, 1, 1, 32);
    const seed = clampInteger(refs.seedInput, Math.floor(Math.random() * 2 ** 31), 0, 2 ** 31 - 1);
    const durationScale = clampNumber(refs.durationScaleInput, 1, 0.5, 1.5);
    const maxRefSecondsRaw = optionalNumber(refs.maxRefSecondsInput, 0, 120);
    const maxRefSeconds = maxRefSecondsRaw === null || maxRefSecondsRaw <= 0 ? 120 : maxRefSecondsRaw;
    const refNormalizeDb = optionalNumber(refs.refNormalizeDbInput, -60, 0);
    const refEnsureMax = refs.refEnsureMaxInput.checked;
    const tScheduleMode = refs.tScheduleModeSel.value;
    const swayCoeff = clampNumber(refs.swayCoeffInput, -1, -1, 1.5);
    const cfgText = clampNumber(refs.cfgScaleTextInput, 0, 0, 10);
    const cfgCaption = clampNumber(refs.cfgScaleCaptionInput, 0, 0, 10);
    const rawCfgSpeaker = clampNumber(refs.cfgScaleSpeakerInput, 0, 0, 10);
    const cfgMode = refs.cfgModeSel.value;
    const cfgOverride = optionalNumber(refs.cfgOverrideInput, 0, 10);
    const cfgMinT = clampNumber(refs.cfgMinTInput, 0.5, 0, 1);
    const cfgMaxT = clampNumber(refs.cfgMaxTInput, 1, 0, 1);
    const maxTextLen = optionalInteger(refs.maxTextLenInput, 1, 512) || 256;
    const maxCaptionLen = optionalInteger(refs.maxCaptionLenInput, 1, 512) || maxTextLen;
    const truncationFactor = optionalNumber(refs.truncationFactorInput, 0, 2);
    const rescaleK = optionalNumber(refs.rescaleKInput, 0.001, 100);
    const rescaleSigma = optionalNumber(refs.rescaleSigmaInput, 0.001, 100);
    const speakerKvScale = optionalNumber(refs.speakerKvScaleInput, 0.001, 10);
    const speakerKvMinT = optionalNumber(refs.speakerKvMinTInput, 0, 1);
    const speakerKvMaxLayers = optionalInteger(refs.speakerKvMaxLayersInput, 1, 999);
    const trimTail = refs.trimTailInput.checked;
    const tailWindowSize = clampInteger(refs.tailWindowSizeInput, 20, 1, 200);
    const tailStdThreshold = clampNumber(refs.tailStdThresholdInput, 0.05, 0, 1);
    const tailMeanThreshold = clampNumber(refs.tailMeanThresholdInput, 0.1, 0, 1);
    const referenceFile = refs.referenceAudio.files?.[0] || null;
    const secondsText = refs.secondsInput.value.trim();
    const cfgScaleText = cfgOverride ?? cfgText;
    const cfgScaleCaption = cfgOverride ?? cfgCaption;
    const cfgScaleSpeaker = referenceFile ? (cfgOverride ?? rawCfgSpeaker) : 0;

    if (cfgMode === 'joint') {
      const enabledScales = [cfgScaleText];
      if (caption) enabledScales.push(cfgScaleCaption);
      if (referenceFile) enabledScales.push(cfgScaleSpeaker);
      if (Math.max(...enabledScales) - Math.min(...enabledScales) > 1e-6) {
        throw new Error("joint CFG requires equal enabled scales. Set CFG Scale Override or match text/caption/speaker scales.");
      }
    }

    setStatus('loading', `Encoding text, ${caption ? 'caption' : 'blank caption'}, and ${referenceFile ? 'reference audio' : 'no-reference'} condition…`);
    refs.runLog.textContent += `[local] text=${text}\n`;
    refs.runLog.textContent += `[local] caption=${caption || '(blank)'}\n`;
    const textCond = await encodeText(text, maxTextLen);
    const captionCond = await encodeCaption(caption, maxCaptionLen);
    const speakerCond = referenceFile ? await encodeReferenceAudio(referenceFile, maxRefSeconds, refNormalizeDb, refEnsureMax) : await encodeNoReference();
    let latentSteps;
    let seconds;
    if (secondsText) {
      seconds = clampNumber(refs.secondsInput, 1, 0.5, 30);
      latentSteps = Math.max(1, Math.ceil((seconds * SAMPLE_RATE) / HOP_LENGTH));
    } else {
      latentSteps = await predictLatentSteps(text, textCond, captionCond, speakerCond, durationScale, maxTextLen);
      seconds = (latentSteps * HOP_LENGTH) / SAMPLE_RATE;
    }
    refs.runLog.textContent += `[local] seconds=${seconds.toFixed(2)} latent_steps=${latentSteps} steps=${numSteps} seed=${seed} candidates=${numCandidates} schedule=${tScheduleMode}\n`;
    refs.runLog.textContent += `[local] cfg=${cfgMode} text=${cfgScaleText} caption=${cfgScaleCaption} speaker=${cfgScaleSpeaker} cfg_t=[${cfgMinT},${cfgMaxT}]\n`;
    if (speakerCond.referenceSeconds) refs.runLog.textContent += `[local] reference_seconds=${speakerCond.referenceSeconds.toFixed(2)}\n`;
    if (refs.contextKvCacheInput.checked || speakerKvScale !== null || speakerKvMaxLayers !== null || speakerKvMinT !== null) {
      refs.runLog.textContent += '[local] context/speaker KV cache options require a fused ONNX export with KV-cache inputs; ignored by this split-step export when absent.\n';
    }
    if (refs.speakerUncondModeSel.value !== 'mask') {
      refs.runLog.textContent += '[local] speaker_uncond_mode=noise is PyTorch-runtime specific unless the ONNX export exposes it; using mask-style zero speaker uncond.\n';
    }

    setStatus('loading', 'Sampling latent with ONNX Runtime Web…');
    const outputs = [];
    for (let i = 0; i < numCandidates; i++) {
      const candidateSeed = (seed + i) >>> 0;
      refs.runLog.textContent += `[local] candidate ${i + 1}/${numCandidates} seed=${candidateSeed}\n`;
      const latent = await sampleLatent({
        ...textCond,
        ...captionCond,
        ...speakerCond,
        latentSteps,
        numSteps,
        seed: candidateSeed,
        cfgText: cfgScaleText,
        cfgCaption: cfgScaleCaption,
        cfgSpeaker: cfgScaleSpeaker,
        cfgMode,
        cfgMinT,
        cfgMaxT,
        truncationFactor,
        rescaleK,
        rescaleSigma,
        tScheduleMode,
        swayCoeff,
      });

      setStatus('loading', `Decoding DACVAE audio candidate ${i + 1}…`);
      refs.progressLabel.textContent = `Decoding audio ${i + 1} / ${numCandidates}`;
      refs.progressMeta.textContent = '283 MB';
      refs.progressFill.style.width = '100%';
      const audio = await decodeLatent(latent);
      let targetSamples = Math.min(audio.length, Math.floor(seconds * SAMPLE_RATE));
      if (trimTail) {
        const flatteningPoint = findFlatteningPoint(latent, tailWindowSize, tailStdThreshold, tailMeanThreshold);
        const flatteningSamples = flatteningPoint * HOP_LENGTH;
        if (flatteningSamples > 0) targetSamples = Math.min(targetSamples, flatteningSamples);
      }
      outputs.push({
        blob: float32ToWav(audio.slice(0, targetSamples), SAMPLE_RATE),
        seconds,
        steps: numSteps,
        seed: candidateSeed,
        caption,
        schedule: tScheduleMode,
        referenceName: speakerCond.referenceName,
      });
    }
    displayAudio(outputs);
    setDot(refs.dotOutput, 'done');
    refs.progressWrap.classList.add('hidden');
    const elapsed = ((performance.now() - started) / 1000).toFixed(1);
    refs.runLog.textContent += `[local] done in ${elapsed}s\n`;
    setStatus('success', `Done — generated locally in ${elapsed}s`);
  } catch (error) {
    console.error(error);
    refs.runLog.textContent += `\n[error] ${error.stack || error.message}\n`;
    setStatus('error', error.message);
    setDot(refs.dotOutput, 'error');
    refs.resultsArea.innerHTML = `<div class="results-placeholder"><span class="icon">⚠️</span><p>${escapeHtml(error.message)}</p></div>`;
  } finally {
    refs.generateBtn.disabled = false;
    refs.generateBtn.textContent = '🎙️ Generate';
    updateBackendUI();
  }
}

document.querySelectorAll('.sample-chip').forEach(button => {
  button.addEventListener('click', () => {
    refs.textInput.value = button.dataset.text || '';
    if (button.dataset.caption) refs.captionInput.value = button.dataset.caption;
    refs.textInput.focus();
  });
});

refs.resetBtn.addEventListener('click', resetParams);
refs.generateBtn.addEventListener('click', generateSpeech);
refs.backendToggle.addEventListener('click', switchBackend);
refs.cacheClearBtn.addEventListener('click', clearModelCache);
refs.tScheduleModeSel.addEventListener('change', updateBackendUI);

resetParams();
updateBackendUI();
setDot(refs.dotRuntime, 'done');
preloadModelFiles();
