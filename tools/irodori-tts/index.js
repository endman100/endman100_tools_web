import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/ort.webgpu.min.mjs';
import { AutoTokenizer, env } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.2';

const HF_BASE = 'https://huggingface.co/mtsmfm/Irodori-TTS-500M-v3-ONNX/resolve/main';
const TOKENIZER_ID = 'llm-jp/llm-jp-3-150m';
const DB_NAME = 'irodori-v3-onnx-cache';
const STORE_NAME = 'files';
const CACHE_NAME = 'irodori-v3-onnx-files';
const OPFS_DIR = 'irodori-v3-onnx-files-v2';
const SAMPLE_RATE = 48000;
const HOP_LENGTH = 1920;
const LATENT_DIM = 32;
const HEAD_DIM = 64;

const MODEL_DEFS = [
  { id: 'text', label: 'Text Encoder', file: 'text_encoder.onnx', size: '338 MB' },
  { id: 'speaker', label: 'Speaker Encoder', file: 'speaker_encoder.onnx', size: '242 MB' },
  { id: 'dit', label: 'Diffusion Transformer Step', file: 'dit_step.onnx', size: '1.4 GB' },
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
  textInput: document.getElementById('textInput'),
  referenceAudio: document.getElementById('referenceAudio'),
  numStepsInput: document.getElementById('numStepsInput'),
  numCandidatesInput: document.getElementById('numCandidatesInput'),
  seedInput: document.getElementById('seedInput'),
  secondsInput: document.getElementById('secondsInput'),
  durationScaleInput: document.getElementById('durationScaleInput'),
  cfgModeSel: document.getElementById('cfgModeSel'),
  cfgScaleTextInput: document.getElementById('cfgScaleTextInput'),
  cfgScaleSpeakerInput: document.getElementById('cfgScaleSpeakerInput'),
  cfgOverrideInput: document.getElementById('cfgOverrideInput'),
  cfgMinTInput: document.getElementById('cfgMinTInput'),
  cfgMaxTInput: document.getElementById('cfgMaxTInput'),
  contextKvCacheInput: document.getElementById('contextKvCacheInput'),
  truncationFactorInput: document.getElementById('truncationFactorInput'),
  rescaleKInput: document.getElementById('rescaleKInput'),
  rescaleSigmaInput: document.getElementById('rescaleSigmaInput'),
  speakerKvScaleInput: document.getElementById('speakerKvScaleInput'),
  speakerKvMinTInput: document.getElementById('speakerKvMinTInput'),
  speakerKvMaxLayersInput: document.getElementById('speakerKvMaxLayersInput'),
};

const DEFAULTS = {
  numSteps: '40',
  numCandidates: '1',
  seed: '',
  seconds: '',
  durationScale: '1.0',
  cfgMode: 'independent',
  cfgScaleText: '2.0',
  cfgScaleSpeaker: '0',
  cfgOverride: '',
  cfgMinT: '0.5',
  cfgMaxT: '1',
  contextKvCache: true,
  truncationFactor: '',
  rescaleK: '',
  rescaleSigma: '',
  speakerKvScale: '',
  speakerKvMinT: '0.9',
  speakerKvMaxLayers: '',
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

function resetParams() {
  refs.referenceAudio.value = '';
  refs.numStepsInput.value = DEFAULTS.numSteps;
  refs.numCandidatesInput.value = DEFAULTS.numCandidates;
  refs.seedInput.value = DEFAULTS.seed;
  refs.secondsInput.value = DEFAULTS.seconds;
  refs.durationScaleInput.value = DEFAULTS.durationScale;
  refs.cfgModeSel.value = DEFAULTS.cfgMode;
  refs.cfgScaleTextInput.value = DEFAULTS.cfgScaleText;
  refs.cfgScaleSpeakerInput.value = DEFAULTS.cfgScaleSpeaker;
  refs.cfgOverrideInput.value = DEFAULTS.cfgOverride;
  refs.cfgMinTInput.value = DEFAULTS.cfgMinT;
  refs.cfgMaxTInput.value = DEFAULTS.cfgMaxT;
  refs.contextKvCacheInput.checked = DEFAULTS.contextKvCache;
  refs.truncationFactorInput.value = DEFAULTS.truncationFactor;
  refs.rescaleKInput.value = DEFAULTS.rescaleK;
  refs.rescaleSigmaInput.value = DEFAULTS.rescaleSigma;
  refs.speakerKvScaleInput.value = DEFAULTS.speakerKvScale;
  refs.speakerKvMinTInput.value = DEFAULTS.speakerKvMinT;
  refs.speakerKvMaxLayersInput.value = DEFAULTS.speakerKvMaxLayers;
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
    if (await dbGet(url)) return true;
    return Boolean(await cacheMatch(url));
  } catch {
    return false;
  }
}

async function downloadToOpfs(def, response, total) {
  if (!response.body) return false;
  const handle = await opfsFileHandle(def, true);
  if (!handle) return false;
  const writable = await handle.createWritable();
  const reader = response.body.getReader();
  let loaded = 0;
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      await writable.write(value);
      loaded += value.byteLength;
      const pct = total ? Math.min(100, (loaded / total) * 100) : 0;
      refs.progressLabel.textContent = `Downloading ${def.label}`;
      refs.progressMeta.textContent = `${fmtBytes(loaded)} / ${fmtBytes(total) || def.size}`;
      refs.progressFill.style.width = `${pct}%`;
      await new Promise(resolve => setTimeout(resolve, 0));
    }
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

  let response = await fetch(url);
  if (!response.ok) throw new Error(`HTTP ${response.status} downloading ${def.file}`);
  const total = Number.parseInt(response.headers.get('content-length') || '0', 10);

  refs.progressLabel.textContent = `Downloading ${def.label}`;
  refs.progressMeta.textContent = fmtBytes(total) || def.size;
  refs.progressFill.style.width = '12%';

  try {
    if (await downloadToOpfs(def, response.clone(), total)) return;
  } catch (error) {
    console.warn('OPFS cache failed; falling back for', def.file, error);
    await removeOpfsFile(def);
    response = await fetch(url);
    if (!response.ok) throw new Error(`HTTP ${response.status} downloading ${def.file}`);
  }

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
  }
}

function providerOptions() {
  return {
    executionProviders: navigator.gpu ? ['webgpu', 'wasm'] : ['wasm'],
    graphOptimizationLevel: 'all',
  };
}

async function loadEngine() {
  if (sessions && tokenizer) return;
  setStatus('loading', 'Loading tokenizer and ONNX models…');
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

async function encodeText(text) {
  const encoded = await tokenizer(text, {
    add_special_tokens: true,
    padding: true,
    truncation: true,
    max_length: 256,
  });
  const inputIds = new ort.Tensor('int64', BigInt64Array.from(encoded.input_ids.data), encoded.input_ids.dims);
  const attentionMask = boolTensorFromMask(encoded.attention_mask);
  const feeds = {
    input_ids: inputIds,
    attention_mask: attentionMask,
    freqs_cis: makeFreqsCis(inputIds.dims[1]),
  };
  const out = await sessions.text.run(feeds);
  return { textState: out.text_state, textMask: attentionMask };
}

async function encodeNoReference() {
  const refLatent = new ort.Tensor('float32', new Float32Array(LATENT_DIM), [1, 1, LATENT_DIM]);
  const refMask = new ort.Tensor('bool', new Uint8Array([0]), [1, 1]);
  const out = await sessions.speaker.run({
    ref_latent: refLatent,
    ref_mask: refMask,
    freqs_cis: makeFreqsCis(1),
  });
  return { speakerState: out.speaker_state, speakerMask: out.speaker_mask };
}

async function audioFileToFloat32(file) {
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
  const maxSamples = SAMPLE_RATE * 15;
  if (samples.length > maxSamples) samples = samples.slice(0, maxSamples);
  if (samples.length < HOP_LENGTH) throw new Error('Reference audio must be at least 0.04 seconds long.');
  const usableLength = Math.floor(samples.length / HOP_LENGTH) * HOP_LENGTH;
  return samples.slice(0, usableLength);
}

async function encodeReferenceAudio(file) {
  await loadOptionalSession('encoder');
  const samples = await audioFileToFloat32(file);
  refs.progressLabel.textContent = 'Encoding reference audio';
  refs.progressFill.style.width = '100%';
  const audioTensor = new ort.Tensor('float32', samples, [1, 1, samples.length]);
  const encoded = await sessions.encoder.run({ audio: audioTensor });
  const refLatent = encoded.latent;
  const refFrames = refLatent.dims[1];
  const refMask = new ort.Tensor('bool', new Uint8Array(refFrames).fill(1), [1, refFrames]);
  const out = await sessions.speaker.run({
    ref_latent: refLatent,
    ref_mask: refMask,
    freqs_cis: makeFreqsCis(refFrames),
  });
  return {
    speakerState: out.speaker_state,
    speakerMask: out.speaker_mask,
    referenceSeconds: samples.length / SAMPLE_RATE,
    referenceName: file.name,
    hasSpeaker: true,
  };
}

async function predictLatentSteps(textCond, speakerCond, durationScale) {
  await loadOptionalSession('duration');
  refs.progressLabel.textContent = 'Predicting duration';
  refs.progressFill.style.width = '100%';
  const out = await sessions.duration.run({
    text_state: textCond.textState,
    text_mask: textCond.textMask,
    speaker_state: speakerCond.speakerState,
    has_speaker: new ort.Tensor('bool', new Uint8Array([speakerCond.hasSpeaker ? 1 : 0]), [1]),
  });
  const frames = Math.exp(Number(out.log_frames.data[0])) * durationScale;
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

async function ditStep(x, t, textState, textMask, speakerState, speakerMask) {
  const out = await sessions.dit.run({
    x_t: x,
    t: new ort.Tensor('float32', new Float32Array([t]), [1]),
    text_state: textState,
    text_mask: textMask,
    speaker_state: speakerState,
    speaker_mask: speakerMask,
    freqs_cis: makeFreqsCis(x.dims[1]),
  });
  return out.v_pred;
}

async function sampleLatent(args) {
  const { textState, textMask, speakerState, speakerMask, latentSteps, numSteps, seed, cfgText, cfgSpeaker, cfgMinT, cfgMaxT } = args;
  const total = latentSteps * LATENT_DIM;
  let xData = randnArray(total, seed);
  let x = new ort.Tensor('float32', xData, [1, latentSteps, LATENT_DIM]);
  const textZero = zeroLike(textState);
  const textMaskZero = zeroLike(textMask);
  const speakerZero = zeroLike(speakerState);
  const speakerMaskZero = zeroLike(speakerMask);

  for (let i = 0; i < numSteps; i++) {
    const t = (1 - i / numSteps) * 0.999;
    const tNext = (1 - (i + 1) / numSteps) * 0.999;
    const dt = tNext - t;
    refs.progressLabel.textContent = `Sampling ${i + 1} / ${numSteps}`;
    refs.progressFill.style.width = `${((i + 1) / numSteps) * 100}%`;

    const cond = await ditStep(x, t, textState, textMask, speakerState, speakerMask);
    const velocity = new Float32Array(cond.data);

    if (cfgMinT <= t && t <= cfgMaxT && cfgText > 0) {
      const uncondText = await ditStep(x, t, textZero, textMaskZero, speakerState, speakerMask);
      for (let k = 0; k < velocity.length; k++) velocity[k] += cfgText * (cond.data[k] - uncondText.data[k]);
    }
    if (cfgMinT <= t && t <= cfgMaxT && cfgSpeaker > 0) {
      const uncondSpeaker = await ditStep(x, t, textState, textMask, speakerZero, speakerMaskZero);
      for (let k = 0; k < velocity.length; k++) velocity[k] += cfgSpeaker * (cond.data[k] - uncondSpeaker.data[k]);
    }

    for (let k = 0; k < xData.length; k++) xData[k] += velocity[k] * dt;
    x = new ort.Tensor('float32', xData, [1, latentSteps, LATENT_DIM]);
    await new Promise(resolve => setTimeout(resolve, 0));
  }
  return x;
}

async function decodeLatent(latent) {
  const out = await sessions.decoder.run({ latent });
  return new Float32Array(out.audio.data);
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

function displayAudio(blob, meta) {
  if (currentAudioUrl) URL.revokeObjectURL(currentAudioUrl);
  currentAudioUrl = URL.createObjectURL(blob);
  refs.resultsArea.innerHTML = `
    <div class="results-list">
      <div class="result-item">
        <div class="result-meta">
          <span>${meta.seconds.toFixed(2)}s target</span>
          <span>${meta.steps} steps</span>
          <span>seed ${meta.seed}</span>
          <span>${meta.referenceName ? 'ref audio' : 'no-ref'}</span>
        </div>
        <audio controls preload="metadata" src="${currentAudioUrl}"></audio>
        <div class="result-actions">
          <a href="${currentAudioUrl}" download="irodori-onnx-${Date.now()}.wav" class="btn btn-ghost">⬇ Download WAV</a>
        </div>
      </div>
    </div>`;
}

async function generateSpeech() {
  const text = refs.textInput.value.trim();
  if (!text) {
    alert('Please enter Japanese text.');
    return;
  }
  refs.generateBtn.disabled = true;
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
    const seed = clampInteger(refs.seedInput, Math.floor(Math.random() * 2 ** 31), 0, 2 ** 31 - 1);
    const durationScale = clampNumber(refs.durationScaleInput, 1, 0.5, 1.5);
    const cfgText = clampNumber(refs.cfgScaleTextInput, 0, 0, 10);
    const cfgSpeaker = clampNumber(refs.cfgScaleSpeakerInput, 0, 0, 10);
    const cfgMinT = clampNumber(refs.cfgMinTInput, 0.5, 0, 1);
    const cfgMaxT = clampNumber(refs.cfgMaxTInput, 1, 0, 1);
    const referenceFile = refs.referenceAudio.files?.[0] || null;
    const secondsText = refs.secondsInput.value.trim();

    setStatus('loading', `Encoding text and ${referenceFile ? 'reference audio' : 'no-reference'} speaker condition…`);
    refs.runLog.textContent += `[local] text=${text}\n`;
    const textCond = await encodeText(text);
    const speakerCond = referenceFile ? await encodeReferenceAudio(referenceFile) : await encodeNoReference();
    let latentSteps;
    let seconds;
    if (secondsText) {
      seconds = clampNumber(refs.secondsInput, 1, 0.5, 30);
      latentSteps = Math.max(1, Math.ceil((seconds * SAMPLE_RATE) / HOP_LENGTH));
    } else {
      latentSteps = await predictLatentSteps(textCond, speakerCond, durationScale);
      seconds = (latentSteps * HOP_LENGTH) / SAMPLE_RATE;
    }
    refs.runLog.textContent += `[local] seconds=${seconds.toFixed(2)} latent_steps=${latentSteps} steps=${numSteps} seed=${seed} mode=${referenceFile ? `ref:${referenceFile.name}` : 'no-ref'}\n`;
    if (speakerCond.referenceSeconds) refs.runLog.textContent += `[local] reference_seconds=${speakerCond.referenceSeconds.toFixed(2)}\n`;

    setStatus('loading', 'Sampling latent with ONNX Runtime Web…');
    const latent = await sampleLatent({ ...textCond, ...speakerCond, latentSteps, numSteps, seed, cfgText, cfgSpeaker, cfgMinT, cfgMaxT });

    setStatus('loading', 'Decoding DACVAE audio…');
    refs.progressLabel.textContent = 'Decoding audio';
    refs.progressMeta.textContent = '283 MB';
    refs.progressFill.style.width = '100%';
    const audio = await decodeLatent(latent);
    const targetSamples = Math.min(audio.length, Math.floor(seconds * SAMPLE_RATE));
    const wav = float32ToWav(audio.slice(0, targetSamples), SAMPLE_RATE);
    displayAudio(wav, { seconds, steps: numSteps, seed, referenceName: speakerCond.referenceName });
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
  }
}

document.querySelectorAll('.sample-chip').forEach(button => {
  button.addEventListener('click', () => {
    refs.textInput.value = button.dataset.text || '';
    refs.textInput.focus();
  });
});

refs.resetBtn.addEventListener('click', resetParams);
refs.generateBtn.addEventListener('click', generateSpeech);

resetParams();
setDot(refs.dotRuntime, 'done');
preloadModelFiles();
