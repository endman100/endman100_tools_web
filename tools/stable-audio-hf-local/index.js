const TRANSFORMERS_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.0.0-next.11';
const ORT_CDN = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.23.2/dist/';
const HF_RESOLVE = 'https://huggingface.co/{repo}/resolve/main/{path}';
const HF_COOKIE = 'endman_hf_access_token';
const MANIFEST_URL = './model-manifest.json';

let MODEL_PRESETS = {
  'sa3-sm-music': {
    label: 'Stable Audio 3 Small Music ONNX',
    engine: 'sa3-onnx',
    repo: 'stabilityai/stable-audio-3-optimized',
    ditPath: 'onnx/sa3-sm-music/dit.onnx',
    decoderPath: 'onnx/same-s/dec_dynamic_bf16.onnx',
    textEncoderPath: 'onnx/t5gemma/encoder.onnx',
    tokenizerId: '{repo}',
    tokenizerPath: 'tensorRT/sm_90/t5gemma/tokenizer.json',
    tokenizerSubfolder: 'tensorRT/sm_90/t5gemma',
    tokenizerFallbackId: 'google/t5gemma-b-b-ul2',
    maxDuration: 11,
    defaultDuration: 3,
    defaultSteps: 2,
    maxSteps: 8,
    defaultCfg: 1,
    notes: 'Official SA3 optimized ONNX. Very large download; WebGPU strongly recommended.',
  },
  'sa3-sm-sfx': {
    label: 'Stable Audio 3 Small SFX ONNX',
    engine: 'sa3-onnx',
    repo: 'stabilityai/stable-audio-3-optimized',
    ditPath: 'onnx/sa3-sm-sfx/dit.onnx',
    decoderPath: 'onnx/same-s/dec_dynamic_bf16.onnx',
    textEncoderPath: 'onnx/t5gemma/encoder.onnx',
    tokenizerId: '{repo}',
    tokenizerPath: 'tensorRT/sm_90/t5gemma/tokenizer.json',
    tokenizerSubfolder: 'tensorRT/sm_90/t5gemma',
    tokenizerFallbackId: 'google/t5gemma-b-b-ul2',
    maxDuration: 11,
    defaultDuration: 3,
    defaultSteps: 2,
    maxSteps: 8,
    defaultCfg: 1,
    notes: 'Official SA3 optimized ONNX for sound effects. Same browser constraints as Small Music.',
  },
  'musicgen-small': {
    label: 'MusicGen Small Transformers.js',
    engine: 'transformers-js',
    modelId: 'Xenova/musicgen-small',
    repo: 'Xenova/musicgen-small',
    maxDuration: 8,
    defaultDuration: 3,
    defaultSteps: 64,
    maxSteps: 256,
    defaultCfg: 3,
    notes: 'Transformers.js text-to-audio model; useful for fast local-browser validation.',
  },
};

const SAMPLE_RATE = 44100;
const SAMPLES_PER_LATENT = 4096;
const SA3_CHANNELS = 256;
const SAME_PATCH_SIZE = 256;
const PROMPT_TOKENS = 256;

const refs = {
  statusBadge: document.getElementById('statusBadge'),
  statusMsg: document.getElementById('statusMsg'),
  backendBadge: document.getElementById('backendBadge'),
  progressWrap: document.getElementById('progressWrap'),
  progressLabel: document.getElementById('progressLabel'),
  progressMeta: document.getElementById('progressMeta'),
  progressFill: document.getElementById('progressFill'),
  dotRuntime: document.getElementById('dot-runtime'),
  dotModel: document.getElementById('dot-model'),
  dotGenerate: document.getElementById('dot-generate'),
  modelSelect: document.getElementById('modelSelect'),
  deviceSelect: document.getElementById('deviceSelect'),
  hfTokenInput: document.getElementById('hfTokenInput'),
  loadBtn: document.getElementById('loadBtn'),
  validateBtn: document.getElementById('validateBtn'),
  clearCacheBtn: document.getElementById('clearCacheBtn'),
  repoInput: document.getElementById('repoInput'),
  notesInput: document.getElementById('notesInput'),
  durationInput: document.getElementById('durationInput'),
  stepsInput: document.getElementById('stepsInput'),
  cfgInput: document.getElementById('cfgInput'),
  seedInput: document.getElementById('seedInput'),
  promptInput: document.getElementById('promptInput'),
  generateBtn: document.getElementById('generateBtn'),
  resetBtn: document.getElementById('resetBtn'),
  resultsArea: document.getElementById('resultsArea'),
  logCard: document.getElementById('logCard'),
  runLog: document.getElementById('runLog'),
};

const modelState = {
  key: '',
  preset: null,
  transformers: null,
  musicgenModel: null,
  musicgenTokenizer: null,
  tokenizer: null,
  sa3: null,
};

let audioUrl = null;
let abortController = null;
let loadedManifest = null;

function setStatus(type, message) {
  refs.statusBadge.className = `status-badge ${type}`;
  refs.statusBadge.textContent = type.toUpperCase();
  refs.statusMsg.textContent = message;
}

function setDot(dot, state) {
  dot.className = `dot ${state}`;
}

function setProgress(label, meta, pct) {
  refs.progressWrap.classList.remove('hidden');
  refs.progressLabel.textContent = label;
  refs.progressMeta.textContent = meta || '';
  refs.progressFill.style.width = `${Math.max(0, Math.min(100, pct))}%`;
}

function hideProgress() {
  refs.progressWrap.classList.add('hidden');
  refs.progressFill.style.width = '0%';
}

function logLine(message) {
  refs.logCard.classList.remove('hidden');
  refs.runLog.textContent += `${message}\n`;
  refs.runLog.scrollTop = refs.runLog.scrollHeight;
}

function selectedBasePreset() {
  return MODEL_PRESETS[refs.modelSelect.value] || MODEL_PRESETS['sa3-sm-music'];
}

function selectedPreset() {
  const preset = { ...selectedBasePreset() };
  const repo = refs.repoInput?.value.trim();
  if (preset.engine === 'sa3-onnx' && repo) preset.repo = repo;
  return preset;
}

function updateModelUi(resetValues = true, resetRepo = true) {
  const preset = selectedBasePreset();
  if (resetRepo) refs.repoInput.value = preset.repo;
  refs.notesInput.value = preset.notes;
  refs.durationInput.max = String(preset.maxDuration);
  refs.stepsInput.max = String(preset.maxSteps);
  refs.stepsInput.min = preset.engine === 'sa3-onnx' ? '1' : '16';
  refs.cfgInput.min = preset.engine === 'sa3-onnx' ? '1' : '1';
  refs.backendBadge.textContent = preset.engine === 'sa3-onnx' ? 'ONNX Runtime' : 'Transformers.js';

  if (resetValues) {
    refs.durationInput.value = String(preset.defaultDuration);
    refs.stepsInput.value = String(preset.defaultSteps);
    refs.cfgInput.value = String(preset.defaultCfg);
  }

  if (modelState.key !== refs.modelSelect.value) unloadModel(false);
  const manifestText = loadedManifest?.generatedAt ? ` Manifest ${loadedManifest.generatedAt}.` : '';
  setStatus('info', `Selected ${preset.label}.${manifestText}`);
}

async function loadModelManifest() {
  try {
    const response = await fetch(MANIFEST_URL, { cache: 'no-store' });
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    const manifest = await response.json();
    if (manifest?.presets && typeof manifest.presets === 'object') {
      for (const [key, value] of Object.entries(manifest.presets)) {
        if (MODEL_PRESETS[key]) MODEL_PRESETS[key] = { ...MODEL_PRESETS[key], ...value };
      }
    }
    loadedManifest = manifest;
  } catch (error) {
    loadedManifest = null;
    logLine(`[manifest] using built-in defaults: ${error.message}`);
  }
}

function saveToken() {
  const token = refs.hfTokenInput.value.trim();
  if (token) {
    document.cookie = `${HF_COOKIE}=${encodeURIComponent(token)}; max-age=31536000; path=/; SameSite=Lax`;
  }
}

function loadToken() {
  const found = document.cookie.split('; ').find(row => row.startsWith(`${HF_COOKIE}=`));
  refs.hfTokenInput.value = found ? decodeURIComponent(found.slice(HF_COOKIE.length + 1)) : '';
}

function unloadModel(resetDots = true) {
  modelState.key = '';
  modelState.preset = null;
  modelState.musicgenModel = null;
  modelState.musicgenTokenizer = null;
  modelState.tokenizer = null;
  modelState.sa3 = null;
  if (resetDots) {
    setDot(refs.dotModel, 'pending');
    setDot(refs.dotGenerate, 'pending');
  }
}

async function ensureTransformers() {
  if (modelState.transformers) return modelState.transformers;
  setDot(refs.dotRuntime, 'loading');
  setProgress('Loading Transformers.js runtime', '@huggingface/transformers', 5);
  modelState.transformers = await import(TRANSFORMERS_URL);
  setDot(refs.dotRuntime, 'done');
  return modelState.transformers;
}

function configureTransformersHub(env) {
  env.allowLocalModels = false;
  env.useBrowserCache = true;
  env.fetch = (url, init = {}) => {
    const headers = new Headers(init.headers || {});
    const token = refs.hfTokenInput.value.trim();
    if (token && String(url).startsWith('https://huggingface.co/')) {
      headers.set('Authorization', `Bearer ${token}`);
    }
    return fetch(url, { ...init, headers });
  };
}

function configureOrt() {
  if (!window.ort) throw new Error('ONNX Runtime Web script did not load.');
  ort.env.wasm.wasmPaths = ORT_CDN;
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = true;
}

function providers() {
  return refs.deviceSelect.value === 'webgpu' && navigator.gpu ? ['webgpu', 'wasm'] : ['wasm'];
}

function hfUrl(repo, path) {
  return HF_RESOLVE.replace('{repo}', repo).replace('{path}', path.split('/').map(encodeURIComponent).join('/'));
}

function tokenizerRepoForPreset(preset) {
  return preset.tokenizerId === '{repo}' ? preset.repo : (preset.tokenizerId || preset.repo);
}

function authHeaders() {
  const token = refs.hfTokenInput.value.trim();
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function fetchArrayBufferWithProgress(url, label, weight, offset) {
  setProgress(`Downloading ${label}`, 'starting', offset);
  const response = await fetch(url, { headers: authHeaders(), signal: abortController?.signal });
  if (!response.ok) throw new Error(`${label} download failed: ${response.status} ${response.statusText}`);
  const total = Number(response.headers.get('content-length')) || 0;
  const reader = response.body?.getReader();
  if (!reader) return response.arrayBuffer();
  const chunks = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    const pct = total ? offset + (received / total) * weight : offset;
    const meta = total ? `${formatBytes(received)} / ${formatBytes(total)}` : formatBytes(received);
    setProgress(`Downloading ${label}`, meta, pct);
  }
  const merged = new Uint8Array(received);
  let cursor = 0;
  for (const chunk of chunks) {
    merged.set(chunk, cursor);
    cursor += chunk.byteLength;
  }
  return merged.buffer;
}

async function createOrtSessionFromHf(preset, path, label, weight, offset) {
  const buffer = await fetchArrayBufferWithProgress(hfUrl(preset.repo, path), label, weight, offset);
  setProgress(`Initializing ${label}`, providers().join(' > '), offset + weight);
  try {
    return await ort.InferenceSession.create(buffer, {
      executionProviders: providers(),
      graphOptimizationLevel: 'all',
    });
  } catch (error) {
    if (providers()[0] === 'webgpu') {
      logLine(`[fallback] ${label} WebGPU init failed: ${error.message}`);
      return ort.InferenceSession.create(buffer, {
        executionProviders: ['wasm'],
        graphOptimizationLevel: 'all',
      });
    }
    throw error;
  }
}

async function loadModel() {
  const preset = selectedPreset();
  saveToken();
  abortController = new AbortController();
  refs.loadBtn.disabled = true;
  refs.generateBtn.disabled = true;
  refs.runLog.textContent = '';
  refs.logCard.classList.add('hidden');
  setDot(refs.dotModel, 'loading');
  setDot(refs.dotGenerate, 'pending');
  setStatus('loading', `Loading ${preset.label}...`);

  try {
    if (preset.engine === 'transformers-js') await loadMusicGen(preset);
    else {
      await preflightStableAudio3Files(preset);
      await loadStableAudio3Onnx(preset);
    }
    modelState.key = refs.modelSelect.value;
    modelState.preset = preset;
    setDot(refs.dotModel, 'done');
    setStatus('success', `${preset.label} loaded in browser.`);
    setProgress('Model ready', preset.engine, 100);
  } catch (error) {
    console.error(error);
    unloadModel(false);
    setDot(refs.dotModel, 'error');
    setStatus('error', error.message);
    logLine(`[error] ${error.message}`);
  } finally {
    refs.loadBtn.disabled = false;
    refs.generateBtn.disabled = false;
    abortController = null;
  }
}

async function validateModelFiles() {
  const preset = selectedPreset();
  abortController = new AbortController();
  refs.validateBtn.disabled = true;
  setStatus('loading', `Checking ${preset.repo || preset.modelId}...`);
  try {
    if (preset.engine === 'transformers-js') {
      await probeHfFile(preset.repo, 'config.json', preset.label);
    } else {
      await preflightStableAudio3Files(preset);
    }
    setStatus('success', 'Required remote model files are reachable.');
    setProgress('Validated model files', preset.repo || preset.modelId, 100);
  } catch (error) {
    console.error(error);
    setStatus('error', error.message);
    logLine(`[validate] ${error.message}`);
  } finally {
    refs.validateBtn.disabled = false;
    abortController = null;
  }
}

async function preflightStableAudio3Files(preset) {
  if (!preset.repo) throw new Error('HF model repository is required.');
  const files = [
    ['T5Gemma encoder ONNX', preset.textEncoderPath],
    ['SA3 DiT ONNX', preset.ditPath],
    ['SAME-S decoder ONNX', preset.decoderPath],
  ];
  for (let index = 0; index < files.length; index++) {
    const [label, path] = files[index];
    setProgress(`Checking ${label}`, path, 4 + index * 4);
    await probeHfFile(preset.repo, path, label);
  }
  const tokenizerRepo = tokenizerRepoForPreset(preset);
  const tokenizerPath = preset.tokenizerPath || 'tokenizer_config.json';
  setProgress('Checking tokenizer access', tokenizerRepo, 16);
  await probeHfFile(tokenizerRepo, tokenizerPath, 'T5Gemma tokenizer');
}

async function loadMusicGen(preset) {
  const { AutoTokenizer, MusicgenForConditionalGeneration, env } = await ensureTransformers();
  configureTransformersHub(env);
  env.backends.onnx.wasm.wasmPaths = ORT_CDN;
  env.backends.onnx.wasm.numThreads = 1;
  const options = {
    device: refs.deviceSelect.value === 'webgpu' && navigator.gpu ? 'webgpu' : 'wasm',
    dtype: {
      text_encoder: 'q8',
      decoder_model_merged: refs.deviceSelect.value === 'webgpu' && navigator.gpu ? 'q4f16' : 'q8',
      encodec_decode: 'fp32',
    },
    progress_callback: progress => {
      const file = progress.file || preset.modelId;
      const pct = Number.isFinite(progress.progress) ? 10 + progress.progress * 0.7 : 10;
      setProgress('Loading MusicGen', `${file} ${Math.round(progress.progress || 0)}%`, pct);
    },
  };
  logLine(`[load] Transformers.js ${preset.modelId} device=${options.device}`);
  modelState.musicgenTokenizer = await AutoTokenizer.from_pretrained(preset.modelId, {
    progress_callback: options.progress_callback,
  });
  modelState.musicgenModel = await MusicgenForConditionalGeneration.from_pretrained(preset.modelId, options);
}

async function loadStableAudio3Onnx(preset) {
  configureOrt();
  await ensureTransformers();
  logLine(`[load] ORT providers=${providers().join(',')}`);
  logLine(`[load] repo=${preset.repo}`);
  modelState.tokenizer = await loadSa3Tokenizer(preset);
  modelState.sa3 = {
    textEncoder: await createOrtSessionFromHf(preset, preset.textEncoderPath, 'T5Gemma encoder', 24, 8),
    dit: await createOrtSessionFromHf(preset, preset.ditPath, 'SA3 DiT', 42, 34),
    decoder: await createOrtSessionFromHf(preset, preset.decoderPath, 'SAME-S decoder', 18, 76),
  };
  logLine(`[session] t5 inputs=${modelState.sa3.textEncoder.inputNames.join(', ')}`);
  logLine(`[session] dit inputs=${modelState.sa3.dit.inputNames.join(', ')}`);
  logLine(`[session] decoder inputs=${modelState.sa3.decoder.inputNames.join(', ')}`);
}

async function loadSa3Tokenizer(preset) {
  const { PreTrainedTokenizer, env } = await ensureTransformers();
  configureTransformersHub(env);
  const tokenizerRepo = tokenizerRepoForPreset(preset);
  const tokenizerPath = preset.tokenizerPath || 'tokenizer_config.json';
  setProgress('Loading T5Gemma tokenizer', tokenizerRepo, 8);
  await probeHfFile(tokenizerRepo, tokenizerPath, 'T5Gemma tokenizer');
  try {
    const tokenizerJson = await fetchHfJson(tokenizerRepo, tokenizerPath, 'T5Gemma tokenizer');
    return new PreTrainedTokenizer(tokenizerJson, {
      tokenizer_class: 'GemmaTokenizer',
      model_max_length: 2048,
      bos_token: '<bos>',
      eos_token: '<eos>',
      unk_token: '<unk>',
      pad_token: '<pad>',
      padding_side: 'right',
    });
  } catch (error) {
    throw new Error(`T5Gemma tokenizer load failed from ${tokenizerRepo}/${tokenizerPath}. Details: ${error.message}`);
  }
}

async function fetchHfJson(repo, path, label) {
  const response = await fetch(hfUrl(repo, path), {
    headers: authHeaders(),
    signal: abortController?.signal,
  });
  if (response.status === 401 || response.status === 403) {
    throw new Error(`${label} requires Hugging Face access. Accept the model license on Hugging Face, then paste an HF access token here.`);
  }
  if (!response.ok) throw new Error(`${label} JSON fetch failed: ${response.status} ${response.statusText}`);
  return response.json();
}

async function probeHfFile(repo, path, label) {
  let response = await fetch(hfUrl(repo, path), {
    method: 'HEAD',
    headers: authHeaders(),
    signal: abortController?.signal,
  });
  if (response.status === 405 || response.status === 501) {
    response = await fetch(hfUrl(repo, path), {
      headers: { ...authHeaders(), Range: 'bytes=0-0' },
      signal: abortController?.signal,
    });
  }
  if (response.status === 401 || response.status === 403) {
    throw new Error(`${label} requires Hugging Face access. Accept the model license on Hugging Face, then paste an HF access token here.`);
  }
  if (!response.ok) throw new Error(`${label} access check failed: ${response.status} ${response.statusText}`);
}

async function generateAudio() {
  const preset = selectedPreset();
  if (modelState.key !== refs.modelSelect.value) await loadModel();
  if (!modelState.key) return;

  abortController = new AbortController();
  refs.generateBtn.disabled = true;
  refs.loadBtn.disabled = true;
  setDot(refs.dotGenerate, 'loading');
  setStatus('loading', `Generating with ${preset.label}...`);

  try {
    let result;
    if (preset.engine === 'transformers-js') result = await generateMusicGen(preset);
    else result = await generateStableAudio3(preset);
    displayAudio(result.audio, result.sampleRate, preset, result.meta);
    setDot(refs.dotGenerate, 'done');
    setStatus('success', `Generated audio with ${preset.label}.`);
    setProgress('Audio ready', `${result.audio.length} samples`, 100);
  } catch (error) {
    console.error(error);
    setDot(refs.dotGenerate, 'error');
    setStatus('error', error.message);
    refs.resultsArea.innerHTML = `<div class="results-placeholder"><span class="icon">ERR</span><p>${escapeHtml(error.message)}</p></div>`;
    logLine(`[error] ${error.message}`);
  } finally {
    refs.generateBtn.disabled = false;
    refs.loadBtn.disabled = false;
    abortController = null;
  }
}

async function generateMusicGen(preset) {
  const prompt = promptText();
  const maxTokens = clampInt(refs.stepsInput, preset.defaultSteps, 16, preset.maxSteps);
  const guidance = clampFloat(refs.cfgInput, preset.defaultCfg, 1, 8);
  setProgress('Running MusicGen', `${maxTokens} tokens`, 20);
  logLine(`[generate] MusicGen prompt=${JSON.stringify(prompt)}`);
  const inputs = modelState.musicgenTokenizer(prompt);
  const audioValues = await modelState.musicgenModel.generate({
    ...inputs,
    max_new_tokens: maxTokens,
    do_sample: true,
    guidance_scale: guidance,
  });
  if (!audioValues?.data) throw new Error('MusicGen returned no audio tensor.');
  const sampleRate = modelState.musicgenModel.config.audio_encoder.sampling_rate || 32000;
  return { audio: normalizeAudioArray(audioValues.data), sampleRate, meta: `${maxTokens} tokens · cfg ${guidance}` };
}

async function generateStableAudio3(preset) {
  const prompt = promptText();
  const seconds = clampFloat(refs.durationInput, preset.defaultDuration, 1, preset.maxDuration);
  const steps = clampInt(refs.stepsInput, preset.defaultSteps, 1, preset.maxSteps);
  const cfg = clampFloat(refs.cfgInput, preset.defaultCfg, 1, 8);
  const seed = seedValue();
  let latentSteps = Math.max(1, Math.ceil(seconds * SAMPLE_RATE / SAMPLES_PER_LATENT));
  if (latentSteps % 2 !== 0) latentSteps += 1;

  logLine(`[generate] SA3 prompt=${JSON.stringify(prompt)}`);
  logLine(`[generate] seconds=${seconds} steps=${steps} cfg=${cfg} seed=${seed} T_lat=${latentSteps}`);

  const { embeds, mask } = await encodePrompt(prompt);
  const cross = buildCrossAttention(embeds, mask, seconds);
  const global = cross.slice(PROMPT_TOKENS * 768, PROMPT_TOKENS * 768 + 768);
  let x = seededNormal(seed, SA3_CHANNELS * latentSteps, 0.9);
  const schedule = buildSchedule(steps);

  for (let index = 0; index < steps; index++) {
    const tCurr = schedule[index];
    const tNext = schedule[index + 1];
    setProgress('Running SA3 DiT', `step ${index + 1} / ${steps}`, 28 + (index / steps) * 46);
    const velocity = await runDit(x, tCurr, cross, global, latentSteps, cfg);
    const denoised = new Float32Array(x.length);
    for (let i = 0; i < x.length; i++) denoised[i] = x[i] - tCurr * velocity[i];
    if (index < steps - 1 && tNext > 0) {
      const noise = seededNormal(seed + index + 1000, x.length, 0.9);
      for (let i = 0; i < x.length; i++) x[i] = (1 - tNext) * denoised[i] + tNext * noise[i];
    } else {
      x = denoised;
    }
  }

  setProgress('Decoding SAME-S latents', `${latentSteps} latent frames`, 80);
  const patches = await runDecoder(x, latentSteps);
  const audio = unpatchAudio(patches, seconds);
  return { audio, sampleRate: SAMPLE_RATE, meta: `${steps} SA3 ONNX steps · T_lat ${latentSteps}` };
}

async function encodePrompt(prompt) {
  setProgress('Encoding prompt', 'T5Gemma', 18);
  const encoded = await modelState.tokenizer([prompt], {
    padding: 'max_length',
    truncation: true,
    max_length: PROMPT_TOKENS,
    return_tensor: false,
  });
  const inputIds = toBigInt64(flattenTokenField(encoded.input_ids));
  const maskValues = flattenTokenField(encoded.attention_mask).map(Number);
  const feeds = feedByName(modelState.sa3.textEncoder, {
    input_ids: new ort.Tensor('int64', inputIds, [1, PROMPT_TOKENS]),
    attention_mask: new ort.Tensor('int64', toBigInt64(maskValues), [1, PROMPT_TOKENS]),
  });
  const output = await modelState.sa3.textEncoder.run(feeds);
  const tensor = output.last_hidden_state || output[Object.keys(output)[0]];
  return { embeds: tensorToFloat32(tensor), mask: maskValues };
}

function buildCrossAttention(embeds, mask, seconds) {
  const cross = new Float32Array((PROMPT_TOKENS + 1) * 768);
  const padding = new Float32Array(768);
  for (let token = 0; token < PROMPT_TOKENS; token++) {
    const valid = mask[token] ? 1 : 0;
    const src = token * 768;
    for (let channel = 0; channel < 768; channel++) {
      cross[src + channel] = valid ? embeds[src + channel] : padding[channel];
    }
  }
  cross.set(secondsFourierToken(seconds), PROMPT_TOKENS * 768);
  return cross;
}

async function runDit(x, t, cross, global, latentSteps, cfg) {
  const session = modelState.sa3.dit;
  const feeds = feedByName(session, {
    x: new ort.Tensor('float32', x, [1, SA3_CHANNELS, latentSteps]),
    t: new ort.Tensor('float32', new Float32Array([t]), [1]),
    cross_attn_cond_raw: new ort.Tensor('float32', cross, [1, PROMPT_TOKENS + 1, 768]),
    cross_attn_cond: new ort.Tensor('float32', cross, [1, PROMPT_TOKENS + 1, 768]),
    global_cond_raw: new ort.Tensor('float32', global, [1, 768]),
    global_cond: new ort.Tensor('float32', global, [1, 768]),
    local_add_cond: new ort.Tensor('float32', new Float32Array(latentSteps * 257), [1, latentSteps, 257]),
    cfg_scale: new ort.Tensor('float32', new Float32Array([cfg]), [1]),
  });
  const output = await session.run(feeds);
  const tensor = output[Object.keys(output)[0]];
  return tensorToFloat32(tensor);
}

async function runDecoder(latents, latentSteps) {
  const session = modelState.sa3.decoder;
  const feeds = feedByName(session, {
    latents: new ort.Tensor('float32', latents, [1, SA3_CHANNELS, latentSteps]),
    x: new ort.Tensor('float32', latents, [1, SA3_CHANNELS, latentSteps]),
  });
  const output = await session.run(feeds);
  const tensor = output[Object.keys(output)[0]];
  return tensorToFloat32(tensor);
}

function feedByName(session, candidates) {
  const feeds = {};
  for (const name of session.inputNames) {
    if (candidates[name]) {
      feeds[name] = candidates[name];
      continue;
    }
    const normalized = name.split(':')[0].split('/').pop();
    if (candidates[normalized]) feeds[name] = candidates[normalized];
  }
  const missing = session.inputNames.filter(name => !feeds[name]);
  if (missing.length) throw new Error(`Cannot map ONNX inputs: ${missing.join(', ')}`);
  return feeds;
}

function secondsFourierToken(seconds) {
  const out = new Float32Array(768);
  const normalized = Math.max(0, Math.min(384, seconds)) / 384;
  for (let i = 0; i < 128; i++) {
    const ramp = i / 127;
    const freq = Math.exp(ramp * (Math.log(10000) - Math.log(0.5)) + Math.log(0.5));
    const angle = normalized * freq * Math.PI * 2;
    out[i] = Math.cos(angle);
    out[i + 128] = Math.sin(angle);
  }
  return out;
}

function buildSchedule(steps) {
  const values = [];
  for (let i = 0; i <= steps; i++) {
    const raw = 1 - i / steps;
    const logsnr = 2.0 - raw * (2.0 - -6.2);
    let shifted = 1 / (1 + Math.exp(logsnr));
    if (raw <= 0) shifted = 0;
    if (raw >= 1) shifted = 1;
    values.push(i === 0 ? 1 : shifted);
  }
  return values;
}

function unpatchAudio(patches, seconds) {
  const totalSamples = Math.round(seconds * SAMPLE_RATE);
  const channels = 2;
  const patchFrames = Math.floor(patches.length / (channels * SAME_PATCH_SIZE));
  const audio = new Float32Array(Math.min(totalSamples, patchFrames * SAME_PATCH_SIZE) * channels);
  let cursor = 0;
  for (let frame = 0; frame < patchFrames && cursor < audio.length; frame++) {
    for (let sample = 0; sample < SAME_PATCH_SIZE && cursor < audio.length; sample++) {
      for (let channel = 0; channel < channels; channel++) {
        const sourceIndex = channel * SAME_PATCH_SIZE * patchFrames + sample * patchFrames + frame;
        audio[cursor++] = Math.max(-1, Math.min(1, patches[sourceIndex] || 0));
      }
    }
  }
  return audio;
}

function displayAudio(audio, sampleRate, preset, meta) {
  if (audioUrl) URL.revokeObjectURL(audioUrl);
  const wav = encodeWav(audio, sampleRate);
  const blob = new Blob([wav], { type: 'audio/wav' });
  audioUrl = URL.createObjectURL(blob);
  refs.resultsArea.innerHTML = `
    <div class="results-list">
      <div class="result-item">
        <div class="result-meta">
          <span>${escapeHtml(preset.label)}</span>
          <span>${sampleRate} Hz</span>
          <span>${escapeHtml(meta)}</span>
          <span>${formatBytes(blob.size)}</span>
        </div>
        <audio controls preload="metadata" src="${audioUrl}"></audio>
        <div class="result-actions">
          <a class="btn btn-ghost" href="${audioUrl}" download="${safeFileName(preset.label)}-${Date.now()}.wav">Download WAV</a>
        </div>
      </div>
    </div>`;
}

function encodeWav(samples, sampleRate) {
  const channels = samples.length % 2 === 0 ? 2 : 1;
  const frameCount = Math.floor(samples.length / channels);
  const bytesPerSample = 2;
  const blockAlign = channels * bytesPerSample;
  const buffer = new ArrayBuffer(44 + frameCount * blockAlign);
  const view = new DataView(buffer);
  writeString(view, 0, 'RIFF');
  view.setUint32(4, 36 + frameCount * blockAlign, true);
  writeString(view, 8, 'WAVE');
  writeString(view, 12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, channels, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * blockAlign, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true);
  writeString(view, 36, 'data');
  view.setUint32(40, frameCount * blockAlign, true);
  let offset = 44;
  for (let i = 0; i < frameCount * channels; i++, offset += 2) {
    const value = Math.max(-1, Math.min(1, samples[i] || 0));
    view.setInt16(offset, value < 0 ? value * 32768 : value * 32767, true);
  }
  return buffer;
}

function writeString(view, offset, text) {
  for (let i = 0; i < text.length; i++) view.setUint8(offset + i, text.charCodeAt(i));
}

function normalizeAudioArray(audio) {
  if (audio instanceof Float32Array) return audio;
  if (Array.isArray(audio)) return Float32Array.from(audio.flat(Infinity));
  if (audio?.data) return normalizeAudioArray(audio.data);
  return Float32Array.from(audio);
}

function tensorToFloat32(tensor) {
  if (tensor.data instanceof Float32Array) return tensor.data;
  if (tensor.type === 'float16' || tensor.type === 'bfloat16') return float16ToFloat32(tensor.data);
  return Float32Array.from(tensor.data);
}

function float16ToFloat32(data) {
  const out = new Float32Array(data.length);
  for (let i = 0; i < data.length; i++) out[i] = halfToFloat(data[i]);
  return out;
}

function halfToFloat(value) {
  const sign = (value & 0x8000) ? -1 : 1;
  const exponent = (value >> 10) & 0x1f;
  const fraction = value & 0x03ff;
  if (exponent === 0) return sign * Math.pow(2, -14) * (fraction / 1024);
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return sign * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
}

function seededNormal(seed, length, scale = 1) {
  const rng = mulberry32(seed || 1);
  const values = new Float32Array(length);
  for (let i = 0; i < length; i += 2) {
    const u1 = Math.max(rng(), 1e-7);
    const u2 = rng();
    const radius = Math.sqrt(-2 * Math.log(u1)) * scale;
    const theta = Math.PI * 2 * u2;
    values[i] = radius * Math.cos(theta);
    if (i + 1 < length) values[i + 1] = radius * Math.sin(theta);
  }
  return values;
}

function mulberry32(seed) {
  let value = seed >>> 0;
  return function random() {
    value += 0x6D2B79F5;
    let t = value;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function flattenTokenField(value) {
  if (Array.isArray(value)) return value.flat(Infinity);
  if (value?.data) return Array.from(value.data);
  return Array.from(value);
}

function toBigInt64(values) {
  const out = new BigInt64Array(values.length);
  for (let i = 0; i < values.length; i++) out[i] = BigInt(values[i]);
  return out;
}

function promptText() {
  const value = refs.promptInput.value.trim();
  if (!value) throw new Error('請輸入 prompt。');
  return value;
}

function seedValue() {
  const parsed = Number.parseInt(refs.seedInput.value.trim(), 10);
  return Number.isFinite(parsed) ? parsed : Math.floor(Math.random() * 2147483647);
}

function clampFloat(input, fallback, min, max) {
  const value = Number.parseFloat(input.value);
  if (!Number.isFinite(value)) return fallback;
  return Math.max(min, Math.min(max, value));
}

function clampInt(input, fallback, min, max) {
  const value = Number.parseInt(input.value, 10);
  if (!Number.isFinite(value)) return fallback;
  return Math.max(min, Math.min(max, value));
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return '0 B';
  if (bytes > 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
  if (bytes > 1024 ** 2) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  if (bytes > 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${bytes} B`;
}

function safeFileName(value) {
  return String(value).toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/(^-|-$)/g, '');
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, char => ({
    '&': '&amp;',
    '<': '&lt;',
    '>': '&gt;',
    '"': '&quot;',
    "'": '&#39;',
  }[char]));
}

async function clearModelCache() {
  if (!window.confirm('清除 Hugging Face / ONNX 瀏覽器快取？下次會重新下載模型。')) return;
  if ('caches' in window) {
    const keys = await caches.keys();
    await Promise.all(keys.filter(key => /transformers|huggingface|onnx/i.test(key)).map(key => caches.delete(key)));
  }
  unloadModel();
  setStatus('success', '模型快取清除請求已完成。');
}

function resetForm() {
  refs.modelSelect.value = 'sa3-sm-music';
  refs.deviceSelect.value = navigator.gpu ? 'webgpu' : 'wasm';
  refs.promptInput.value = 'Impending tribal, epic orchestral buildup with huge drums, tense strings, and a dramatic final hit.';
  refs.seedInput.value = '';
  refs.runLog.textContent = '';
  refs.logCard.classList.add('hidden');
  refs.resultsArea.innerHTML = '<div class="results-placeholder"><span class="icon">WAV</span><p>生成完成後會在這裡播放與下載。</p></div>';
  unloadModel();
  hideProgress();
  updateModelUi(true);
}

function attachEvents() {
  refs.modelSelect.addEventListener('change', () => updateModelUi(true));
  refs.deviceSelect.addEventListener('change', () => unloadModel());
  refs.repoInput.addEventListener('change', () => unloadModel());
  refs.hfTokenInput.addEventListener('change', saveToken);
  refs.loadBtn.addEventListener('click', loadModel);
  refs.validateBtn.addEventListener('click', validateModelFiles);
  refs.generateBtn.addEventListener('click', generateAudio);
  refs.resetBtn.addEventListener('click', resetForm);
  refs.clearCacheBtn.addEventListener('click', clearModelCache);
  document.querySelectorAll('.sample-chip').forEach(button => {
    button.addEventListener('click', () => {
      refs.promptInput.value = button.dataset.text || '';
      refs.promptInput.focus();
    });
  });
}

async function init() {
  await loadModelManifest();
  attachEvents();
  loadToken();
  refs.deviceSelect.value = navigator.gpu ? 'webgpu' : 'wasm';
  updateModelUi(true);
  setDot(refs.dotRuntime, window.ort ? 'done' : 'error');
  setStatus('info', 'HF local audio runtime ready.');
}

init().catch(error => {
  console.error(error);
  setStatus('error', error.message);
  logLine(`[init] ${error.message}`);
});
