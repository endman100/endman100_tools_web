const ORT_VERSION = "1.26.0";
const ORT_SCRIPT_URL = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/ort.webgpu.min.js`;
const MODEL_REPO = "endman100/sdxl-onnx";
const HF_MODEL_BASE = `https://huggingface.co/${MODEL_REPO}/resolve/main/`;
const HF_API_TREE_URL = `https://huggingface.co/api/models/${MODEL_REPO}/tree/main/onnx?recursive=true`;
const MANIFEST_URL = `${HF_MODEL_BASE}onnx/model-manifest.json`;
const CACHE_NAME = "endman100-sdxl-onnx-web-shared-v1";
const OPFS_DIR = "endman100-sdxl-onnx-webgpu-shared";
const DOWNLOAD_CHUNK_SIZE = 32 * 1024 * 1024;
const MAX_WEBGPU_ATTENTION_TOKENS = 16384;
const MIN_QUALITY_STEPS = 2;
const DEFAULT_QUALITY_STEPS = 8;
const MAX_QUALITY_STEPS = 40;
const RANDOM_SEED_SENTINEL = -1;
const SHARED_MODEL_MODULES = ["text_encoder", "text_encoder_2", "vae_encoder", "vae_decoder"];
const DEFAULT_UNET_MODULE = "unet_fp8";
const FALLBACK_UNET_MODULE = "unet";
const SDXL_LATENT_SCALE = 0.13025;
const searchParams = new URLSearchParams(location.search);
const SKIP_MODEL_WARM = searchParams.has("skipModelWarm");
const PROBE_MANIFEST = searchParams.has("probeManifest");
const AUTO_MODEL_WARM = searchParams.has("autoWarm") && !SKIP_MODEL_WARM;

const paths = {
  vocab: "./tokenizer/vocab.json",
  merges: "./tokenizer/merges.txt",
  vocab2: "./tokenizer_2/vocab.json",
  merges2: "./tokenizer_2/merges.txt",
  scheduler: "./scheduler_config.json"
};

const state = {
  loading: null,
  sessions: null,
  tokenizer: null,
  tokenizer2: null,
  scheduler: null,
  manifest: null,
  hfTree: null,
  manifestBase: null,
  cacheWarming: null,
  cacheReady: false,
  ortLoading: null,
  sharedLoading: null,
  sharedSessions: null,
  unetSession: null,
  unetModule: DEFAULT_UNET_MODULE
};

const preloadedBuffers = new Map();
const downloadPromises = new Map();

const statusBox = document.querySelector("#status");
const readyState = document.querySelector("#readyState");
const cacheState = document.querySelector("#cacheState");
const generateButton = document.querySelector("#generate");
const loadButton = document.querySelector("#loadModels");
const clearCacheButton = document.querySelector("#clearCache");
const widthInput = document.querySelector("#width");
const heightInput = document.querySelector("#height");
const guidanceInput = document.querySelector("#guidance");
const seedInput = document.querySelector("#seed");
const samplerInput = document.querySelector("#sampler");
const unetVariantInput = document.querySelector("#unetVariant");
const stepsInput = document.querySelector("#steps");
const strengthInput = document.querySelector("#strength");
const batchCountInput = document.querySelector("#batchCount");
const sourceFitInput = document.querySelector("#sourceFit");
const maskBlurInput = document.querySelector("#maskBlur");
const invertMaskInput = document.querySelector("#invertMask");
const modeInputs = Array.from(document.querySelectorAll('input[name="mode"]'));
const statusBadge = document.querySelector("#statusBadge");
const progressWrap = document.querySelector("#progressWrap");
const progressFill = document.querySelector("#progressFill");
const progressLabel = document.querySelector("#progressLabel");
const progressMeta = document.querySelector("#progressMeta");
const dotRuntime = document.querySelector("#dot-runtime");
const dotModel = document.querySelector("#dot-model");
const dotGenerate = document.querySelector("#dot-generate");
const sourceImageInput = document.querySelector("#sourceImage");
const sourceFrame = document.querySelector("#sourceFrame");
const sourceCanvas = document.querySelector("#sourceCanvas");
const maskCanvas = document.querySelector("#maskCanvas");
const brushSizeInput = document.querySelector("#brushSize");
const clearMaskButton = document.querySelector("#clearMask");
const sourcePlaceholder = document.querySelector("#sourcePlaceholder");
const outputFrame = document.querySelector(".canvas-frame");
const originalConsoleError = console.error.bind(console);
const sourceCtx = sourceCanvas?.getContext("2d");
const maskCtx = maskCanvas?.getContext("2d", { willReadFrequently: true });
let sourceImageBitmap = null;
let isMaskDrawing = false;
let maskEraseMode = false;
let lastMaskPoint = null;

console.error = (...args) => {
  const text = args.map(value => String(value)).join(" ");
  if (text.includes("[W:onnxruntime:") && text.includes("VerifyEachNodeIsAssignedToAnEp")) return;
  originalConsoleError(...args);
};

function setRunStatus(message) {
  statusBox.textContent = message;
}

function log(message) {
  statusBox.textContent += `\n${message}`;
  statusBox.scrollTop = statusBox.scrollHeight;
}

function setReady(message) {
  readyState.textContent = message;
}

function setStatusBadge(type) {
  if (!statusBadge) return;
  statusBadge.className = `status-badge ${type}`;
  statusBadge.textContent = type.toUpperCase();
}

function setDot(dot, stateName) {
  if (!dot) return;
  dot.className = `dot ${stateName}`;
}

function fmtBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "";
  const units = ["B", "KB", "MB", "GB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit++;
  }
  return `${value >= 10 || unit === 0 ? value.toFixed(0) : value.toFixed(1)} ${units[unit]}`;
}

function showProgress(label, meta = "", pct = 0) {
  if (!progressWrap) return;
  progressWrap.classList.remove("hidden");
  progressLabel.textContent = label;
  progressMeta.textContent = meta;
  progressFill.style.width = `${Math.max(0, Math.min(100, pct))}%`;
}

function hideProgress() {
  progressWrap?.classList.add("hidden");
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
    return dir.getFileHandle(def.cacheKey, { create });
  } catch {
    return null;
  }
}

async function opfsFile(def) {
  const handle = await opfsFileHandle(def, false);
  if (!handle) return null;
  try {
    const file = await handle.getFile();
    if (def.bytes && file.size !== def.bytes) {
      await removeOpfsFile(def);
      log(`Discarded incomplete cached file: ${def.path} (${fmtBytes(file.size)} / ${fmtBytes(def.bytes)}).`);
      return null;
    }
    return file.size > 0 ? file : null;
  } catch {
    return null;
  }
}

async function removeOpfsFile(def) {
  try {
    const dir = await opfsDir(false);
    await dir?.removeEntry(def.cacheKey);
  } catch {
    // Cache cleanup is best effort.
  }
}

async function cacheMatch(url) {
  if (!("caches" in window)) return null;
  try {
    const cache = await caches.open(CACHE_NAME);
    return cache.match(url);
  } catch {
    return null;
  }
}

async function removeCacheEntry(url) {
  if (!("caches" in window)) return;
  try {
    const cache = await caches.open(CACHE_NAME);
    await cache.delete(url);
  } catch {
    // Cache cleanup is best effort.
  }
}

async function ensureOrt() {
  if (window.ort) return window.ort;
  if (state.ortLoading) return state.ortLoading;

  state.ortLoading = new Promise((resolve, reject) => {
    setDot(dotRuntime, "loading");
    log(`Loading ONNX Runtime Web ${ORT_VERSION}...`);
    const script = document.createElement("script");
    script.src = ORT_SCRIPT_URL;
    script.async = true;
    script.onload = () => {
      if (!window.ort) {
        reject(new Error("ONNX Runtime Web script loaded, but window.ort is missing."));
        return;
      }
      window.ort.env.wasm.wasmPaths = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}/dist/`;
      window.ort.env.logLevel = "fatal";
      setDot(dotRuntime, "done");
      resolve(window.ort);
    };
    script.onerror = () => reject(new Error(`Failed to load ONNX Runtime Web ${ORT_VERSION}.`));
    document.head.appendChild(script);
  });

  try {
    return await state.ortLoading;
  } finally {
    state.ortLoading = null;
  }
}

function moduleUrl(path) {
  return new URL(path, state.manifestBase).toString();
}

async function cachedFetch(url, responseType = "arrayBuffer") {
  const cache = await caches.open(CACHE_NAME);
  let response = await cache.match(url);
  if (!response) {
    response = await fetch(url);
    if (!response.ok) throw new Error(`Fetch failed ${response.status}: ${url}`);
    try {
      await cache.put(url, response.clone());
    } catch (error) {
      log(`Cache skipped for ${new URL(url).pathname}: ${error.message}`);
    }
  }
  if (responseType === "json") return response.json();
  if (responseType === "text") return response.text();
  if (responseType === "response") return response;
  return response.arrayBuffer();
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function modelAssetDefs(moduleName, moduleInfo) {
  const externalData = externalDataEntries(moduleInfo);
  return [
    {
      moduleName,
      role: "model",
      path: moduleInfo.model ?? moduleInfo.file,
      bytes: moduleInfo.bytes ?? 0,
      label: `${moduleName} model`
    },
    ...externalData.map((entry, index) => ({
      moduleName,
      role: "external",
      path: entry.path,
      bytes: entry.bytes ?? 0,
      label: `${moduleName} external ${index + 1}`
    }))
  ].map(def => ({
    ...def,
    url: moduleUrl(def.path),
    cacheKey: def.path.replace(/[^\w.-]/g, "_")
  }));
}

function externalDataEntries(moduleInfo) {
  const raw = moduleInfo.externalData ?? moduleInfo.external_data ?? moduleInfo.external_data_files ?? [];
  const entries = Array.isArray(raw)
    ? raw
    : Array.isArray(raw.shards)
      ? raw.shards
      : Array.isArray(raw.files)
        ? raw.files
        : [raw];
  return entries
    .filter(Boolean)
    .map(entry => {
      const path = entry.path ?? entry.file ?? entry.filename ?? entry.name;
      if (!path) return null;
      return {
        path,
        bytes: entry.bytes ?? entry.size ?? 0
      };
    })
    .filter(Boolean);
}

function availableUnetModules(manifest = state.manifest) {
  const modules = manifest?.modules ?? {};
  const declared = manifest?.runtime?.available_unets ?? manifest?.runtime?.availableUnets ?? [];
  const candidates = declared.length ? declared : [DEFAULT_UNET_MODULE, "unet_bf16", "unet_fp32", FALLBACK_UNET_MODULE];
  return candidates.filter(moduleName => modules[moduleName]);
}

function selectedUnetModule(manifest = state.manifest) {
  const modules = manifest?.modules ?? {};
  const selected = unetVariantInput?.value || state.unetModule;
  const available = availableUnetModules(manifest);
  if (modules[selected]) return selected;
  if (modules[DEFAULT_UNET_MODULE]) return DEFAULT_UNET_MODULE;
  if (modules[FALLBACK_UNET_MODULE]) return FALLBACK_UNET_MODULE;
  if (available[0]) return available[0];
  throw new Error("No supported UNet module was found in the Hugging Face manifest.");
}

function unetLabel(moduleName, moduleInfo) {
  const dtype = String(moduleInfo?.dtype ?? moduleName.replace(/^unet_?/, "")).toUpperCase();
  const totalExternal = externalDataEntries(moduleInfo).reduce((sum, entry) => sum + (entry.bytes || 0), 0);
  const sizeLabel = totalExternal ? ` - ${fmtBytes(totalExternal)}` : "";
  const suffix = moduleName === DEFAULT_UNET_MODULE ? " recommended" : "";
  return `${dtype}${suffix}${sizeLabel}`;
}

function configureUnetSelector(manifest = state.manifest) {
  if (!unetVariantInput || !manifest?.modules) return;
  const available = availableUnetModules(manifest);
  if (!available.length) return;
  const previous = unetVariantInput.value || state.unetModule;
  unetVariantInput.innerHTML = "";
  for (const moduleName of available) {
    const option = document.createElement("option");
    option.value = moduleName;
    option.textContent = unetLabel(moduleName, manifest.modules[moduleName]);
    unetVariantInput.appendChild(option);
  }
  unetVariantInput.value = available.includes(previous)
    ? previous
    : available.includes(DEFAULT_UNET_MODULE)
      ? DEFAULT_UNET_MODULE
      : available[0];
  state.unetModule = unetVariantInput.value;
}

function modelAssetDefsForModules(moduleNames) {
  return moduleNames.flatMap(moduleName => {
    const moduleInfo = state.manifest.modules[moduleName];
    if (!moduleInfo) throw new Error(`Missing module in manifest: ${moduleName}`);
    return modelAssetDefs(moduleName, moduleInfo);
  });
}

function sharedModelAssetDefs() {
  return modelAssetDefsForModules(SHARED_MODEL_MODULES);
}

function selectedUnetAssetDefs() {
  return modelAssetDefsForModules([selectedUnetModule()]);
}

function allModelAssetDefs() {
  return modelAssetDefsForModules([...SHARED_MODEL_MODULES, selectedUnetModule()]);
}

function allKnownModelAssetDefs() {
  const modules = [...SHARED_MODEL_MODULES, ...availableUnetModules()];
  return modelAssetDefsForModules([...new Set(modules)]);
}

async function isCached(def) {
  try {
    if (preloadedBuffers.has(def.url)) return true;
    if (await opfsFile(def)) return true;
    return Boolean(await cacheMatch(def.url));
  } catch (error) {
    console.warn("Cache probe failed; treating as uncached", def.path, error);
    return false;
  }
}

async function cachedArrayBuffer(def) {
  if (preloadedBuffers.has(def.url)) return preloadedBuffers.get(def.url);
  const opfsCached = await opfsFile(def);
  if (opfsCached) {
    try {
      return await opfsCached.arrayBuffer();
    } catch (error) {
      console.warn("OPFS cached file read failed", def.path, error);
      await removeOpfsFile(def);
      log(`Cached OPFS file could not be read and was removed: ${def.path}. Retrying download...`);
      return null;
    }
  }
  const cached = await cacheMatch(def.url);
  if (cached) {
    try {
      return await cached.arrayBuffer();
    } catch (error) {
      console.warn("Cache API response read failed", def.path, error);
      await removeCacheEntry(def.url);
      log(`Cached response could not be read and was removed: ${def.path}. Retrying download...`);
      return null;
    }
  }
  return null;
}

async function downloadToOpfs(def, total) {
  if (!Number.isFinite(total) || total <= 0) return false;
  const handle = await opfsFileHandle(def, true);
  if (!handle) return false;
  const writable = await handle.createWritable();
  let loaded = 0;
  try {
    for (let start = 0; start < total; start += DOWNLOAD_CHUNK_SIZE) {
      const end = Math.min(total - 1, start + DOWNLOAD_CHUNK_SIZE - 1);
      const response = await fetch(def.url, {
        headers: { Range: `bytes=${start}-${end}` },
        cache: "no-store"
      });
      if (!(response.status === 206 || (start === 0 && response.ok && total <= DOWNLOAD_CHUNK_SIZE))) {
        throw new Error(`Range request failed: HTTP ${response.status}`);
      }
      const chunk = new Uint8Array(await response.arrayBuffer());
      await writable.write({ type: "write", position: start, data: chunk });
      loaded += chunk.byteLength;
      showProgress(`Downloading ${def.label}`, `${fmtBytes(loaded)} / ${fmtBytes(total)}`, (loaded / total) * 100);
      await sleep(0);
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
  if (await isCached(def)) return null;

  let total = def.bytes || 0;
  if (!total) {
    try {
      const head = await fetch(def.url, { method: "HEAD", cache: "no-store" });
      total = Number.parseInt(head.headers.get("content-length") || "0", 10);
    } catch {
      total = 0;
    }
  }

  showProgress(`Downloading ${def.label}`, fmtBytes(total), 8);
  try {
    if (await downloadToOpfs(def, total)) return null;
  } catch (error) {
    console.warn("OPFS cache failed; falling back for", def.path, error);
    await removeOpfsFile(def);
  }

  const response = await fetch(def.url);
  if (!response.ok) throw new Error(`HTTP ${response.status} downloading ${def.path}`);
  if ("caches" in window) {
    const cache = await caches.open(CACHE_NAME);
    try {
      await cache.put(def.url, response.clone());
      showProgress(`${def.label} cached`, fmtBytes(total), 100);
      return null;
    } catch (error) {
      console.warn("Cache API failed; falling back to memory buffer for", def.path, error);
    }
  }

  if (!response.body || total > 512 * 1024 * 1024) {
    throw new Error(`Browser cache failed for ${def.path}. Free browser storage or restart Chrome and try again.`);
  }

  const buffer = await response.arrayBuffer();
  preloadedBuffers.set(def.url, buffer);
  showProgress(`${def.label} downloaded`, fmtBytes(buffer.byteLength), 100);
  return buffer;
}

async function ensureCached(def) {
  if (await isCached(def)) {
    showProgress(`${def.label} loaded from cache`, fmtBytes(def.bytes), 100);
    return null;
  }
  if (!downloadPromises.has(def.url)) {
    downloadPromises.set(def.url, downloadToCache(def).finally(() => downloadPromises.delete(def.url)));
  }
  return downloadPromises.get(def.url);
}

async function fetchCached(def) {
  const transientBuffer = await ensureCached(def);
  if (transientBuffer) return transientBuffer;
  let buffer = await cachedArrayBuffer(def);
  if (!buffer) {
    preloadedBuffers.delete(def.url);
    await removeOpfsFile(def);
    await removeCacheEntry(def.url);
    await ensureCached(def);
    buffer = await cachedArrayBuffer(def);
  }
  if (!buffer) throw new Error(`Model file was not cached or could not be read after retry: ${def.path}`);
  return buffer;
}

async function loadHfOnnxTree() {
  if (state.hfTree) return state.hfTree;
  const response = await fetch(HF_API_TREE_URL, { cache: "force-cache" });
  if (!response.ok) throw new Error(`Failed to load Hugging Face ONNX tree: HTTP ${response.status}`);
  state.hfTree = await response.json();
  return state.hfTree;
}

function isOnnxSidecar(path) {
  const normalized = path.replace(/^onnx\//, "");
  return normalized
    && !normalized.includes("/")
    && !normalized.endsWith(".onnx")
    && normalized !== "model-manifest.json";
}

async function augmentSdxlManifestExternalData(manifest) {
  const unet = manifest.modules?.unet;
  if (!unet || externalDataEntries(unet).length > 0) return manifest;
  if ((unet.file ?? unet.model) !== "unet.onnx") return manifest;

  const tree = await loadHfOnnxTree();
  const sidecars = tree
    .filter(item => item.type === "file" && isOnnxSidecar(item.path))
    .map(item => ({
      path: item.path.replace(/^onnx\//, ""),
      bytes: item.size ?? 0
    }));

  if (sidecars.length > 0) {
    unet.externalData = sidecars;
    log(`Detected ${sidecars.length} SDXL UNet external data file(s) from Hugging Face tree.`);
  }
  return manifest;
}

async function loadManifest() {
  if (state.manifest) return state.manifest;
  const manifestUrl = new URL(MANIFEST_URL, location.href);
  state.manifestBase = manifestUrl;
  state.manifest = await cachedFetch(manifestUrl.toString(), "json");
  await augmentSdxlManifestExternalData(state.manifest);
  configureUnetSelector(state.manifest);
  return state.manifest;
}

async function loadSidecars() {
  if (state.tokenizer && state.tokenizer2 && state.scheduler) return;
  const [vocab, merges, vocab2, merges2, scheduler] = await Promise.all([
    cachedFetch(new URL(paths.vocab, location.href).toString(), "json"),
    cachedFetch(new URL(paths.merges, location.href).toString(), "text"),
    cachedFetch(new URL(paths.vocab2, location.href).toString(), "json"),
    cachedFetch(new URL(paths.merges2, location.href).toString(), "text"),
    cachedFetch(new URL(paths.scheduler, location.href).toString(), "json")
  ]);
  state.tokenizer = new ClipTokenizer(vocab, merges);
  state.tokenizer2 = new ClipTokenizer(vocab2, merges2);
  state.scheduler = scheduler;
}

async function warmModelCache() {
  if (state.cacheReady) return;
  if (state.cacheWarming) return state.cacheWarming;

  state.cacheWarming = (async () => {
    setStatusBadge("loading");
    setDot(dotRuntime, "pending");
    setDot(dotModel, "loading");
    setDot(dotGenerate, "pending");
    setReady("Downloading");
    cacheState.textContent = "downloading shared models";
    generateButton.disabled = true;
    setRunStatus(`Checking WebGPU, tokenizer, scheduler, and ${MODEL_REPO} manifest...`);
    await loadManifest();
    await loadSidecars();

    for (const def of sharedModelAssetDefs()) {
      log(`Caching ${def.label}: ${def.path}`);
      await ensureCached(def);
      await sleep(40);
    }

    state.cacheReady = true;
    cacheState.textContent = "shared models cached";
    setStatusBadge("success");
    setDot(dotModel, "done");
    setReady("Ready");
    hideProgress();
    log(`Shared model files cached from ${MODEL_REPO}. Pick a UNet precision; that UNet downloads when Generate is clicked.`);
    generateButton.disabled = false;
  })();

  try {
    return await state.cacheWarming;
  } finally {
    state.cacheWarming = null;
  }
}

function sessionOptions(moduleInfo, externalBuffers = new Map()) {
  const externalData = externalDataEntries(moduleInfo).map(entry => ({
    path: entry.path,
    data: externalBuffers.get(entry.path) ?? moduleUrl(entry.path)
  }));
  return {
    executionProviders: ["webgpu", "wasm"],
    externalData
  };
}

async function createSession(moduleName) {
  const moduleInfo = state.manifest.modules[moduleName];
  if (!moduleInfo) throw new Error(`Missing module in manifest: ${moduleName}`);
  const defs = modelAssetDefs(moduleName, moduleInfo);
  const modelDef = defs.find(def => def.role === "model");
  const externalDefs = defs.filter(def => def.role === "external");
  const externalBuffers = new Map();

  log(`Loading ${moduleName}...`);
  showProgress(`Creating ${moduleName} session`, moduleName, 10);
  const modelBytes = await fetchCached(modelDef);
  for (const def of externalDefs) {
    externalBuffers.set(def.path, await fetchCached(def));
  }
  const options = sessionOptions(moduleInfo, externalBuffers);
  showProgress(`Creating ${moduleName} session`, fmtBytes(modelDef.bytes), 100);
  return ort.InferenceSession.create(modelBytes, options);
}

async function loadSharedModels() {
  if (state.sharedSessions) return state.sharedSessions;
  if (state.sharedLoading) return state.sharedLoading;

  state.sharedLoading = (async () => {
    if (!("gpu" in navigator)) {
      throw new Error("WebGPU is not available in this browser. Use Chrome with WebGPU enabled.");
    }

    generateButton.disabled = true;
    loadButton.disabled = true;
    setStatusBadge("loading");
    setDot(dotRuntime, "loading");
    setDot(dotModel, "loading");
    setDot(dotGenerate, "pending");
    setReady("Loading shared models");
    cacheState.textContent = "loading shared models";
    setRunStatus(`Loading shared SDXL models from ${MODEL_REPO}. UNet downloads after you choose a precision and click Generate.`);
    await ensureOrt();
    await loadManifest();
    await loadSidecars();

    const shared = {};
    shared.text = await createSession("text_encoder");
    shared.text2 = await createSession("text_encoder_2");
    shared.vaeEncoder = await createSession("vae_encoder");
    shared.vae = await createSession("vae_decoder");
    state.sharedSessions = shared;
    state.cacheReady = true;
    cacheState.textContent = "shared models ready";
    setStatusBadge("success");
    setDot(dotRuntime, "done");
    setDot(dotModel, "done");
    setReady("Ready");
    hideProgress();
    log("Shared SDXL sessions are loaded. Pick BF16, FP8, or FP32; the selected UNet downloads on Generate.");
    generateButton.disabled = false;
    return shared;
  })();

  try {
    return await state.sharedLoading;
  } finally {
    state.sharedLoading = null;
    loadButton.disabled = false;
  }
}

async function loadAll() {
  const requestedUnet = state.manifest ? selectedUnetModule() : (unetVariantInput?.value || state.unetModule);
  if (state.sessions && state.unetModule === requestedUnet) return state.sessions;
  if (state.loading) return state.loading;

  state.loading = (async () => {
    if (!("gpu" in navigator)) {
      throw new Error("WebGPU is not available in this browser. Use Chrome with WebGPU enabled.");
    }

    generateButton.disabled = true;
    loadButton.disabled = true;
    setStatusBadge("loading");
    setDot(dotRuntime, "loading");
    setDot(dotModel, "loading");
    setDot(dotGenerate, "pending");
    setReady("Loading");
    setRunStatus(`Checking WebGPU, cache, tokenizer, scheduler, and ${MODEL_REPO} manifest...`);
    await ensureOrt();

    await loadManifest();
    await loadSidecars();
    const unetModule = selectedUnetModule();
    state.unetModule = unetModule;
    cacheState.textContent = state.sharedSessions ? "loading selected unet" : "loading shared models";

    const sharedSessions = await loadSharedModels();
    const sessions = { ...sharedSessions };
    log("Reusing shared SDXL sessions.");

    generateButton.disabled = true;
    loadButton.disabled = true;
    setStatusBadge("loading");
    setReady("Loading selected UNet");
    cacheState.textContent = "loading selected unet";
    log(`Loading selected UNet: ${unetModule}.`);
    sessions.unet = await createSession(unetModule);
    state.unetSession = sessions.unet;
    state.sessions = sessions;
    setStatusBadge("success");
    setDot(dotRuntime, "done");
    setDot(dotModel, "done");
    setReady("Ready");
    hideProgress();
    log(`Shared sessions and ${unetModule} loaded from ${MODEL_REPO}.`);
    generateButton.disabled = false;
    return sessions;
  })();

  try {
    return await state.loading;
  } finally {
    state.loading = null;
    loadButton.disabled = false;
  }
}

function normalizeDimension(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return 1024;
  return Math.max(512, Math.min(1024, Math.round(parsed / 64) * 64));
}

function normalizeSteps(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return DEFAULT_QUALITY_STEPS;
  return Math.max(MIN_QUALITY_STEPS, Math.min(MAX_QUALITY_STEPS, Math.round(parsed)));
}

function normalizeCfgScale(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return 7.5;
  return Math.max(1, Math.min(12, Math.round(parsed * 2) / 2));
}

function normalizeStrength(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return 0.65;
  return Math.max(0.05, Math.min(1, Math.round(parsed * 20) / 20));
}

function normalizeBatchCount(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return 1;
  return Math.max(1, Math.min(4, Math.round(parsed)));
}

function normalizeMaskBlur(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return 0;
  return Math.max(0, Math.min(32, Math.round(parsed)));
}

function currentMode() {
  return modeInputs.find(input => input.checked)?.value ?? "txt2img";
}

function createRandomSeed() {
  if (globalThis.crypto?.getRandomValues) {
    return globalThis.crypto.getRandomValues(new Uint32Array(1))[0];
  }
  return Math.floor(Math.random() * 0x100000000);
}

function resolveSeed(value) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed === RANDOM_SEED_SENTINEL) {
    return {
      inputValue: RANDOM_SEED_SENTINEL,
      seed: createRandomSeed(),
      random: true
    };
  }
  const seed = Math.trunc(parsed);
  return {
    inputValue: seed,
    seed,
    random: false
  };
}

function readSettings() {
  const width = normalizeDimension(widthInput.value);
  const height = normalizeDimension(heightInput.value);
  const steps = normalizeSteps(stepsInput.value);
  const cfgScale = normalizeCfgScale(guidanceInput.value);
  const strength = normalizeStrength(strengthInput.value);
  const batchCount = normalizeBatchCount(batchCountInput.value);
  const maskBlur = normalizeMaskBlur(maskBlurInput.value);
  const seedSetting = resolveSeed(seedInput.value);
  widthInput.value = String(width);
  heightInput.value = String(height);
  stepsInput.value = String(steps);
  guidanceInput.value = String(cfgScale);
  strengthInput.value = String(strength);
  batchCountInput.value = String(batchCount);
  maskBlurInput.value = String(maskBlur);
  seedInput.value = String(seedSetting.inputValue);
  const latentW = width / 8;
  const latentH = height / 8;
  const attentionTokens = latentW * latentH;
  setFrameAspect(width, height);
  return {
    mode: currentMode(),
    prompt: document.querySelector("#prompt").value,
    negative: document.querySelector("#negative").value,
    seed: seedSetting.seed,
    seedWasRandom: seedSetting.random,
    steps,
    width,
    height,
    strength,
    batchCount,
    sampler: samplerInput.value,
    sourceFit: sourceFitInput.value,
    maskBlur,
    invertMask: invertMaskInput.checked,
    cfgScale,
    latentW,
    latentH,
    attentionTokens
  };
}

function validateSettings(settings) {
  if (settings.attentionTokens > MAX_WEBGPU_ATTENTION_TOKENS) {
    const latentShape = `${settings.latentW}x${settings.latentH}`;
    throw new Error(
      `This SDXL ONNX UNet would allocate self-attention for ${settings.attentionTokens} latent tokens (${latentShape}). ` +
      "Use a smaller SDXL size or an optimized attention UNet for larger sizes."
    );
  }
  if ((settings.mode === "img2img" || settings.mode === "inpaint") && !sourceImageBitmap) {
    throw new Error("Image to Image and Inpaint Mask modes require an input image.");
  }
  if (settings.mode === "inpaint" && !hasPaintedMask()) {
    throw new Error("Inpaint Mask mode requires a painted mask on the input image.");
  }
  if (settings.sampler !== "ddim") {
    throw new Error(`Unsupported sampler: ${settings.sampler}`);
  }
}

function setFrameAspect(width, height) {
  const aspect = `${width} / ${height}`;
  sourceFrame?.style.setProperty("--image-aspect", aspect);
  outputFrame?.style.setProperty("--image-aspect", aspect);
}

function setCanvasSize(canvas, width, height) {
  if (!canvas || (canvas.width === width && canvas.height === height)) return;
  canvas.width = width;
  canvas.height = height;
}

function clearMask() {
  if (!maskCtx || !maskCanvas) return;
  maskCtx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
}

function containRect(imageW, imageH, targetW, targetH) {
  const scale = Math.min(targetW / imageW, targetH / imageH);
  const width = imageW * scale;
  const height = imageH * scale;
  return {
    x: (targetW - width) / 2,
    y: (targetH - height) / 2,
    width,
    height
  };
}

function coverSourceRect(imageW, imageH, targetW, targetH) {
  const scale = Math.max(targetW / imageW, targetH / imageH);
  const width = targetW / scale;
  const height = targetH / scale;
  return {
    x: (imageW - width) / 2,
    y: (imageH - height) / 2,
    width,
    height
  };
}

function drawEdgePadding(ctx, image, rect, targetW, targetH) {
  const { x, y, width, height } = rect;
  if (x > 0) {
    ctx.drawImage(image, 0, 0, 1, image.height, 0, y, x, height);
    ctx.drawImage(image, image.width - 1, 0, 1, image.height, x + width, y, targetW - x - width, height);
  }
  if (y > 0) {
    ctx.drawImage(image, 0, 0, image.width, 1, x, 0, width, y);
    ctx.drawImage(image, 0, image.height - 1, image.width, 1, x, y + height, width, targetH - y - height);
  }
  if (x > 0 && y > 0) {
    ctx.drawImage(image, 0, 0, 1, 1, 0, 0, x, y);
    ctx.drawImage(image, image.width - 1, 0, 1, 1, x + width, 0, targetW - x - width, y);
    ctx.drawImage(image, 0, image.height - 1, 1, 1, 0, y + height, x, targetH - y - height);
    ctx.drawImage(image, image.width - 1, image.height - 1, 1, 1, x + width, y + height, targetW - x - width, targetH - y - height);
  }
}

function drawSourceImage(width = normalizeDimension(widthInput.value), height = normalizeDimension(heightInput.value)) {
  if (!sourceCanvas || !maskCanvas || !sourceCtx) return;
  const oldMask = maskCanvas.width && maskCanvas.height ? document.createElement("canvas") : null;
  if (oldMask && maskCtx) {
    oldMask.width = maskCanvas.width;
    oldMask.height = maskCanvas.height;
    oldMask.getContext("2d").drawImage(maskCanvas, 0, 0);
  }

  setCanvasSize(sourceCanvas, width, height);
  setCanvasSize(maskCanvas, width, height);
  sourceCtx.clearRect(0, 0, width, height);

  if (sourceImageBitmap) {
    const fit = sourceFitInput.value;
    if (fit === "crop") {
      const crop = coverSourceRect(sourceImageBitmap.width, sourceImageBitmap.height, width, height);
      sourceCtx.drawImage(sourceImageBitmap, crop.x, crop.y, crop.width, crop.height, 0, 0, width, height);
    } else {
      const rect = containRect(sourceImageBitmap.width, sourceImageBitmap.height, width, height);
      if (fit === "edge") {
        drawEdgePadding(sourceCtx, sourceImageBitmap, rect, width, height);
      } else {
        sourceCtx.fillStyle = "#000";
        sourceCtx.fillRect(0, 0, width, height);
      }
      sourceCtx.drawImage(sourceImageBitmap, rect.x, rect.y, rect.width, rect.height);
    }
    sourcePlaceholder?.classList.add("hidden");
  } else {
    sourcePlaceholder?.classList.remove("hidden");
  }

  if (oldMask && maskCtx) {
    maskCtx.clearRect(0, 0, width, height);
    maskCtx.drawImage(oldMask, 0, 0, width, height);
  }
}

async function loadSourceImage(file) {
  sourceImageBitmap?.close?.();
  sourceImageBitmap = null;
  clearMask();
  if (!file) {
    drawSourceImage();
    return;
  }
  sourceImageBitmap = await createImageBitmap(file);
  const settings = readSettings();
  drawSourceImage(settings.width, settings.height);
}

function sourceImageTensor(settings) {
  drawSourceImage(settings.width, settings.height);
  const imageData = sourceCtx.getImageData(0, 0, settings.width, settings.height).data;
  const plane = settings.width * settings.height;
  const data = new Float32Array(3 * plane);
  for (let i = 0; i < plane; i++) {
    data[i] = imageData[i * 4] / 127.5 - 1;
    data[plane + i] = imageData[i * 4 + 1] / 127.5 - 1;
    data[2 * plane + i] = imageData[i * 4 + 2] / 127.5 - 1;
  }
  return new ort.Tensor("float32", data, [1, 3, settings.height, settings.width]);
}

async function encodeSourceLatents(settings) {
  const encoded = await state.sessions.vaeEncoder.run({ sample: sourceImageTensor(settings) });
  const tensor = encoded.latent_sample ?? encoded.latent_parameters ?? encoded.sample ?? Object.values(encoded)[0];
  const latentSize = 4 * settings.latentH * settings.latentW;
  if (tensor.data.length < latentSize) {
    throw new Error(`VAE encoder returned ${tensor.data.length} values, expected at least ${latentSize}.`);
  }
  const out = new Float32Array(latentSize);
  // ONNX VAE encoder exports mean/logvar as 8 channels; use the first 4 mean channels.
  for (let i = 0; i < latentSize; i++) out[i] = tensor.data[i] * SDXL_LATENT_SCALE;
  return out;
}

function hasPaintedMask() {
  if (!maskCtx || !maskCanvas) return false;
  const data = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height).data;
  for (let i = 3; i < data.length; i += 4) {
    if (data[i] > 0) return true;
  }
  return false;
}

function maskToLatentMask(settings) {
  const imageData = maskCtx.getImageData(0, 0, settings.width, settings.height).data;
  const mask = new Float32Array(settings.latentW * settings.latentH);
  for (let ly = 0; ly < settings.latentH; ly++) {
    for (let lx = 0; lx < settings.latentW; lx++) {
      let sum = 0;
      for (let y = 0; y < 8; y++) {
        for (let x = 0; x < 8; x++) {
          const px = Math.min(settings.width - 1, lx * 8 + x);
          const py = Math.min(settings.height - 1, ly * 8 + y);
          sum += imageData[(py * settings.width + px) * 4 + 3] / 255;
        }
      }
      mask[ly * settings.latentW + lx] = sum / 64;
    }
  }
  if (settings.invertMask) {
    for (let i = 0; i < mask.length; i++) mask[i] = 1 - mask[i];
  }
  const latentBlur = Math.round(settings.maskBlur / 8);
  return latentBlur > 0 ? blurLatentMask(mask, settings.latentW, settings.latentH, latentBlur) : mask;
}

function blurLatentMask(mask, width, height, radius) {
  const out = new Float32Array(mask.length);
  const diameter = radius * 2 + 1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let sum = 0;
      let count = 0;
      for (let oy = -radius; oy <= radius; oy++) {
        for (let ox = -radius; ox <= radius; ox++) {
          const sx = Math.max(0, Math.min(width - 1, x + ox));
          const sy = Math.max(0, Math.min(height - 1, y + oy));
          sum += mask[sy * width + sx];
          count++;
        }
      }
      out[y * width + x] = Math.min(1, Math.max(0, sum / Math.max(1, count || diameter * diameter)));
    }
  }
  return out;
}

function addNoise(sample, noise, timestep) {
  const alpha = state.scheduler.alphas_cumprod[timestep];
  const sqrtAlpha = Math.sqrt(alpha);
  const sqrtBeta = Math.sqrt(1 - alpha);
  const out = new Float32Array(sample.length);
  for (let i = 0; i < sample.length; i++) out[i] = sqrtAlpha * sample[i] + sqrtBeta * noise[i];
  return out;
}

function denoiseTimesteps(settings) {
  const timesteps = makeTimesteps(settings.steps);
  if (settings.mode === "txt2img") return timesteps;
  const initTimestep = Math.min(settings.steps, Math.max(1, Math.floor(settings.steps * settings.strength)));
  return timesteps.slice(settings.steps - initTimestep);
}

function blendMaskedLatents(latents, baseLatents, latentMask) {
  const plane = latentMask.length;
  for (let i = 0; i < latents.length; i++) {
    const mask = latentMask[i % plane];
    latents[i] = baseLatents[i] * (1 - mask) + latents[i] * mask;
  }
  return latents;
}

function mulberry32(seed) {
  return function next() {
    let value = seed += 0x6D2B79F5;
    value = Math.imul(value ^ value >>> 15, value | 1);
    value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    return ((value ^ value >>> 14) >>> 0) / 4294967296;
  };
}

function randomNormalArray(size, seed) {
  const rand = mulberry32(seed);
  const out = new Float32Array(size);
  for (let index = 0; index < size; index += 2) {
    const u1 = Math.max(rand(), 1e-7);
    const u2 = rand();
    const radius = Math.sqrt(-2 * Math.log(u1));
    out[index] = radius * Math.cos(2 * Math.PI * u2);
    if (index + 1 < size) out[index + 1] = radius * Math.sin(2 * Math.PI * u2);
  }
  return out;
}

function bytesToUnicode() {
  const bs = [];
  for (let i = 33; i <= 126; i++) bs.push(i);
  for (let i = 161; i <= 172; i++) bs.push(i);
  for (let i = 174; i <= 255; i++) bs.push(i);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n += 1;
    }
  }
  const map = new Map();
  bs.forEach((b, i) => map.set(b, String.fromCharCode(cs[i])));
  return map;
}

class ClipTokenizer {
  constructor(vocab, mergesText) {
    this.vocab = vocab;
    this.byteEncoder = bytesToUnicode();
    this.cache = new Map();
    this.start = vocab["<|startoftext|>"];
    this.end = vocab["<|endoftext|>"];
    this.ranks = new Map();
    mergesText.split(/\r?\n/).filter(line => line && !line.startsWith("#")).forEach((line, index) => {
      this.ranks.set(line.trim(), index);
    });
  }

  bpe(token) {
    if (this.cache.has(token)) return this.cache.get(token);
    let word = [...token];
    if (word.length === 0) return [];
    word[word.length - 1] = `${word[word.length - 1]}</w>`;
    while (word.length > 1) {
      let best = null;
      let bestRank = Infinity;
      for (let i = 0; i < word.length - 1; i++) {
        const pair = `${word[i]} ${word[i + 1]}`;
        const rank = this.ranks.has(pair) ? this.ranks.get(pair) : Infinity;
        if (rank < bestRank) {
          bestRank = rank;
          best = [word[i], word[i + 1]];
        }
      }
      if (!best) break;
      const next = [];
      for (let i = 0; i < word.length; i++) {
        if (i < word.length - 1 && word[i] === best[0] && word[i + 1] === best[1]) {
          next.push(best[0] + best[1]);
          i += 1;
        } else {
          next.push(word[i]);
        }
      }
      word = next;
    }
    const ids = word.map(piece => this.vocab[piece] ?? this.end);
    this.cache.set(token, ids);
    return ids;
  }

  encode(text) {
    const ids = [this.start];
    const regex = /<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+/giu;
    for (const match of text.toLowerCase().matchAll(regex)) {
      const bytes = new TextEncoder().encode(match[0]);
      let token = "";
      bytes.forEach(byte => { token += this.byteEncoder.get(byte); });
      ids.push(...this.bpe(token));
    }
    ids.push(this.end);
    while (ids.length < 77) ids.push(this.end);
    return ids.slice(0, 77);
  }
}

function makeTimesteps(steps) {
  const trainSteps = state.scheduler.num_train_timesteps ?? 1000;
  const stepsOffset = state.scheduler.steps_offset ?? 0;
  const stepRatio = Math.floor(trainSteps / steps);
  return Array.from({ length: steps }, (_, i) => (steps - 1 - i) * stepRatio + stepsOffset);
}

function ddimStep(noise, timestep, sample, steps) {
  const alpha = state.scheduler.alphas_cumprod[timestep];
  const prevT = timestep - Math.floor(1000 / steps);
  const prevAlpha = prevT >= 0 ? state.scheduler.alphas_cumprod[prevT] : state.scheduler.final_alpha_cumprod;
  const beta = 1 - alpha;
  const out = new Float32Array(sample.length);
  for (let i = 0; i < sample.length; i++) {
    const predOriginal = (sample[i] - Math.sqrt(beta) * noise[i]) / Math.sqrt(alpha);
    const predDir = Math.sqrt(1 - prevAlpha) * noise[i];
    out[i] = Math.sqrt(prevAlpha) * predOriginal + predDir;
  }
  return out;
}

function drawImage(decoded, width, height) {
  const canvas = document.querySelector("#canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  const imageData = ctx.createImageData(width, height);
  const plane = width * height;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const p = y * width + x;
      imageData.data[p * 4] = Math.round(Math.min(1, Math.max(0, decoded[p] / 2 + 0.5)) * 255);
      imageData.data[p * 4 + 1] = Math.round(Math.min(1, Math.max(0, decoded[plane + p] / 2 + 0.5)) * 255);
      imageData.data[p * 4 + 2] = Math.round(Math.min(1, Math.max(0, decoded[2 * plane + p] / 2 + 0.5)) * 255);
      imageData.data[p * 4 + 3] = 255;
    }
  }
  ctx.putImageData(imageData, 0, 0);
}

function concatLastDim(a, b, batch, sequence) {
  const aWidth = a.dims?.[2] ?? (a.data.length / (batch * sequence));
  const bWidth = b.dims?.[2] ?? (b.data.length / (batch * sequence));
  const outWidth = aWidth + bWidth;
  const out = new Float32Array(batch * sequence * outWidth);
  for (let row = 0; row < batch * sequence; row++) {
    out.set(a.data.subarray(row * aWidth, (row + 1) * aWidth), row * outWidth);
    out.set(b.data.subarray(row * bWidth, (row + 1) * bWidth), row * outWidth + aWidth);
  }
  return new ort.Tensor("float32", out, [batch, sequence, outWidth]);
}

async function encodeSdxlPrompt(settings) {
  const batch = 2;
  const sequence = 77;
  const ids1 = new BigInt64Array([...state.tokenizer.encode(settings.negative), ...state.tokenizer.encode(settings.prompt)].map(BigInt));
  const ids2 = new BigInt64Array([...state.tokenizer2.encode(settings.negative), ...state.tokenizer2.encode(settings.prompt)].map(BigInt));
  const text1 = await state.sessions.text.run({
    input_ids: new ort.Tensor("int64", ids1, [batch, sequence])
  });
  const text2 = await state.sessions.text2.run({
    input_ids: new ort.Tensor("int64", ids2, [batch, sequence])
  });
  const hidden1 = text1.hidden_states ?? text1.last_hidden_state ?? Object.values(text1)[0];
  const hidden2 = text2.hidden_states ?? text2.last_hidden_state ?? Object.values(text2)[0];
  const pooled = text2.pooled_output ?? Object.values(text2)[1];
  if (!hidden1 || !hidden2 || !pooled) throw new Error("SDXL text encoders did not return hidden states and pooled output.");
  return {
    embeddings: concatLastDim(hidden1, hidden2, batch, sequence),
    pooled
  };
}

function makeSdxlTimeIds(settings) {
  return new ort.Tensor("float32", new Float32Array([
    settings.height, settings.width, 0, 0, settings.height, settings.width,
    settings.height, settings.width, 0, 0, settings.height, settings.width
  ]), [2, 6]);
}

async function generate() {
  generateButton.disabled = true;
  try {
    const settings = readSettings();
    validateSettings(settings);
    setStatusBadge("loading");
    setDot(dotGenerate, "loading");
    const seedLabel = settings.seedWasRandom ? `${settings.seed} (random)` : String(settings.seed);
    const strengthLabel = settings.mode === "txt2img" ? "" : `, strength ${settings.strength}`;
    const sourceFitLabel = settings.mode === "txt2img" ? "" : `, source fit ${settings.sourceFit}`;
    setRunStatus(`Starting ${settings.mode}, ${settings.sampler.toUpperCase()}, ${settings.width}x${settings.height}, ${settings.steps} step(s), CFG ${settings.cfgScale}${strengthLabel}${sourceFitLabel}, batch ${settings.batchCount}, seed ${seedLabel}...`);
    cacheState.textContent = "loading model";
    await loadAll();

    const promptConditioning = await encodeSdxlPrompt(settings);
    const timeIds = makeSdxlTimeIds(settings);
    let initLatents = null;
    let latentMask = null;

    if (settings.mode !== "txt2img") {
      log("Encoding input image to VAE latent space...");
      initLatents = await encodeSourceLatents(settings);
      if (settings.mode === "inpaint") {
        latentMask = maskToLatentMask(settings);
        log("Using latent mask blending: painted mask is regenerated, unpainted area is preserved.");
      }
    }

    for (let batchIndex = 0; batchIndex < settings.batchCount; batchIndex++) {
      const batchSeed = settings.seedWasRandom ? createRandomSeed() : settings.seed + batchIndex;
      const timesteps = denoiseTimesteps(settings);
      let latents;
      let initNoise = null;
      const batchLabel = settings.batchCount > 1 ? `Batch ${batchIndex + 1}/${settings.batchCount} ` : "";
      log(`${batchLabel}seed ${batchSeed}`);

      if (settings.mode === "txt2img") {
        latents = randomNormalArray(4 * settings.latentH * settings.latentW, batchSeed);
        const initSigma = state.scheduler.init_noise_sigma ?? 1;
        for (let i = 0; i < latents.length; i++) latents[i] *= initSigma;
      } else {
        initNoise = randomNormalArray(initLatents.length, batchSeed);
        latents = addNoise(initLatents, initNoise, timesteps[0]);
      }

      for (let index = 0; index < timesteps.length; index++) {
        const t = timesteps[index];
        log(`${batchLabel}Step ${index + 1}/${timesteps.length} timestep ${t}`);
        const doubled = new Float32Array(latents.length * 2);
        doubled.set(latents, 0);
        doubled.set(latents, latents.length);
        const out = await state.sessions.unet.run({
          sample: new ort.Tensor("float32", doubled, [2, 4, settings.latentH, settings.latentW]),
          timestep: new ort.Tensor("int64", BigInt64Array.from([BigInt(t)]), []),
          encoder_hidden_states: promptConditioning.embeddings,
          text_embeds: promptConditioning.pooled,
          time_ids: timeIds
        });
        const noise = (out.noise_pred ?? Object.values(out)[0]).data;
        const guided = new Float32Array(latents.length);
        for (let i = 0; i < latents.length; i++) {
          guided[i] = noise[i] + settings.cfgScale * (noise[i + latents.length] - noise[i]);
        }
        latents = ddimStep(guided, t, latents, settings.steps);
        if (latentMask) {
          const baseLatents = index < timesteps.length - 1
            ? addNoise(initLatents, initNoise, timesteps[index + 1])
            : initLatents;
          latents = blendMaskedLatents(latents, baseLatents, latentMask);
        }
      }

      const scaled = new Float32Array(latents.length);
      for (let i = 0; i < latents.length; i++) scaled[i] = latents[i] / SDXL_LATENT_SCALE;
      const decoded = await state.sessions.vae.run({
        latent_sample: new ort.Tensor("float32", scaled, [1, 4, settings.latentH, settings.latentW])
      });
      drawImage((decoded.sample ?? Object.values(decoded)[0]).data, settings.width, settings.height);
    }

    setStatusBadge("success");
    setDot(dotGenerate, "done");
    setReady("Ready");
    hideProgress();
    log("Image generated in browser.");
  } catch (error) {
    setStatusBadge("error");
    if (!state.sessions) setDot(dotModel, "error");
    setDot(dotGenerate, "error");
    setReady("Error");
    log(`ERROR: ${error.message}`);
  } finally {
    generateButton.disabled = false;
  }
}

async function clearModelCache() {
  await caches.delete(CACHE_NAME);
  if (state.manifest) {
    for (const def of allKnownModelAssetDefs()) await removeOpfsFile(def);
  }
  preloadedBuffers.clear();
  downloadPromises.clear();
  cacheState.textContent = "cache cleared";
  state.loading = null;
  state.sessions = null;
  state.sharedSessions = null;
  state.unetSession = null;
  state.manifest = null;
  state.cacheWarming = null;
  state.cacheReady = false;
  setStatusBadge("info");
  setDot(dotModel, "pending");
  setDot(dotGenerate, "pending");
  generateButton.disabled = false;
  setReady("Cache cleared");
  setRunStatus("Model cache cleared. Reload the page to download the browser assets again.");
}

function maskPoint(event) {
  const rect = maskCanvas.getBoundingClientRect();
  return {
    x: (event.clientX - rect.left) * (maskCanvas.width / rect.width),
    y: (event.clientY - rect.top) * (maskCanvas.height / rect.height)
  };
}

function drawMaskStroke(from, to, erase) {
  if (!maskCtx) return;
  const radius = Number(brushSizeInput.value) || 32;
  maskCtx.save();
  maskCtx.globalCompositeOperation = erase ? "destination-out" : "source-over";
  maskCtx.strokeStyle = "rgba(108,99,255,.68)";
  maskCtx.fillStyle = "rgba(108,99,255,.68)";
  maskCtx.lineWidth = radius;
  maskCtx.lineCap = "round";
  maskCtx.lineJoin = "round";
  maskCtx.beginPath();
  maskCtx.moveTo(from.x, from.y);
  maskCtx.lineTo(to.x, to.y);
  maskCtx.stroke();
  maskCtx.beginPath();
  maskCtx.arc(to.x, to.y, radius / 2, 0, Math.PI * 2);
  maskCtx.fill();
  maskCtx.restore();
}

function startMaskDraw(event) {
  if (!maskCanvas || !sourceImageBitmap) return;
  event.preventDefault();
  isMaskDrawing = true;
  maskEraseMode = event.button === 2 || (event.buttons & 2) === 2;
  lastMaskPoint = maskPoint(event);
  drawMaskStroke(lastMaskPoint, lastMaskPoint, maskEraseMode);
  maskCanvas.setPointerCapture?.(event.pointerId);
}

function moveMaskDraw(event) {
  if (!isMaskDrawing || !lastMaskPoint) return;
  event.preventDefault();
  const next = maskPoint(event);
  drawMaskStroke(lastMaskPoint, next, maskEraseMode || (event.buttons & 2) === 2);
  lastMaskPoint = next;
}

function stopMaskDraw(event) {
  if (!isMaskDrawing) return;
  event?.preventDefault?.();
  isMaskDrawing = false;
  lastMaskPoint = null;
  if (event?.pointerId !== undefined) maskCanvas.releasePointerCapture?.(event.pointerId);
}

function wireEvents() {
  generateButton.addEventListener("click", generate);
  loadButton.addEventListener("click", () => loadSharedModels().catch(error => {
    setStatusBadge("error");
    setDot(dotModel, "error");
    setReady("Error");
    log(`ERROR: ${error.message}`);
  }));
  clearCacheButton.addEventListener("click", () => clearModelCache().catch(error => log(`ERROR: ${error.message}`)));
  guidanceInput.addEventListener("change", readSettings);
  seedInput.addEventListener("change", readSettings);
  samplerInput.addEventListener("change", readSettings);
  unetVariantInput?.addEventListener("change", () => {
    state.unetModule = unetVariantInput.value;
    state.unetSession = null;
    state.sessions = null;
    state.loading = null;
    cacheState.textContent = state.sharedSessions ? "shared models ready" : "download on generate";
    setStatusBadge("info");
    setDot(dotGenerate, "pending");
    setReady("Ready");
    log(`UNet precision selected: ${state.unetModule}. This UNet will download when Generate is clicked.`);
  });
  strengthInput.addEventListener("change", readSettings);
  batchCountInput.addEventListener("change", readSettings);
  maskBlurInput.addEventListener("change", readSettings);
  invertMaskInput.addEventListener("change", readSettings);
  modeInputs.forEach(input => input.addEventListener("change", readSettings));
  sourceFitInput.addEventListener("change", () => {
    const settings = readSettings();
    drawSourceImage(settings.width, settings.height);
  });
  widthInput.addEventListener("change", () => {
    const settings = readSettings();
    drawSourceImage(settings.width, settings.height);
  });
  heightInput.addEventListener("change", () => {
    const settings = readSettings();
    drawSourceImage(settings.width, settings.height);
  });
  stepsInput.addEventListener("change", readSettings);
  sourceImageInput.addEventListener("change", () => {
    loadSourceImage(sourceImageInput.files?.[0]).catch(error => log(`ERROR: ${error.message}`));
  });
  clearMaskButton.addEventListener("click", clearMask);
  maskCanvas.addEventListener("contextmenu", event => event.preventDefault());
  maskCanvas.addEventListener("pointerdown", startMaskDraw);
  maskCanvas.addEventListener("pointermove", moveMaskDraw);
  maskCanvas.addEventListener("pointerup", stopMaskDraw);
  maskCanvas.addEventListener("pointercancel", stopMaskDraw);
  maskCanvas.addEventListener("pointerleave", stopMaskDraw);
}

wireEvents();
drawSourceImage();
readSettings();
setStatusBadge("info");
setDot(dotRuntime, "pending");
setDot(dotModel, "pending");
setDot(dotGenerate, "pending");
setReady("Booting");
cacheState.textContent = "download pending";
loadButton.style.display = "inline-flex";
const scheduleCacheWarm = window.requestIdleCallback || (callback => window.setTimeout(callback, 1200));
async function probeManifestOnly() {
  await loadManifest();
  await loadSidecars();
  const defs = allModelAssetDefs();
  const externals = defs.filter(def => def.role === "external").map(def => `${def.path} (${fmtBytes(def.bytes)})`);
  log(`Manifest probe loaded ${defs.length} asset definition(s).`);
  log(`External data: ${externals.length ? externals.join(", ") : "none"}`);
}

if (SKIP_MODEL_WARM || !AUTO_MODEL_WARM) {
  generateButton.disabled = false;
  setReady(SKIP_MODEL_WARM ? "QA mode" : "Ready");
  cacheState.textContent = SKIP_MODEL_WARM ? "model warm skipped" : "download on generate";
  log(SKIP_MODEL_WARM
    ? "Model cache warm skipped by URL parameter."
    : "Model download starts when you click Generate Image or Load models.");
  if (PROBE_MANIFEST) {
    log("Manifest probe requested.");
    try {
      await probeManifestOnly();
    } catch (error) {
      setStatusBadge("error");
      setReady("Error");
      log(`ERROR: ${error.message}`);
    }
  }
} else {
  scheduleCacheWarm(() => {
    warmModelCache().catch(error => {
      setStatusBadge("error");
      setDot(dotModel, "error");
      setReady("Error");
      generateButton.disabled = false;
      log(`ERROR: ${error.message}`);
    });
  });
}
