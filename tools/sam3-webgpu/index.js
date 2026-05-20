const MODEL_ID = 'onnx-community/sam3-tracker-ONNX';
const IMPORT_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@next';
const params = new URLSearchParams(window.location.search);
const mockMode = params.has('mock');

let Transformers = null;
let model = null;
let processor = null;
let imageInput = null;
let imageProcessed = null;
let imageEmbeddings = null;
let currentImageUrl = '';
let promptMode = 'point';
let points = [];
let activeBox = null;
let isDrawingBox = false;
let boxStart = null;
let isModelReady = false;
let isEncoding = false;
let isDecoding = false;
let decodePending = false;
let bestMask = null;
let lastObjectUrl = null;
const fileMap = new Map();

const refs = {
  statusBadge: document.getElementById('statusBadge'),
  statusMsg: document.getElementById('statusMsg'),
  backendBadge: document.getElementById('backendBadge'),
  loadModelBtn: document.getElementById('loadModelBtn'),
  clearCacheBtn: document.getElementById('clearCacheBtn'),
  progressWrap: document.getElementById('progressWrap'),
  progressFill: document.getElementById('progressFill'),
  progressLabel: document.getElementById('progressLabel'),
  progressMeta: document.getElementById('progressMeta'),
  fileList: document.getElementById('fileList'),
  dotRuntime: document.getElementById('dot-runtime'),
  dotModel: document.getElementById('dot-model'),
  dotImage: document.getElementById('dot-image'),
  dotMask: document.getElementById('dot-mask'),
  stage: document.getElementById('stage'),
  uploadArea: document.getElementById('uploadArea'),
  imageInput: document.getElementById('imageInput'),
  imageDisplay: document.getElementById('imageDisplay'),
  maskCanvas: document.getElementById('maskCanvas'),
  guideCanvas: document.getElementById('guideCanvas'),
  stageHint: document.getElementById('stageHint'),
  pointLabelSel: document.getElementById('pointLabelSel'),
  overlayAlphaInput: document.getElementById('overlayAlphaInput'),
  maskColorInput: document.getElementById('maskColorInput'),
  scoreThresholdInput: document.getElementById('scoreThresholdInput'),
  pointCount: document.getElementById('pointCount'),
  maskScore: document.getElementById('maskScore'),
  imageSize: document.getElementById('imageSize'),
  decodeBtn: document.getElementById('decodeBtn'),
  clearPromptsBtn: document.getElementById('clearPromptsBtn'),
  resetImageBtn: document.getElementById('resetImageBtn'),
  downloadCutoutBtn: document.getElementById('downloadCutoutBtn'),
  downloadMaskBtn: document.getElementById('downloadMaskBtn'),
  downloadOverlayBtn: document.getElementById('downloadOverlayBtn'),
  logCard: document.getElementById('logCard'),
  runLog: document.getElementById('runLog'),
};

const maskCtx = refs.maskCanvas.getContext('2d', { willReadFrequently: true });
const guideCtx = refs.guideCanvas.getContext('2d');

function setStatus(type, message) {
  refs.statusBadge.className = `status-badge ${type}`;
  refs.statusBadge.textContent = type.toUpperCase();
  refs.statusMsg.textContent = message;
}

function setDot(dot, state) {
  dot.className = `dot ${state}`;
}

function log(message) {
  refs.logCard.classList.remove('hidden');
  const time = new Date().toLocaleTimeString();
  refs.runLog.textContent += `[${time}] ${message}\n`;
  refs.runLog.scrollTop = refs.runLog.scrollHeight;
}

function fmtBytes(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return '';
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}

function sanitizeKey(value) {
  return `file-${value.replace(/[^a-z0-9]/gi, '-')}`;
}

function addFileDot(key, label) {
  const id = sanitizeKey(key);
  if (document.getElementById(id)) return;
  const item = document.createElement('div');
  item.className = 'model-item';
  item.innerHTML = `<span class="dot loading" id="${id}"></span>${label}`;
  refs.fileList.appendChild(item);
}

function onProgress(info) {
  const file = info.file || info.name || '';
  if (!file) return;
  const shortName = file.split('/').pop() || file;
  addFileDot(file, shortName);
  refs.progressWrap.classList.remove('hidden');

  if (info.status === 'progress') {
    fileMap.set(file, { loaded: info.loaded || 0, total: info.total || 0 });
    const totals = [...fileMap.values()].reduce((acc, item) => {
      acc.loaded += item.loaded || 0;
      acc.total += item.total || 0;
      return acc;
    }, { loaded: 0, total: 0 });
    if (totals.total > 0) {
      refs.progressFill.style.width = `${Math.min(100, (totals.loaded / totals.total) * 100)}%`;
      refs.progressMeta.textContent = `${fmtBytes(totals.loaded)} / ${fmtBytes(totals.total)}`;
    }
    refs.progressLabel.textContent = `Downloading ${shortName}`;
    setStatus('loading', `下載 ${shortName}...`);
  }

  if (info.status === 'done') {
    const dot = document.getElementById(sanitizeKey(file));
    if (dot) dot.className = 'dot done';
  }
}

function hasImage() {
  return Boolean(currentImageUrl && refs.imageDisplay.complete && refs.imageDisplay.naturalWidth);
}

function canDecode() {
  if (!isModelReady || !hasImage() || isEncoding || isDecoding) return false;
  if (promptMode === 'box') return Boolean(activeBox);
  return points.length > 0;
}

function updateButtons() {
  refs.loadModelBtn.disabled = isModelReady || isEncoding || isDecoding;
  refs.imageInput.disabled = !isModelReady || isEncoding;
  refs.decodeBtn.disabled = !canDecode();
  refs.clearPromptsBtn.disabled = !hasImage() || (!points.length && !activeBox && !bestMask);
  refs.resetImageBtn.disabled = !hasImage() || isEncoding;
  refs.downloadCutoutBtn.disabled = !bestMask;
  refs.downloadMaskBtn.disabled = !bestMask;
  refs.downloadOverlayBtn.disabled = !bestMask;
  refs.pointCount.textContent = String(points.length + (activeBox ? 1 : 0));
}

function clamp(value, min = 0, max = 1) {
  return Math.min(max, Math.max(min, value));
}

function canvasRect() {
  return refs.maskCanvas.getBoundingClientRect();
}

function getNormalizedPoint(event) {
  const rect = canvasRect();
  return {
    x: clamp((event.clientX - rect.left) / rect.width),
    y: clamp((event.clientY - rect.top) / rect.height),
  };
}

function updateCanvasGeometry() {
  if (!hasImage()) return;
  const { naturalWidth, naturalHeight } = refs.imageDisplay;
  const containerRect = refs.stage.getBoundingClientRect();
  const imageRatio = naturalWidth / naturalHeight;
  const containerRatio = containerRect.width / containerRect.height;
  let width;
  let height;
  let left;
  let top;

  if (imageRatio > containerRatio) {
    width = containerRect.width;
    height = width / imageRatio;
    left = 0;
    top = (containerRect.height - height) / 2;
  } else {
    height = containerRect.height;
    width = height * imageRatio;
    left = (containerRect.width - width) / 2;
    top = 0;
  }

  for (const canvas of [refs.maskCanvas, refs.guideCanvas]) {
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    canvas.style.left = `${left}px`;
    canvas.style.top = `${top}px`;
    canvas.width = naturalWidth;
    canvas.height = naturalHeight;
  }
  redrawMask();
  drawGuides();
}

function markerPosition(point) {
  const stageRect = refs.stage.getBoundingClientRect();
  const rect = canvasRect();
  return {
    left: rect.left - stageRect.left + point.x * rect.width,
    top: rect.top - stageRect.top + point.y * rect.height,
  };
}

function renderMarkers() {
  refs.stage.querySelectorAll('.marker').forEach(marker => marker.remove());
  for (const point of points) {
    const marker = document.createElement('span');
    marker.className = `marker ${point.label === 1 ? 'positive' : 'negative'}`;
    const pos = markerPosition(point);
    marker.style.left = `${pos.left}px`;
    marker.style.top = `${pos.top}px`;
    refs.stage.appendChild(marker);
  }
}

function drawGuides() {
  guideCtx.clearRect(0, 0, refs.guideCanvas.width, refs.guideCanvas.height);
  if (!activeBox) return;
  const x = activeBox.x1 * refs.guideCanvas.width;
  const y = activeBox.y1 * refs.guideCanvas.height;
  const w = (activeBox.x2 - activeBox.x1) * refs.guideCanvas.width;
  const h = (activeBox.y2 - activeBox.y1) * refs.guideCanvas.height;
  guideCtx.save();
  guideCtx.strokeStyle = '#f2b84b';
  guideCtx.lineWidth = Math.max(3, refs.guideCanvas.width / 360);
  guideCtx.setLineDash([10, 6]);
  guideCtx.strokeRect(x, y, w, h);
  guideCtx.fillStyle = 'rgba(242,184,75,.12)';
  guideCtx.fillRect(x, y, w, h);
  guideCtx.restore();
}

function hexToRgb(hex) {
  const value = hex.replace('#', '');
  return {
    r: parseInt(value.slice(0, 2), 16),
    g: parseInt(value.slice(2, 4), 16),
    b: parseInt(value.slice(4, 6), 16),
  };
}

function redrawMask() {
  maskCtx.clearRect(0, 0, refs.maskCanvas.width, refs.maskCanvas.height);
  if (!bestMask || bestMask.width !== refs.maskCanvas.width || bestMask.height !== refs.maskCanvas.height) return;
  const alpha = Math.round(Number(refs.overlayAlphaInput.value || 62) * 2.55);
  const color = hexToRgb(refs.maskColorInput.value || '#0072bd');
  const imageData = maskCtx.createImageData(bestMask.width, bestMask.height);
  for (let i = 0; i < bestMask.data.length; i++) {
    if (!bestMask.data[i]) continue;
    const offset = i * 4;
    imageData.data[offset] = color.r;
    imageData.data[offset + 1] = color.g;
    imageData.data[offset + 2] = color.b;
    imageData.data[offset + 3] = alpha;
  }
  maskCtx.putImageData(imageData, 0, 0);
}

function setBestMask(data, width, height, score) {
  bestMask = { data, width, height };
  refs.maskScore.textContent = Number.isFinite(score) ? score.toFixed(3) : '--';
  setDot(refs.dotMask, 'done');
  redrawMask();
  updateButtons();
}

function clearMaskOnly() {
  bestMask = null;
  refs.maskScore.textContent = '--';
  maskCtx.clearRect(0, 0, refs.maskCanvas.width, refs.maskCanvas.height);
  setDot(refs.dotMask, 'pending');
}

function clearPrompts() {
  points = [];
  activeBox = null;
  isDrawingBox = false;
  boxStart = null;
  clearMaskOnly();
  renderMarkers();
  drawGuides();
  refs.stageHint.textContent = '提示已清除。你可以重新點選、右鍵加入負向點，或切換框選模式。';
  updateButtons();
}

function resetImage() {
  if (lastObjectUrl) URL.revokeObjectURL(lastObjectUrl);
  lastObjectUrl = null;
  currentImageUrl = '';
  imageInput = null;
  imageProcessed = null;
  imageEmbeddings = null;
  refs.imageDisplay.removeAttribute('src');
  refs.imageDisplay.style.display = 'none';
  refs.uploadArea.classList.remove('hidden');
  refs.imageSize.textContent = '--';
  setDot(refs.dotImage, 'pending');
  clearPrompts();
  updateButtons();
}

async function importTransformers() {
  if (Transformers) return Transformers;
  setDot(refs.dotRuntime, 'loading');
  setStatus('loading', '載入 Transformers.js runtime...');
  Transformers = await import(IMPORT_URL);
  Transformers.env.allowLocalModels = false;
  Transformers.env.useBrowserCache = true;
  setDot(refs.dotRuntime, 'done');
  return Transformers;
}

async function loadModel() {
  if (isModelReady) return;
  refs.loadModelBtn.disabled = true;
  refs.progressWrap.classList.remove('hidden');
  refs.progressFill.style.width = '0%';

  try {
    if (mockMode) {
      setStatus('loading', '啟用 mock 推理模式，用於 UI 自動測試...');
      await new Promise(resolve => setTimeout(resolve, 250));
      isModelReady = true;
      refs.backendBadge.textContent = 'Mock WebGPU';
      setDot(refs.dotRuntime, 'done');
      setDot(refs.dotModel, 'done');
      refs.progressFill.style.width = '100%';
      refs.progressWrap.classList.add('hidden');
      setStatus('success', 'Mock 模式已就緒，可測試所有互動流程。');
      updateButtons();
      return;
    }

    if (!navigator.gpu) {
      throw new Error('這個瀏覽器沒有 navigator.gpu。請使用支援 WebGPU 的 Chrome/Edge，並確認硬體加速已啟用。');
    }

    const { Sam3TrackerModel, AutoProcessor } = await importTransformers();
    setDot(refs.dotModel, 'loading');
    setStatus('loading', `從 Hugging Face 下載 / 載入 ${MODEL_ID}...`);
    log(`Loading ${MODEL_ID} with WebGPU.`);

    model = await Sam3TrackerModel.from_pretrained(MODEL_ID, {
      dtype: {
        vision_encoder: 'q4',
        prompt_encoder_mask_decoder: 'fp32',
      },
      device: 'webgpu',
      progress_callback: onProgress,
    });
    processor = await AutoProcessor.from_pretrained(MODEL_ID, {
      progress_callback: onProgress,
    });

    isModelReady = true;
    refs.backendBadge.textContent = 'WebGPU · q4/fp32';
    setDot(refs.dotModel, 'done');
    refs.progressFill.style.width = '100%';
    refs.progressWrap.classList.add('hidden');
    setStatus('success', 'SAM3 Tracker WebGPU 模型已載入。請上傳圖片或選範例。');
    log('Model loaded.');
  } catch (error) {
    console.error(error);
    setDot(refs.dotModel, 'error');
    setStatus('error', `模型載入失敗：${error.message}`);
    log(`Model load failed: ${error.message}`);
  } finally {
    updateButtons();
  }
}

function loadImageElement(url) {
  return new Promise((resolve, reject) => {
    refs.imageDisplay.onload = () => resolve();
    refs.imageDisplay.onerror = () => reject(new Error('圖片載入失敗'));
    if (/^https?:/i.test(url)) refs.imageDisplay.crossOrigin = 'anonymous';
    else refs.imageDisplay.removeAttribute('crossorigin');
    refs.imageDisplay.src = url;
  });
}

async function encodeImage(url) {
  if (!isModelReady || isEncoding) return;
  isEncoding = true;
  updateButtons();
  clearPrompts();
  setDot(refs.dotImage, 'loading');
  setStatus('loading', '建立影像 embedding...');

  try {
    currentImageUrl = url;
    refs.uploadArea.classList.add('hidden');
    refs.imageDisplay.style.display = 'block';
    await loadImageElement(url);
    updateCanvasGeometry();
    refs.imageSize.textContent = `${refs.imageDisplay.naturalWidth}×${refs.imageDisplay.naturalHeight}`;

    if (mockMode) {
      imageInput = { width: refs.imageDisplay.naturalWidth, height: refs.imageDisplay.naturalHeight };
      imageProcessed = { original_sizes: [[imageInput.height, imageInput.width]], reshaped_input_sizes: [[imageInput.height, imageInput.width]] };
      imageEmbeddings = { mock: true };
      await new Promise(resolve => setTimeout(resolve, 180));
    } else {
      const { RawImage } = await importTransformers();
      imageInput = await RawImage.fromURL(url);
      imageProcessed = await processor(imageInput);
      imageEmbeddings = await model.get_image_embeddings(imageProcessed);
    }

    setDot(refs.dotImage, 'done');
    setStatus('success', 'Image embedding 完成。可以開始互動分割。');
    refs.stageHint.textContent = '點選加入正向點；右鍵加入負向點。框選模式可拖曳建立 box prompt。';
    log(`Image encoded: ${refs.imageDisplay.naturalWidth}x${refs.imageDisplay.naturalHeight}.`);
  } catch (error) {
    console.error(error);
    setDot(refs.dotImage, 'error');
    setStatus('error', `圖片處理失敗：${error.message}`);
    refs.uploadArea.classList.remove('hidden');
    refs.imageDisplay.style.display = 'none';
    log(`Image encode failed: ${error.message}`);
  } finally {
    isEncoding = false;
    updateButtons();
  }
}

function maskFromMockPrompt() {
  const width = refs.maskCanvas.width || 1;
  const height = refs.maskCanvas.height || 1;
  const data = new Uint8Array(width * height);

  if (activeBox) {
    const x1 = Math.floor(activeBox.x1 * width);
    const y1 = Math.floor(activeBox.y1 * height);
    const x2 = Math.ceil(activeBox.x2 * width);
    const y2 = Math.ceil(activeBox.y2 * height);
    for (let y = y1; y < y2; y++) {
      for (let x = x1; x < x2; x++) data[y * width + x] = 1;
    }
    return { data, width, height, score: 0.912 };
  }

  const positives = points.filter(point => point.label === 1);
  const negatives = points.filter(point => point.label === 0);
  const seed = positives[positives.length - 1] || points[points.length - 1];
  if (!seed) return { data, width, height, score: 0 };
  const cx = seed.x * width;
  const cy = seed.y * height;
  const rx = Math.max(width * 0.12, 28);
  const ry = Math.max(height * 0.16, 28);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const inside = ((x - cx) ** 2) / (rx ** 2) + ((y - cy) ** 2) / (ry ** 2) <= 1;
      if (inside) data[y * width + x] = 1;
    }
  }

  for (const neg of negatives) {
    const nx = neg.x * width;
    const ny = neg.y * height;
    const nr = Math.max(Math.min(width, height) * 0.08, 18);
    for (let y = Math.max(0, Math.floor(ny - nr)); y < Math.min(height, Math.ceil(ny + nr)); y++) {
      for (let x = Math.max(0, Math.floor(nx - nr)); x < Math.min(width, Math.ceil(nx + nr)); x++) {
        if ((x - nx) ** 2 + (y - ny) ** 2 <= nr ** 2) data[y * width + x] = 0;
      }
    }
  }
  return { data, width, height, score: 0.884 };
}

async function decodePointPrompt() {
  const { Tensor, RawImage } = await importTransformers();
  const reshaped = imageProcessed.reshaped_input_sizes[0];
  const coords = points.map(point => [point.x * reshaped[1], point.y * reshaped[0]]).flat();
  const labels = points.map(point => BigInt(point.label));
  const input_points = new Tensor('float32', coords, [1, 1, points.length, 2]);
  const input_labels = new Tensor('int64', labels, [1, 1, points.length]);
  const { pred_masks, iou_scores } = await model({ ...imageEmbeddings, input_points, input_labels });
  const masks = await processor.post_process_masks(
    pred_masks,
    imageProcessed.original_sizes,
    imageProcessed.reshaped_input_sizes,
  );
  return maskTensorToBestMask(RawImage.fromTensor(masks[0][0]), iou_scores.data);
}

async function decodeBoxPrompt() {
  const { RawImage } = await importTransformers();
  const width = refs.imageDisplay.naturalWidth;
  const height = refs.imageDisplay.naturalHeight;
  const input_boxes = [[[
    activeBox.x1 * width,
    activeBox.y1 * height,
    activeBox.x2 * width,
    activeBox.y2 * height,
  ]]];
  const inputs = await processor(imageInput, { input_boxes });
  const outputs = await model(inputs);
  const masks = await processor.post_process_masks(outputs.pred_masks, inputs.original_sizes, inputs.reshaped_input_sizes);
  return maskTensorToBestMask(RawImage.fromTensor(masks[0][0]), outputs.iou_scores.data);
}

function maskTensorToBestMask(maskImage, scores) {
  const numMasks = scores.length || 1;
  let bestIndex = 0;
  for (let i = 1; i < numMasks; i++) {
    if (scores[i] > scores[bestIndex]) bestIndex = i;
  }
  const width = maskImage.width;
  const height = maskImage.height;
  const data = new Uint8Array(width * height);
  for (let i = 0; i < data.length; i++) {
    data[i] = maskImage.data[numMasks * i + bestIndex] === 1 ? 1 : 0;
  }
  return { data, width, height, score: scores[bestIndex] };
}

async function decode() {
  if (!canDecode()) {
    if (isDecoding) decodePending = true;
    return;
  }
  isDecoding = true;
  refs.decodeBtn.disabled = true;
  setDot(refs.dotMask, 'loading');
  setStatus('loading', promptMode === 'box' ? '用 box prompt 產生遮罩...' : '用 point prompt 產生遮罩...');

  try {
    const result = mockMode ? maskFromMockPrompt() : (promptMode === 'box' ? await decodeBoxPrompt() : await decodePointPrompt());
    setBestMask(result.data, result.width, result.height, result.score);
    setStatus('success', `遮罩完成，best IoU ${result.score.toFixed(3)}。`);
    log(`Mask decoded (${promptMode}), score=${result.score.toFixed(3)}.`);
  } catch (error) {
    console.error(error);
    setDot(refs.dotMask, 'error');
    setStatus('error', `遮罩產生失敗：${error.message}`);
    log(`Decode failed: ${error.message}`);
  } finally {
    isDecoding = false;
    updateButtons();
  }

  if (decodePending) {
    decodePending = false;
    window.setTimeout(decode, 0);
  }
}

function setPromptMode(mode) {
  promptMode = mode;
  if (mode === 'box') {
    points = [];
    clearMaskOnly();
    renderMarkers();
  }
  document.querySelectorAll('.segment').forEach(button => {
    button.classList.toggle('active', button.dataset.mode === mode);
  });
  refs.stageHint.textContent = mode === 'box'
    ? '框選模式：在圖片上拖曳一個目標框，放開後可產生遮罩。'
    : mode === 'hover'
      ? '預覽模式：移動滑鼠會用單一正向點快速預覽，點擊可固定提示。'
      : '點選模式：左鍵正向點；右鍵負向點，或使用下拉選單指定標籤。';
  updateButtons();
}

function addPoint(event, forcePositive = false) {
  if (!imageEmbeddings || promptMode === 'box') return;
  const normalized = getNormalizedPoint(event);
  const label = forcePositive ? 1 : (event.button === 2 ? 0 : Number(refs.pointLabelSel.value));
  points.push({ ...normalized, label });
  activeBox = null;
  renderMarkers();
  drawGuides();
  updateButtons();
  decode();
}

function updateHoverPreview(event) {
  if (!imageEmbeddings || promptMode !== 'hover' || isDecoding) return;
  const normalized = getNormalizedPoint(event);
  points = [{ ...normalized, label: 1 }];
  activeBox = null;
  renderMarkers();
  updateButtons();
  decode();
}

function startBox(event) {
  if (!imageEmbeddings || promptMode !== 'box' || event.button !== 0) return;
  isDrawingBox = true;
  boxStart = getNormalizedPoint(event);
  activeBox = { x1: boxStart.x, y1: boxStart.y, x2: boxStart.x, y2: boxStart.y };
  clearMaskOnly();
  drawGuides();
}

function updateBox(event) {
  if (!isDrawingBox || promptMode !== 'box') return;
  const point = getNormalizedPoint(event);
  activeBox = {
    x1: Math.min(boxStart.x, point.x),
    y1: Math.min(boxStart.y, point.y),
    x2: Math.max(boxStart.x, point.x),
    y2: Math.max(boxStart.y, point.y),
  };
  drawGuides();
}

function finishBox() {
  if (!isDrawingBox || promptMode !== 'box') return;
  isDrawingBox = false;
  const tooSmall = !activeBox || (activeBox.x2 - activeBox.x1) < 0.01 || (activeBox.y2 - activeBox.y1) < 0.01;
  if (tooSmall) activeBox = null;
  points = [];
  renderMarkers();
  drawGuides();
  updateButtons();
  if (activeBox) decode();
}

function makeSourceCanvas() {
  const canvas = document.createElement('canvas');
  canvas.width = refs.imageDisplay.naturalWidth;
  canvas.height = refs.imageDisplay.naturalHeight;
  canvas.getContext('2d').drawImage(refs.imageDisplay, 0, 0, canvas.width, canvas.height);
  return canvas;
}

function downloadCanvas(canvas, filename) {
  try {
    canvas.toBlob(blob => {
      if (!blob) return;
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.download = filename;
      link.href = url;
      link.style.display = 'none';
      document.body.appendChild(link);
      link.click();
      window.setTimeout(() => {
        link.remove();
        URL.revokeObjectURL(url);
      }, 0);
      setStatus('success', `${filename} 已建立，大小 ${fmtBytes(blob.size)}。`);
      log(`Exported ${filename}, ${blob.size} bytes.`);
    }, 'image/png');
  } catch (error) {
    setStatus('error', `${filename} 匯出失敗：${error.message}`);
    log(`Export failed ${filename}: ${error.message}`);
  }
}

function downloadMask() {
  if (!bestMask) return;
  const canvas = document.createElement('canvas');
  canvas.width = bestMask.width;
  canvas.height = bestMask.height;
  const ctx = canvas.getContext('2d');
  const imageData = ctx.createImageData(canvas.width, canvas.height);
  for (let i = 0; i < bestMask.data.length; i++) {
    const value = bestMask.data[i] ? 255 : 0;
    const offset = i * 4;
    imageData.data[offset] = value;
    imageData.data[offset + 1] = value;
    imageData.data[offset + 2] = value;
    imageData.data[offset + 3] = 255;
  }
  ctx.putImageData(imageData, 0, 0);
  downloadCanvas(canvas, 'sam3-mask.png');
}

function downloadOverlay() {
  if (!bestMask) return;
  try {
    const canvas = makeSourceCanvas();
    canvas.getContext('2d').drawImage(refs.maskCanvas, 0, 0, canvas.width, canvas.height);
    downloadCanvas(canvas, 'sam3-overlay.png');
  } catch (error) {
    setStatus('error', `sam3-overlay.png 匯出失敗：${error.message}`);
    log(`Export failed sam3-overlay.png: ${error.message}`);
  }
}

function downloadCutout() {
  if (!bestMask) return;
  try {
    const source = makeSourceCanvas();
    const sourceData = source.getContext('2d').getImageData(0, 0, source.width, source.height);
    const out = document.createElement('canvas');
    out.width = source.width;
    out.height = source.height;
    const outCtx = out.getContext('2d');
    const outData = outCtx.createImageData(out.width, out.height);
    for (let i = 0; i < bestMask.data.length; i++) {
      if (!bestMask.data[i]) continue;
      const offset = i * 4;
      outData.data[offset] = sourceData.data[offset];
      outData.data[offset + 1] = sourceData.data[offset + 1];
      outData.data[offset + 2] = sourceData.data[offset + 2];
      outData.data[offset + 3] = 255;
    }
    outCtx.putImageData(outData, 0, 0);
    downloadCanvas(out, 'sam3-cutout.png');
  } catch (error) {
    setStatus('error', `sam3-cutout.png 匯出失敗：${error.message}`);
    log(`Export failed sam3-cutout.png: ${error.message}`);
  }
}

async function clearCache() {
  if (!window.confirm('清除 Hugging Face / Transformers.js 瀏覽器快取？清除後模型需要重新下載。')) return;
  try {
    if ('caches' in window) {
      const keys = await caches.keys();
      await Promise.all(keys.filter(key => key.toLowerCase().includes('transformers') || key.toLowerCase().includes('huggingface')).map(key => caches.delete(key)));
    }
    setStatus('info', '已嘗試清除 Cache API。IndexedDB 中的模型快取可由瀏覽器站台設定清除。');
    log('Cache clear requested.');
  } catch (error) {
    setStatus('error', `清除快取失敗：${error.message}`);
  }
}

function wireEvents() {
  refs.loadModelBtn.addEventListener('click', loadModel);
  refs.clearCacheBtn.addEventListener('click', clearCache);
  refs.imageInput.addEventListener('change', event => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (lastObjectUrl) URL.revokeObjectURL(lastObjectUrl);
    lastObjectUrl = URL.createObjectURL(file);
    encodeImage(lastObjectUrl);
  });

  refs.uploadArea.addEventListener('click', event => {
    if (event.target.closest('.example-thumb')) return;
    if (!refs.imageInput.disabled) refs.imageInput.click();
  });
  refs.uploadArea.addEventListener('keydown', event => {
    if (event.key !== 'Enter' && event.key !== ' ') return;
    event.preventDefault();
    if (!refs.imageInput.disabled) refs.imageInput.click();
  });

  refs.uploadArea.addEventListener('dragover', event => {
    event.preventDefault();
    refs.uploadArea.classList.add('dragover');
  });
  refs.uploadArea.addEventListener('dragleave', () => refs.uploadArea.classList.remove('dragover'));
  refs.uploadArea.addEventListener('drop', event => {
    event.preventDefault();
    refs.uploadArea.classList.remove('dragover');
    const file = event.dataTransfer.files?.[0];
    if (!file || !file.type.startsWith('image/')) return;
    if (lastObjectUrl) URL.revokeObjectURL(lastObjectUrl);
    lastObjectUrl = URL.createObjectURL(file);
    encodeImage(lastObjectUrl);
  });

  document.querySelectorAll('.example-thumb').forEach(button => {
    button.addEventListener('click', event => {
      event.preventDefault();
      event.stopPropagation();
      encodeImage(button.dataset.src);
    });
  });

  document.querySelectorAll('.segment').forEach(button => {
    button.addEventListener('click', () => setPromptMode(button.dataset.mode));
  });

  refs.imageDisplay.draggable = false;
  refs.imageDisplay.style.webkitUserDrag = 'none';
  refs.stage.addEventListener('contextmenu', event => event.preventDefault());
  refs.stage.addEventListener('dragstart', event => event.preventDefault());
  refs.imageDisplay.addEventListener('dragstart', event => event.preventDefault());
  refs.stage.addEventListener('pointerdown', event => {
    if (!hasImage() || !imageEmbeddings) return;
    if (promptMode === 'box') {
      refs.stage.setPointerCapture?.(event.pointerId);
      startBox(event);
    }
    else if (event.button === 0 || event.button === 2) addPoint(event);
  });
  refs.stage.addEventListener('pointermove', event => {
    updateBox(event);
    updateHoverPreview(event);
  });
  refs.stage.addEventListener('pointerup', event => {
    finishBox();
    refs.stage.releasePointerCapture?.(event.pointerId);
  });
  refs.stage.addEventListener('pointercancel', finishBox);

  refs.decodeBtn.addEventListener('click', decode);
  refs.clearPromptsBtn.addEventListener('click', clearPrompts);
  refs.resetImageBtn.addEventListener('click', resetImage);
  refs.downloadCutoutBtn.addEventListener('click', downloadCutout);
  refs.downloadMaskBtn.addEventListener('click', downloadMask);
  refs.downloadOverlayBtn.addEventListener('click', downloadOverlay);
  refs.overlayAlphaInput.addEventListener('input', redrawMask);
  refs.maskColorInput.addEventListener('input', redrawMask);
  window.addEventListener('resize', () => {
    updateCanvasGeometry();
    renderMarkers();
  });
}

function init() {
  if (mockMode) {
    refs.backendBadge.textContent = 'Mock WebGPU';
    setStatus('info', 'Mock 測試模式：會自動初始化 mock 模型，不下載 5.49GB 權重。');
  } else if (!navigator.gpu) {
    setStatus('error', '未偵測到 WebGPU。請使用支援 WebGPU 的 Chrome/Edge。');
    refs.backendBadge.textContent = 'WebGPU required';
  }
  wireEvents();
  updateButtons();
  if (mockMode || navigator.gpu) window.setTimeout(loadModel, 0);
}

init();
