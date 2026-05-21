import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';

const HF_TREE_URL = 'https://huggingface.co/api/models/TencentARC/Pixal3D/tree/main?recursive=1';
const HF_REPO_ID = 'TencentARC/Pixal3D';
const MAX_VERTICES = 256 * 256;
const VERTEX_FLOATS = 16;
const VERTEX_BYTES = VERTEX_FLOATS * 4;
const PIXAL3D_STAGES = [
  { id: 'sparse-structure', name: 'Sparse Structure', resolutions: ['32', '64'], modelHint: 'ss_flow_img_dit_*_proj_finetune' },
  { id: 'shape', name: 'Shape', resolutions: ['256', '512', '1024'], modelHint: 'slat_flow_img2shape_*_proj_finetune' },
  { id: 'texture', name: 'Texture / PBR', resolutions: ['256', '512', '1024'], modelHint: 'slat_flow_imgshape2tex_*_proj_finetune' },
];

const refs = {
  statusBadge: document.getElementById('statusBadge'),
  statusMsg: document.getElementById('statusMsg'),
  backendBadge: document.getElementById('backendBadge'),
  dotWebgpu: document.getElementById('dot-webgpu'),
  dotImage: document.getElementById('dot-image'),
  dotMesh: document.getElementById('dot-mesh'),
  dotRepo: document.getElementById('dot-repo'),
  imageInput: document.getElementById('imageInput'),
  dropZone: document.getElementById('dropZone'),
  sourcePreview: document.getElementById('sourcePreview'),
  sampleBtn: document.getElementById('sampleBtn'),
  clearBtn: document.getElementById('clearBtn'),
  resolutionInput: document.getElementById('resolutionInput'),
  modeInput: document.getElementById('modeInput'),
  depthInput: document.getElementById('depthInput'),
  edgeInput: document.getElementById('edgeInput'),
  smoothInput: document.getElementById('smoothInput'),
  depthValue: document.getElementById('depthValue'),
  edgeValue: document.getElementById('edgeValue'),
  smoothValue: document.getElementById('smoothValue'),
  generateBtn: document.getElementById('generateBtn'),
  resetViewBtn: document.getElementById('resetViewBtn'),
  viewerCanvas: document.getElementById('viewerCanvas'),
  viewerEmpty: document.getElementById('viewerEmpty'),
  meshStats: document.getElementById('meshStats'),
  exportGlbBtn: document.getElementById('exportGlbBtn'),
  exportObjBtn: document.getElementById('exportObjBtn'),
  exportManifestBtn: document.getElementById('exportManifestBtn'),
  refreshRepoBtn: document.getElementById('refreshRepoBtn'),
  stageGrid: document.getElementById('stageGrid'),
  browserWeightStatus: document.getElementById('browserWeightStatus'),
  runLog: document.getElementById('runLog'),
};

let gpuDevice = null;
let computePipeline = null;
let inputBuffer = null;
let outputBuffer = null;
let readBuffer = null;
let uniformBuffer = null;
let currentImageBitmap = null;
let currentMeshObject = null;
let lastGeometryData = null;
let pixal3dManifest = null;
let renderer = null;
let scene = null;
let camera = null;
let controls = null;

function setStatus(type, message) {
  refs.statusBadge.className = `status-badge ${type}`;
  refs.statusBadge.textContent = type.toUpperCase();
  refs.statusMsg.textContent = message;
}

function setDot(dot, state) {
  dot.className = `dot ${state}`;
}

function log(message) {
  const timestamp = new Date().toLocaleTimeString('zh-TW', { hour12: false });
  refs.runLog.textContent = `[${timestamp}] ${message}\n${refs.runLog.textContent}`.trim();
}

function updateSliderLabels() {
  refs.depthValue.textContent = Number(refs.depthInput.value).toFixed(2);
  refs.edgeValue.textContent = Number(refs.edgeInput.value).toFixed(2);
  refs.smoothValue.textContent = refs.smoothInput.value;
}

function initThree() {
  renderer = new THREE.WebGLRenderer({ canvas: refs.viewerCanvas, antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.outputColorSpace = THREE.SRGBColorSpace;

  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera(45, 1, 0.02, 100);
  camera.position.set(0, 1.15, 2.25);
  controls = new OrbitControls(camera, refs.viewerCanvas);
  controls.enableDamping = true;
  controls.target.set(0, 0.08, 0);

  const ambient = new THREE.HemisphereLight(0xf0fff8, 0x172031, 1.25);
  const key = new THREE.DirectionalLight(0xffffff, 1.65);
  key.position.set(2.5, 3.5, 2.2);
  const rim = new THREE.DirectionalLight(0x8ff0ca, 0.85);
  rim.position.set(-2.2, 1.4, -2.6);
  const grid = new THREE.GridHelper(2.6, 16, 0x33534a, 0x20332d);
  grid.position.y = -0.62;
  scene.add(ambient, key, rim, grid);

  window.addEventListener('resize', resizeRenderer);
  resizeRenderer();
  animate();
}

function resizeRenderer() {
  const rect = refs.viewerCanvas.parentElement.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width));
  const height = Math.max(1, Math.floor(rect.height));
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}

function resetView() {
  camera.position.set(0, 1.15, 2.25);
  controls.target.set(0, 0.08, 0);
  controls.update();
}

async function initWebGPU() {
  setDot(refs.dotWebgpu, 'loading');
  if (!window.isSecureContext) {
    throw new Error('WebGPU 需要 secure context。請用 run.bat 開啟 localhost 後再進入此工具。');
  }
  if (!navigator.gpu) {
    throw new Error('此瀏覽器沒有 navigator.gpu。請使用支援 WebGPU 的 Chrome / Edge。');
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('找不到可用的 WebGPU adapter。');
  gpuDevice = await adapter.requestDevice();
  gpuDevice.lost.then(info => {
    setDot(refs.dotWebgpu, 'error');
    setStatus('error', `WebGPU device lost: ${info.message || info.reason}`);
  });

  createComputePipeline();
  allocateBuffers();
  const adapterName = adapter.info?.description || adapter.info?.vendor || 'WebGPU adapter';
  refs.backendBadge.textContent = adapterName;
  setDot(refs.dotWebgpu, 'done');
  setStatus('success', 'WebGPU 已就緒。載入圖片後即可生成 3D mesh。');
  log(`WebGPU device ready: ${adapterName}`);
}

function createComputePipeline() {
  const shader = gpuDevice.createShaderModule({
    label: 'Pixal3D pixel lifting shader',
    code: `
struct Params {
  width: u32,
  height: u32,
  depth: f32,
  edgeGain: f32,
  smoothRadius: u32,
  _pad0: u32,
  _pad1: u32,
  _pad2: u32,
}

struct VertexOut {
  position: vec4<f32>,
  normal: vec4<f32>,
  uv: vec4<f32>,
  color: vec4<f32>,
}

@group(0) @binding(0) var<storage, read> pixels: array<u32>;
@group(0) @binding(1) var<storage, read_write> output: array<VertexOut>;
@group(0) @binding(2) var<uniform> params: Params;

fn pixelIndex(x: u32, y: u32) -> u32 {
  return y * params.width + x;
}

fn channel(value: u32, shift: u32) -> f32 {
  return f32((value >> shift) & 255u) / 255.0;
}

fn luminanceAt(xIn: i32, yIn: i32) -> f32 {
  let x = u32(clamp(xIn, 0, i32(params.width) - 1));
  let y = u32(clamp(yIn, 0, i32(params.height) - 1));
  let px = pixels[pixelIndex(x, y)];
  let r = channel(px, 0u);
  let g = channel(px, 8u);
  let b = channel(px, 16u);
  return dot(vec3<f32>(r, g, b), vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn colorAt(index: u32) -> vec4<f32> {
  let px = pixels[index];
  return vec4<f32>(channel(px, 0u), channel(px, 8u), channel(px, 16u), 1.0);
}

fn smoothedLuma(x: i32, y: i32) -> f32 {
  let radius = i32(params.smoothRadius);
  var total = 0.0;
  var count = 0.0;
  for (var yy = -3; yy <= 3; yy = yy + 1) {
    for (var xx = -3; xx <= 3; xx = xx + 1) {
      if (abs(xx) <= radius && abs(yy) <= radius) {
        total = total + luminanceAt(x + xx, y + yy);
        count = count + 1.0;
      }
    }
  }
  return total / max(count, 1.0);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  if (id.x >= params.width || id.y >= params.height) { return; }

  let index = pixelIndex(id.x, id.y);
  let xi = i32(id.x);
  let yi = i32(id.y);
  let luma = smoothedLuma(xi, yi);
  let left = smoothedLuma(xi - 1, yi);
  let right = smoothedLuma(xi + 1, yi);
  let up = smoothedLuma(xi, yi - 1);
  let down = smoothedLuma(xi, yi + 1);
  let edge = abs(left - right) + abs(up - down);

  let aspect = f32(params.width) / f32(params.height);
  let u = f32(id.x) / max(f32(params.width - 1u), 1.0);
  let v = f32(id.y) / max(f32(params.height - 1u), 1.0);
  let x = (u - 0.5) * 2.0 * aspect;
  let z = (v - 0.5) * -2.0;
  let height = ((1.0 - luma) * params.depth) + (edge * params.edgeGain);
  let dx = (right - left) * params.depth;
  let dz = (down - up) * params.depth;
  let normal = normalize(vec3<f32>(-dx, 0.12, -dz));

  output[index].position = vec4<f32>(x, height - params.depth * 0.45, z, 1.0);
  output[index].normal = vec4<f32>(normal, 0.0);
  output[index].uv = vec4<f32>(u, 1.0 - v, 0.0, 0.0);
  output[index].color = colorAt(index);
}
`,
  });

  computePipeline = gpuDevice.createComputePipeline({
    label: 'Pixal3D pixel lifting pipeline',
    layout: 'auto',
    compute: { module: shader, entryPoint: 'main' },
  });
}

function allocateBuffers() {
  const inputBytes = MAX_VERTICES * 4;
  const outputBytes = MAX_VERTICES * VERTEX_BYTES;
  inputBuffer = gpuDevice.createBuffer({ size: inputBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  outputBuffer = gpuDevice.createBuffer({ size: outputBytes, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
  readBuffer = gpuDevice.createBuffer({ size: outputBytes, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
  uniformBuffer = gpuDevice.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
}

function renderPipelineStages() {
  refs.stageGrid.innerHTML = PIXAL3D_STAGES.map((stage, index) => `
    <article class="stage-card" data-stage="${stage.id}">
      <div class="stage-top">
        <span class="stage-name">${stage.name}</span>
        <span class="stage-index">${index + 1}</span>
      </div>
      <div class="stage-res">${stage.resolutions.map(resolution => `<span>${resolution}</span>`).join('')}</div>
      <div class="stage-model">${stage.modelHint}</div>
      <div class="stage-state">Idle</div>
    </article>
  `).join('');
}

function setStageState(stageId, state) {
  const stage = refs.stageGrid.querySelector(`[data-stage="${stageId}"] .stage-state`);
  if (!stage) return;
  const labels = { idle: 'Idle', queued: 'Queued', running: 'Running', done: 'Mapped' };
  stage.textContent = labels[state] || state;
}

function setAllStageStates(state) {
  PIXAL3D_STAGES.forEach(stage => setStageState(stage.id, state));
}

function createPixal3DManifest(files, browserReady, checkpoints) {
  const paths = files.map(file => file.path || '').filter(Boolean);
  return {
    repo: HF_REPO_ID,
    checkedAt: new Date().toISOString(),
    frontendRuntime: 'WebGPU only',
    officialPipeline: PIXAL3D_STAGES,
    browserReadyFiles: browserReady,
    checkpointFiles: checkpoints,
    configFiles: paths.filter(path => /(^|\/)(configs?|requirements|inference|app|data_toolkit).*|\.(json|yaml|yml|py|md)$/i.test(path)).slice(0, 200),
    note: 'Official Pixal3D currently targets a PyTorch/CUDA pipeline. Browser-native weights were not found unless browserReadyFiles is non-empty.',
  };
}

function exportManifest() {
  const manifest = pixal3dManifest || createPixal3DManifest([], [], []);
  const blob = new Blob([JSON.stringify(manifest, null, 2)], { type: 'application/json' });
  downloadBlob(blob, 'pixal3d-webgpu-manifest.json');
}

async function checkPixal3DRepo() {
  setDot(refs.dotRepo, 'loading');
  refs.refreshRepoBtn.disabled = true;
  try {
    const response = await fetch(HF_TREE_URL, { headers: { accept: 'application/json' } });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const files = await response.json();
    const paths = files.map(file => file.path || '').filter(Boolean);
    const browserReady = paths.filter(path => /\.(onnx|ort|tflite|wasm|webgpu\.json)$/i.test(path));
    const checkpoints = paths.filter(path => /\.(safetensors|pt|pth|ckpt|bin)$/i.test(path));
    pixal3dManifest = createPixal3DManifest(files, browserReady, checkpoints);
    refs.exportManifestBtn.disabled = false;
    if (browserReady.length) {
      refs.browserWeightStatus.textContent = `找到 ${browserReady.length} 個候選檔`;
      log(`Pixal3D repo has browser-candidate files: ${browserReady.slice(0, 5).join(', ')}`);
    } else {
      refs.browserWeightStatus.textContent = '未提供瀏覽器權重';
      log(`Pixal3D manifest ready: ${checkpoints.length} checkpoint-like files, 0 browser-native ONNX/WebGPU files, ${pixal3dManifest.configFiles.length} config/source entries indexed.`);
    }
    setDot(refs.dotRepo, 'done');
  } catch (error) {
    refs.browserWeightStatus.textContent = '檢查失敗';
    pixal3dManifest = createPixal3DManifest([], [], []);
    refs.exportManifestBtn.disabled = false;
    setDot(refs.dotRepo, 'error');
    log(`Pixal3D repo check failed: ${error.message}`);
  } finally {
    refs.refreshRepoBtn.disabled = false;
  }
}

async function loadImageFile(file) {
  if (!file) return;
  const bitmap = await createImageBitmap(file, { imageOrientation: 'from-image' });
  setCurrentImage(bitmap, file.name);
}

function setCurrentImage(bitmap, label) {
  currentImageBitmap?.close?.();
  currentImageBitmap = bitmap;
  drawPreview(bitmap);
  setDot(refs.dotImage, 'done');
  setDot(refs.dotMesh, 'pending');
  setStatus('info', `${label} 已載入。按下 WebGPU 生成 mesh。`);
  log(`Image loaded: ${label} (${bitmap.width} × ${bitmap.height})`);
}

function drawPreview(bitmap) {
  const canvas = refs.sourcePreview;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  const size = canvas.width;
  ctx.clearRect(0, 0, size, size);
  ctx.fillStyle = '#0b1018';
  ctx.fillRect(0, 0, size, size);
  const scale = Math.min(size / bitmap.width, size / bitmap.height);
  const width = bitmap.width * scale;
  const height = bitmap.height * scale;
  ctx.drawImage(bitmap, (size - width) / 2, (size - height) / 2, width, height);
}

async function loadSampleImage() {
  const canvas = document.createElement('canvas');
  canvas.width = 768;
  canvas.height = 768;
  const ctx = canvas.getContext('2d');
  const gradient = ctx.createLinearGradient(0, 0, 768, 768);
  gradient.addColorStop(0, '#f6d365');
  gradient.addColorStop(0.45, '#3dd6a3');
  gradient.addColorStop(1, '#3245a8');
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, 768, 768);
  ctx.fillStyle = 'rgba(255,255,255,.9)';
  ctx.beginPath();
  ctx.ellipse(384, 350, 190, 235, -0.1, 0, Math.PI * 2);
  ctx.fill();
  ctx.fillStyle = '#1f2937';
  ctx.beginPath();
  ctx.arc(310, 330, 26, 0, Math.PI * 2);
  ctx.arc(460, 330, 26, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = '#1f2937';
  ctx.lineWidth = 18;
  ctx.lineCap = 'round';
  ctx.beginPath();
  ctx.moveTo(315, 455);
  ctx.quadraticCurveTo(384, 520, 470, 450);
  ctx.stroke();
  ctx.fillStyle = 'rgba(255,255,255,.42)';
  for (let i = 0; i < 26; i += 1) {
    const x = 60 + Math.random() * 650;
    const y = 60 + Math.random() * 650;
    const r = 8 + Math.random() * 28;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.fill();
  }
  const bitmap = await createImageBitmap(canvas);
  setCurrentImage(bitmap, 'sample image');
}

function clearImage() {
  currentImageBitmap?.close?.();
  currentImageBitmap = null;
  refs.imageInput.value = '';
  const ctx = refs.sourcePreview.getContext('2d');
  ctx.clearRect(0, 0, refs.sourcePreview.width, refs.sourcePreview.height);
  removeCurrentMesh();
  setDot(refs.dotImage, 'pending');
  setDot(refs.dotMesh, 'pending');
  refs.meshStats.textContent = '尚未生成 mesh。';
  refs.viewerEmpty.classList.remove('hidden');
  setAllStageStates('idle');
  setStatus(gpuDevice ? 'success' : 'loading', gpuDevice ? 'WebGPU 已就緒。載入圖片後即可生成 3D mesh。' : '正在檢查 WebGPU 裝置…');
}

function imageToPixels(bitmap, size) {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, size, size);
  const scale = Math.min(size / bitmap.width, size / bitmap.height);
  const width = bitmap.width * scale;
  const height = bitmap.height * scale;
  ctx.drawImage(bitmap, (size - width) / 2, (size - height) / 2, width, height);
  const imageData = ctx.getImageData(0, 0, size, size);
  return new Uint32Array(imageData.data.buffer.slice(0));
}

async function generateMesh() {
  if (!gpuDevice) {
    setStatus('error', 'WebGPU 尚未就緒。');
    return;
  }
  if (!currentImageBitmap) {
    setStatus('error', '請先載入圖片。');
    return;
  }

  const size = Number.parseInt(refs.resolutionInput.value, 10);
  const vertexCount = size * size;
  const depth = Number.parseFloat(refs.depthInput.value);
  const edgeGain = Number.parseFloat(refs.edgeInput.value);
  const smoothRadius = Number.parseInt(refs.smoothInput.value, 10);
  const pixels = imageToPixels(currentImageBitmap, size);

  refs.generateBtn.disabled = true;
  setDot(refs.dotMesh, 'loading');
  setAllStageStates('queued');
  setStageState('sparse-structure', 'running');
  setStatus('loading', `WebGPU compute pass running at ${size} × ${size}…`);
  log(`Pipeline stage 1 Sparse Structure: lifting ${size} × ${size} pixel features.`);

  const params = new ArrayBuffer(32);
  const view = new DataView(params);
  view.setUint32(0, size, true);
  view.setUint32(4, size, true);
  view.setFloat32(8, depth, true);
  view.setFloat32(12, edgeGain, true);
  view.setUint32(16, smoothRadius, true);

  const byteCount = vertexCount * VERTEX_BYTES;
  gpuDevice.queue.writeBuffer(inputBuffer, 0, pixels);
  gpuDevice.queue.writeBuffer(uniformBuffer, 0, params);

  const bindGroup = gpuDevice.createBindGroup({
    layout: computePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: inputBuffer } },
      { binding: 1, resource: { buffer: outputBuffer } },
      { binding: 2, resource: { buffer: uniformBuffer } },
    ],
  });

  const encoder = gpuDevice.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(computePipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(size / 8), Math.ceil(size / 8));
  pass.end();
  encoder.copyBufferToBuffer(outputBuffer, 0, readBuffer, 0, byteCount);
  gpuDevice.queue.submit([encoder.finish()]);
  await readBuffer.mapAsync(GPUMapMode.READ, 0, byteCount);
  const data = new Float32Array(readBuffer.getMappedRange(0, byteCount).slice(0));
  readBuffer.unmap();

  setStageState('sparse-structure', 'done');
  setStageState('shape', 'running');
  log('Pipeline stage 2 Shape: assembling WebGPU output into surface geometry.');
  buildThreeObject(data, size, refs.modeInput.value);
  setStageState('shape', 'done');
  setStageState('texture', 'running');
  log('Pipeline stage 3 Texture / PBR: transferring source colors as vertex color material.');
  setStageState('texture', 'done');
  refs.generateBtn.disabled = false;
  setDot(refs.dotMesh, 'done');
  refs.exportGlbBtn.disabled = false;
  refs.exportObjBtn.disabled = false;
  refs.viewerEmpty.classList.add('hidden');
  setStatus('success', 'WebGPU mesh generated. 可在右側旋轉預覽並匯出。');
  log(`Generated ${refs.modeInput.value} mesh: ${vertexCount.toLocaleString()} vertices, ${size} × ${size}, depth=${depth.toFixed(2)}, edge=${edgeGain.toFixed(2)}.`);
}

function removeCurrentMesh() {
  if (!currentMeshObject) return;
  scene.remove(currentMeshObject);
  currentMeshObject.traverse?.(object => {
    object.geometry?.dispose?.();
    if (Array.isArray(object.material)) object.material.forEach(material => material.dispose?.());
    else object.material?.dispose?.();
  });
  currentMeshObject = null;
  lastGeometryData = null;
  refs.exportGlbBtn.disabled = true;
  refs.exportObjBtn.disabled = true;
}

function buildThreeObject(data, size, mode) {
  removeCurrentMesh();
  const vertexCount = size * size;
  const positions = new Float32Array(vertexCount * 3);
  const normals = new Float32Array(vertexCount * 3);
  const colors = new Float32Array(vertexCount * 3);
  const uvs = new Float32Array(vertexCount * 2);

  for (let index = 0; index < vertexCount; index += 1) {
    const offset = index * VERTEX_FLOATS;
    positions.set([data[offset], data[offset + 1], data[offset + 2]], index * 3);
    normals.set([data[offset + 4], data[offset + 5], data[offset + 6]], index * 3);
    uvs.set([data[offset + 8], data[offset + 9]], index * 2);
    colors.set([data[offset + 12], data[offset + 13], data[offset + 14]], index * 3);
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('normal', new THREE.BufferAttribute(normals, 3));
  geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  geometry.setAttribute('uv', new THREE.BufferAttribute(uvs, 2));

  let faces = 0;
  if (mode === 'surface') {
    const indices = [];
    for (let y = 0; y < size - 1; y += 1) {
      for (let x = 0; x < size - 1; x += 1) {
        const a = y * size + x;
        const b = a + 1;
        const c = a + size;
        const d = c + 1;
        indices.push(a, c, b, b, c, d);
        faces += 2;
      }
    }
    geometry.setIndex(indices);
    const material = new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.78, metalness: 0.02, side: THREE.DoubleSide });
    currentMeshObject = new THREE.Mesh(geometry, material);
  } else {
    const material = new THREE.PointsMaterial({ size: 0.015, vertexColors: true, sizeAttenuation: true });
    currentMeshObject = new THREE.Points(geometry, material);
  }

  currentMeshObject.name = 'Pixal3D_WebGPU_Output';
  scene.add(currentMeshObject);
  lastGeometryData = { positions, normals, colors, uvs, size, mode, faces };
  refs.meshStats.textContent = `${vertexCount.toLocaleString()} vertices${faces ? ` / ${faces.toLocaleString()} triangles` : ''}，WebGPU compute 生成。`;
  resetView();
}

function exportGlb() {
  if (!currentMeshObject) return;
  const exporter = new GLTFExporter();
  exporter.parse(
    currentMeshObject,
    result => {
      const blob = result instanceof ArrayBuffer
        ? new Blob([result], { type: 'model/gltf-binary' })
        : new Blob([JSON.stringify(result)], { type: 'model/gltf+json' });
      downloadBlob(blob, 'pixal3d-webgpu-output.glb');
    },
    error => setStatus('error', `GLB export failed: ${error.message}`),
    { binary: true, onlyVisible: true },
  );
}

function exportObj() {
  if (!lastGeometryData) return;
  const { positions, size, mode } = lastGeometryData;
  const lines = ['# Pixal3D WebGPU OBJ export'];
  for (let index = 0; index < positions.length; index += 3) {
    lines.push(`v ${positions[index].toFixed(6)} ${positions[index + 1].toFixed(6)} ${positions[index + 2].toFixed(6)}`);
  }
  if (mode === 'surface') {
    for (let y = 0; y < size - 1; y += 1) {
      for (let x = 0; x < size - 1; x += 1) {
        const a = y * size + x + 1;
        const b = a + 1;
        const c = a + size;
        const d = c + 1;
        lines.push(`f ${a} ${c} ${b}`);
        lines.push(`f ${b} ${c} ${d}`);
      }
    }
  }
  downloadBlob(new Blob([lines.join('\n')], { type: 'text/plain' }), 'pixal3d-webgpu-output.obj');
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function wireEvents() {
  refs.imageInput.addEventListener('change', event => loadImageFile(event.target.files?.[0]));
  refs.sampleBtn.addEventListener('click', loadSampleImage);
  refs.clearBtn.addEventListener('click', clearImage);
  refs.generateBtn.addEventListener('click', generateMesh);
  refs.resetViewBtn.addEventListener('click', resetView);
  refs.exportGlbBtn.addEventListener('click', exportGlb);
  refs.exportObjBtn.addEventListener('click', exportObj);
  refs.exportManifestBtn.addEventListener('click', exportManifest);
  refs.refreshRepoBtn.addEventListener('click', checkPixal3DRepo);
  [refs.depthInput, refs.edgeInput, refs.smoothInput].forEach(input => input.addEventListener('input', updateSliderLabels));

  refs.dropZone.addEventListener('dragover', event => {
    event.preventDefault();
    refs.dropZone.classList.add('dragover');
  });
  refs.dropZone.addEventListener('dragleave', () => refs.dropZone.classList.remove('dragover'));
  refs.dropZone.addEventListener('drop', event => {
    event.preventDefault();
    refs.dropZone.classList.remove('dragover');
    loadImageFile(event.dataTransfer.files?.[0]);
  });
}

async function boot() {
  renderPipelineStages();
  initThree();
  wireEvents();
  updateSliderLabels();
  refs.generateBtn.disabled = true;
  try {
    await initWebGPU();
    refs.generateBtn.disabled = false;
  } catch (error) {
    setDot(refs.dotWebgpu, 'error');
    setStatus('error', error.message);
    refs.backendBadge.textContent = 'Unavailable';
    log(error.message);
  }
  checkPixal3DRepo();
}

boot();