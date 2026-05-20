const FALLBACK_MODELS = [
  model('fal-ai/flux/dev', 'FLUX.1 [dev]', 'text-to-image', '高品質文字生圖，支援尺寸、steps、張數與 safety checker。'),
  model('fal-ai/flux/schnell', 'FLUX.1 [schnell]', 'text-to-image', '快速文字生圖模型。'),
  model('fal-ai/flux-pro/v1.1', 'FLUX1.1 [pro]', 'text-to-image', 'FLUX pro 文字生圖。'),
  model('fal-ai/flux-pro/v1.1-ultra', 'FLUX1.1 [pro] Ultra', 'text-to-image', '高解析文字生圖。'),
  model('fal-ai/flux-pro/kontext', 'FLUX.1 Kontext [pro]', 'image-to-image', '文字與參考圖編輯。'),
  model('fal-ai/flux-pro/kontext/max', 'FLUX.1 Kontext [max]', 'image-to-image', '高品質 Kontext 圖像編輯。'),
  model('fal-ai/flux-2', 'FLUX.2 [dev]', 'text-to-image', 'FLUX.2 文字生圖。'),
  model('fal-ai/flux-2/turbo', 'FLUX.2 Turbo', 'text-to-image', '快速 FLUX.2 文字生圖。'),
  model('fal-ai/flux-2-pro', 'FLUX.2 [pro]', 'text-to-image', 'FLUX.2 pro 文字生圖。'),
  model('fal-ai/nano-banana', 'Nano Banana', 'text-to-image', 'Google image generation endpoint。'),
  model('fal-ai/nano-banana/edit', 'Nano Banana Edit', 'image-to-image', 'Nano Banana 圖像編輯。'),
  model('fal-ai/nano-banana-2', 'Nano Banana 2', 'text-to-image', 'Google 新版快速圖像生成。'),
  model('fal-ai/nano-banana-2/edit', 'Nano Banana 2 Edit', 'image-to-image', 'Nano Banana 2 圖像編輯。'),
  model('fal-ai/nano-banana-pro', 'Nano Banana Pro', 'text-to-image', '高品質圖像生成。'),
  model('fal-ai/nano-banana-pro/edit', 'Nano Banana Pro Edit', 'image-to-image', '高品質圖像編輯。'),
  model('fal-ai/gemini-3-pro-image-preview', 'Gemini 3 Pro Image', 'text-to-image', 'Google 高保真圖像生成。'),
  model('fal-ai/gemini-25-flash-image', 'Gemini 2.5 Flash Image', 'text-to-image', '快速圖像生成。'),
  model('fal-ai/imagen4/preview', 'Imagen 4 Preview', 'text-to-image', 'Google Imagen 4 preview。'),
  model('fal-ai/imagen4/preview/fast', 'Imagen 4 Fast', 'text-to-image', '快速 Imagen 4。'),
  model('fal-ai/gpt-image-1.5', 'GPT Image 1.5', 'text-to-image', 'GPT image generation。'),
  model('openai/gpt-image-2', 'GPT Image 2', 'text-to-image', 'OpenAI GPT Image 2 via fal。'),
  model('openai/gpt-image-2/edit', 'GPT Image 2 Edit', 'image-to-image', 'OpenAI image editing via fal。'),
  model('fal-ai/recraft/v4/text-to-image', 'Recraft V4', 'text-to-image', '設計導向文字生圖。'),
  model('fal-ai/recraft/v4/pro/text-to-image', 'Recraft V4 Pro', 'text-to-image', '高品質設計導向文字生圖。'),
  model('fal-ai/recraft/v4.1/text-to-image', 'Recraft V4.1', 'text-to-image', 'Recraft V4.1 文字生圖。'),
  model('fal-ai/recraft/v4.1/pro/text-to-image', 'Recraft V4.1 Pro', 'text-to-image', 'Recraft V4.1 Pro 高解析文字生圖。'),
  model('fal-ai/ideogram/v3', 'Ideogram V3', 'text-to-image', '擅長 typography 的圖像生成。'),
  model('fal-ai/bytedance/seedream/v4/text-to-image', 'Seedream 4.0 Text to Image', 'text-to-image', 'ByteDance Seedream 文字生圖。'),
  model('fal-ai/bytedance/seedream/v4/edit', 'Seedream 4.0 Edit', 'image-to-image', 'ByteDance Seedream 圖像編輯。'),
  model('fal-ai/bytedance/seedream/v4.5/text-to-image', 'Seedream 4.5 Text to Image', 'text-to-image', 'ByteDance Seedream 4.5。'),
  model('fal-ai/bytedance/seedream/v5/lite/text-to-image', 'Seedream 5 Lite', 'text-to-image', 'Seedream 5 lite 文字生圖。'),
  model('fal-ai/qwen-image', 'Qwen Image', 'text-to-image', 'Qwen 圖像生成。'),
  model('fal-ai/z-image/turbo', 'Z-Image Turbo', 'text-to-image', '快速文字生圖。'),
  model('fal-ai/fast-sdxl', 'Fast SDXL', 'text-to-image', '快速 SDXL endpoint。'),
  model('fal-ai/birefnet/v2', 'BiRefNet v2', 'image-to-image', '高解析去背/分割。'),
  model('fal-ai/imageutils/rembg', 'Rembg', 'image-to-image', '圖片去背。'),
  model('fal-ai/bria/background/remove', 'Bria Background Remove', 'image-to-image', 'Bria 圖片去背。'),
  model('fal-ai/topaz/upscale/image', 'Topaz Image Upscale', 'image-to-image', '圖片放大。'),
  model('pixelcut/background-removal', 'Pixelcut Background Removal', 'image-to-image', 'Pixelcut 去背。'),
  model('fal-ai/kling-video/v1/standard/text-to-video', 'Kling 1.0 Standard T2V', 'text-to-video', 'Kling 文字轉影片。'),
  model('fal-ai/kling-video/v1.6/standard/text-to-video', 'Kling 1.6 Standard T2V', 'text-to-video', 'Kling 1.6 文字轉影片。'),
  model('fal-ai/kling-video/v2.1/pro/text-to-video', 'Kling 2.1 Pro T2V', 'text-to-video', 'Kling 2.1 Pro 文字轉影片。'),
  model('fal-ai/kling-video/v2.5-turbo/pro/text-to-video', 'Kling 2.5 Turbo Pro T2V', 'text-to-video', 'Kling 2.5 文字轉影片。'),
  model('fal-ai/kling-video/v3/standard/text-to-video', 'Kling 3.0 Standard T2V', 'text-to-video', 'Kling 3.0 文字轉影片。'),
  model('fal-ai/kling-video/v3/pro/text-to-video', 'Kling 3.0 Pro T2V', 'text-to-video', 'Kling 3.0 Pro 文字轉影片。'),
  model('fal-ai/kling-video/v3/pro/image-to-video', 'Kling 3.0 Pro I2V', 'image-to-video', 'Kling 3.0 圖片轉影片。'),
  model('fal-ai/kling-video/v3/standard/image-to-video', 'Kling 3.0 Standard I2V', 'image-to-video', 'Kling 3.0 圖片轉影片。'),
  model('fal-ai/veo3', 'Veo 3', 'text-to-video', 'Google Veo 3 文字轉影片。'),
  model('fal-ai/veo3/fast', 'Veo 3 Fast', 'text-to-video', 'Google Veo 3 fast。'),
  model('fal-ai/veo3.1', 'Veo 3.1', 'text-to-video', 'Google Veo 3.1 文字轉影片。'),
  model('fal-ai/veo3.1/fast', 'Veo 3.1 Fast', 'text-to-video', 'Google Veo 3.1 fast。'),
  model('fal-ai/veo3.1/image-to-video', 'Veo 3.1 Image to Video', 'image-to-video', 'Google Veo 3.1 圖片轉影片。'),
  model('fal-ai/veo3.1/fast/image-to-video', 'Veo 3.1 Fast I2V', 'image-to-video', 'Google Veo 3.1 fast 圖片轉影片。'),
  model('bytedance/seedance-2.0/text-to-video', 'Seedance 2.0 T2V', 'text-to-video', 'ByteDance Seedance 2 文字轉影片。'),
  model('bytedance/seedance-2.0/image-to-video', 'Seedance 2.0 I2V', 'image-to-video', 'ByteDance Seedance 2 圖片轉影片。'),
  model('bytedance/seedance-2.0/reference-to-video', 'Seedance 2.0 Reference to Video', 'image-to-video', '多參考素材轉影片。'),
  model('fal-ai/pixverse/v6/text-to-video', 'PixVerse V6 T2V', 'text-to-video', 'PixVerse 文字轉影片。'),
  model('fal-ai/pixverse/v6/image-to-video', 'PixVerse V6 I2V', 'image-to-video', 'PixVerse 圖片轉影片。'),
  model('fal-ai/sora-2/text-to-video', 'Sora 2 T2V', 'text-to-video', 'Sora 2 文字轉影片。'),
  model('fal-ai/sora-2/image-to-video', 'Sora 2 I2V', 'image-to-video', 'Sora 2 圖片轉影片。'),
  model('fal-ai/minimax/hailuo-02/standard/text-to-video', 'MiniMax Hailuo 02 Standard T2V', 'text-to-video', 'MiniMax Hailuo 文字轉影片。'),
  model('fal-ai/minimax/hailuo-02/pro/text-to-video', 'MiniMax Hailuo 02 Pro T2V', 'text-to-video', 'MiniMax Hailuo Pro。'),
  model('fal-ai/wan/v2.7/text-to-video', 'Wan 2.7 T2V', 'text-to-video', 'Wan 2.7 文字轉影片。'),
  model('fal-ai/ltx-2.3/text-to-video/fast', 'LTX 2.3 Fast T2V', 'text-to-video', 'LTX fast 文字轉影片。'),
  model('fal-ai/ffmpeg-api/images-to-video', 'Images to Video', 'image-to-video', '圖片序列轉 MP4。'),
  model('fal-ai/minimax/speech-2.8-hd', 'MiniMax Speech 2.8 HD', 'text-to-speech', 'MiniMax 文字轉語音。'),
  model('fal-ai/minimax/speech-2.8-turbo', 'MiniMax Speech 2.8 Turbo', 'text-to-speech', 'MiniMax 快速文字轉語音。'),
  model('fal-ai/minimax/voice-clone', 'MiniMax Voice Clone', 'text-to-speech', '聲音克隆與語音合成。'),
  model('fal-ai/elevenlabs/tts/turbo-v2.5', 'ElevenLabs TTS Turbo', 'text-to-speech', 'ElevenLabs TTS via fal。'),
  model('fal-ai/gemini-3.1-flash-tts', 'Gemini 3.1 Flash TTS', 'text-to-speech', 'Google TTS endpoint。'),
  model('fal-ai/chatterbox/text-to-speech', 'Chatterbox TTS', 'text-to-speech', 'Chatterbox 文字轉語音。'),
  model('fal-ai/stable-audio-25/text-to-audio', 'Stable Audio 2.5', 'text-to-audio', '音樂與音效生成。'),
  model('fal-ai/ace-step', 'ACE-Step', 'text-to-audio', '文字生成音樂。'),
  model('fal-ai/meshy/v6/multi-image-to-3d', 'Meshy 6 Multi Image to 3D', 'image-to-3d', '多圖轉 3D。'),
  model('fal-ai/pixal3d', 'Pixal3D', 'image-to-3d', '單圖轉 3D。')
];

const state = {
  models: [...FALLBACK_MODELS],
  selectedModel: null,
  openapi: null,
  inputSchema: null,
  outputSchema: null,
  fields: [],
  serverUrl: '',
  submitPath: '',
  requestId: '',
  remoteLoaded: false
};

const $ = selector => document.querySelector(selector);
const modelSearch = $('#modelSearch');
const modelList = $('#modelList');
const modelOptions = $('#modelOptions');
const dynamicFields = $('#dynamicFields');

init();

function init() {
  modelSearch.addEventListener('input', renderModelList);
  modelSearch.addEventListener('keydown', event => {
    if (event.key === 'Enter') {
      event.preventDefault();
      selectModel(modelIdFromSearch());
    }
  });
  $('#refreshCatalogBtn').addEventListener('click', loadRemoteCatalog);
  $('#loadSchemaBtn').addEventListener('click', () => selectModel(modelIdFromSearch() || state.selectedModel?.id));
  $('#saveProxyBtn').addEventListener('click', saveProxyUrl);
  $('#clearProxyBtn').addEventListener('click', clearProxyUrl);
  $('#saveKeyBtn').addEventListener('click', saveKey);
  $('#clearKeyBtn').addEventListener('click', clearKey);
  $('#proxyUrlInput').addEventListener('input', () => {
    updateProxyState();
    updateRequestPreview();
  });
  $('#falForm').addEventListener('submit', callFal);
  document.querySelectorAll('.code-tab').forEach(btn => btn.addEventListener('click', switchCodeTab));

  $('#proxyUrlInput').value = readCookie('endman_fal_proxy_url');
  $('#apiKeyInput').value = readCookie('endman_fal_key');
  updateProxyState();
  renderModelList();
  selectModel('fal-ai/flux/dev');
  loadRemoteCatalog();
}

function model(id, title, category, description, thumbnailUrl = '') {
  return { id, title, category, description, thumbnailUrl };
}

async function loadRemoteCatalog() {
  setCatalogStatus(hasProxyUrl() ? '正在透過 proxy 載入 fal.ai models 目錄...' : '正在嘗試直連 fal.ai models 目錄...');
  try {
    const response = await fetch(modelsUrl(), { credentials: 'omit' });
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    const contentType = response.headers.get('content-type') || '';
    const parsed = contentType.includes('application/json') ? modelsFromJson(await response.json()) : parseModelLinks(await response.text());
    if (!parsed.length) throw new Error('沒有從 models 頁面解析到 model ID');
    mergeModels(parsed);
    state.remoteLoaded = true;
    setCatalogStatus(`已載入 ${state.models.length} 個模型。若缺少新模型，也可直接輸入完整 model ID。`);
  } catch (error) {
    const hint = hasProxyUrl() ? '請確認 proxy URL 可用並允許此網頁來源。' : 'fal.ai 直連通常會被 CORS 擋住；可填入 proxy URL 後重新載入。';
    setCatalogStatus(`使用內建模型清單。${hint} 錯誤：${error.message}`);
  }
  renderModelList();
}

function modelsFromJson(payload) {
  if (Array.isArray(payload)) return payload.map(normalizeModel).filter(Boolean);
  if (Array.isArray(payload.models)) return payload.models.map(normalizeModel).filter(Boolean);
  if (Array.isArray(payload.data)) return payload.data.map(normalizeModel).filter(Boolean);
  return [];
}

function normalizeModel(item) {
  if (typeof item === 'string') return model(item, titleFromId(item), categoryFromId(item), '由 proxy 回傳的 fal.ai 模型。');
  const id = item.id || item.model_id || item.endpointId || item.endpoint_id;
  if (!id) return null;
  return model(id, item.title || item.name || titleFromId(id), item.category || categoryFromId(id), item.description || item.about || '由 proxy 回傳的 fal.ai 模型。', item.thumbnailUrl || item.thumbnail_url || '');
}

function parseModelLinks(html) {
  const found = [];
  const seen = new Set();
  const regex = /href=["'](?:https:\/\/fal\.ai)?\/models\/([^"'?#]+)(?:[?#][^"']*)?["']/g;
  let match;
  while ((match = regex.exec(html))) {
    let id = decodeURIComponent(match[1]).replace(/\/(api|playground|examples|llms\.txt)$/i, '');
    if (!id.includes('/') || id.includes('[') || id.includes(']')) continue;
    if (seen.has(id)) continue;
    seen.add(id);
    found.push(model(id, titleFromId(id), categoryFromId(id), '從 fal.ai models 頁面解析的模型。'));
  }
  return found;
}

function mergeModels(incoming) {
  const byId = new Map(state.models.map(item => [item.id, item]));
  incoming.forEach(item => {
    const existing = byId.get(item.id);
    byId.set(item.id, existing ? { ...item, ...existing, title: existing.title || item.title } : item);
  });
  state.models = Array.from(byId.values()).sort((a, b) => a.id.localeCompare(b.id));
}

function renderModelList() {
  const items = filteredModels().slice(0, 90);
  const selectedId = state.selectedModel?.id;
  renderModelOptions();
  modelList.innerHTML = items.map(item => `
    <button type="button" class="model-item${item.id === selectedId ? ' active' : ''}" data-id="${escapeAttr(item.id)}">
      <strong>${escapeHtml(item.title || item.id)}</strong>
      <span>${escapeHtml(item.category || 'model')} · ${escapeHtml(item.id)}</span>
    </button>
  `).join('') || `<button type="button" class="model-item" data-id="${escapeAttr(modelSearch.value.trim())}"><strong>使用這個 model ID</strong><span>${escapeHtml(modelSearch.value.trim() || '請輸入 fal model ID')}</span></button>`;
  modelList.querySelectorAll('.model-item').forEach(btn => btn.addEventListener('click', () => selectModel(btn.dataset.id)));
}

function renderModelOptions() {
  modelOptions.innerHTML = state.models.map(item => `<option value="${escapeAttr(item.id)}" label="${escapeAttr(`${item.title || item.id} · ${item.category || 'model'}`)}"></option>`).join('');
}

function filteredModels() {
  const keyword = modelSearch.value.trim().toLowerCase();
  return state.models.filter(item => !keyword || `${item.id} ${item.title} ${item.category} ${item.description}`.toLowerCase().includes(keyword));
}

function modelIdFromSearch() {
  const typed = modelSearch.value.trim();
  if (!typed) return '';
  const exact = state.models.find(item => item.id.toLowerCase() === typed.toLowerCase());
  if (exact) return exact.id;
  const first = filteredModels()[0];
  return first?.id || typed;
}

async function selectModel(modelId) {
  if (!modelId) {
    setStatus('error', '請先輸入或選擇 model ID。');
    return;
  }
  const known = state.models.find(item => item.id === modelId) || model(modelId, titleFromId(modelId), categoryFromId(modelId), '手動輸入的 fal.ai model ID。');
  state.selectedModel = known;
  modelSearch.value = known.id;
  renderSelectedModel(known);
  renderModelList();
  await loadSchema(known.id);
}

function renderSelectedModel(item) {
  $('#modelCategory').textContent = item.category || 'fal.ai';
  $('#modelTitle').textContent = item.title || item.id;
  $('#modelSummary').textContent = item.description || item.id;
  $('#docsLink').href = `https://fal.ai/models/${item.id}/api`;
  if (item.thumbnailUrl) renderPreviewMedia([{ type: guessMediaType(item.thumbnailUrl), src: item.thumbnailUrl }], 'fal metadata preview');
}

async function loadSchema(modelId) {
  setStatus('loading', `正在載入 ${modelId} 的 OpenAPI schema...`);
  dynamicFields.innerHTML = `<div class="empty-state"><p>載入 schema 中...</p></div>`;
  try {
    const schemaUrl = openApiUrl(modelId);
    const response = await fetch(schemaUrl, { credentials: 'omit' });
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    const openapi = await response.json();
    applyOpenApi(openapi, modelId, schemaUrl);
    setStatus('success', `${modelId} schema 已載入。`);
  } catch (error) {
    state.openapi = null;
    state.inputSchema = null;
    state.fields = [];
    dynamicFields.innerHTML = `<div class="text-output">${escapeHtml(`無法載入 schema。\n\n可能原因：model ID 不存在、fal.ai schema 暫時不可用、proxy URL 不正確，或瀏覽器 CORS 擋住直連。\n\nSchema URL:\n${openApiUrl(modelId)}\n\n錯誤：${error.message}`)}</div>`;
    setStatus('error', `schema 載入失敗：${error.message}`);
    updateRequestPreview();
  }
}

function applyOpenApi(openapi, modelId, schemaUrl) {
  const postPath = findPostPath(openapi, modelId);
  const postOperation = openapi.paths?.[postPath]?.post;
  const inputRef = postOperation?.requestBody?.content?.['application/json']?.schema;
  const inputSchema = resolveSchema(inputRef, openapi);
  const outputSchema = findOutputSchema(openapi, modelId);
  const metadata = openapi.info?.['x-fal-metadata'] || {};

  state.openapi = openapi;
  state.inputSchema = inputSchema;
  state.outputSchema = outputSchema;
  updateProxyState();
  state.submitPath = postPath || `/${modelId}`;
  state.selectedModel = {
    ...state.selectedModel,
    category: metadata.category || state.selectedModel.category,
    description: metadata.about || state.selectedModel.description,
    thumbnailUrl: metadata.thumbnailUrl || state.selectedModel.thumbnailUrl
  };

  renderSelectedModel(state.selectedModel);
  renderSchemaMeta(openapi, schemaUrl);
  renderFields(inputSchema);
  updateRequestPreview();
}

function findPostPath(openapi, modelId) {
  const expected = `/${modelId}`;
  if (openapi.paths?.[expected]?.post) return expected;
  return Object.keys(openapi.paths || {}).find(path => path.endsWith(modelId) && openapi.paths[path]?.post) || expected;
}

function findOutputSchema(openapi, modelId) {
  const resultPath = Object.keys(openapi.paths || {}).find(path => path.includes(modelId) && /requests\/\{request_id\}$/.test(path));
  const ref = resultPath ? openapi.paths[resultPath]?.get?.responses?.['200']?.content?.['application/json']?.schema : null;
  return ref ? resolveSchema(ref, openapi) : null;
}

function renderSchemaMeta(openapi, schemaUrl) {
  const metadata = openapi.info?.['x-fal-metadata'] || {};
  $('#schemaMeta').innerHTML = [
    ['Endpoint', metadata.endpointId || state.selectedModel.id],
    ['Category', metadata.category || state.selectedModel.category || '-'],
    ['Proxy', hasProxyUrl() ? proxyBaseUrl() : 'Direct fal.ai'],
    ['Schema', schemaUrl],
    ['Submit Path', state.submitPath],
    ['Output', state.outputSchema?.title || 'Queue result']
  ].map(([label, value]) => `<div class="meta-pill"><strong>${escapeHtml(label)}</strong><span title="${escapeAttr(value)}">${escapeHtml(value)}</span></div>`).join('');
}

function renderFields(schema) {
  const properties = schema?.properties || {};
  const required = new Set(schema?.required || []);
  const order = schema?.['x-fal-order-properties'] || Object.keys(properties);
  const names = [...order, ...Object.keys(properties).filter(name => !order.includes(name))];
  state.fields = names.map(name => makeField(name, properties[name], required.has(name))).filter(Boolean);

  dynamicFields.innerHTML = state.fields.map(field => fieldHtml(field)).join('') || '<div class="empty-state"><p>這個 schema 沒有可顯示的 input 欄位。</p></div>';
  dynamicFields.querySelectorAll('input,select,textarea').forEach(input => input.addEventListener('input', updateRequestPreview));
}

function makeField(name, rawSchema, required) {
  const schema = simplifySchema(rawSchema, state.openapi);
  const enumValues = schema.enum || schema.enumValues || [];
  const defaultValue = schema.default ?? firstExample(schema) ?? (required && schema.type === 'string' ? '' : '');
  const descriptor = schema.description || '';
  const title = schema.title || titleFromName(name);
  const acceptsJson = schema.acceptsJson || schema.type === 'object' || schema.type === 'array';
  let kind = schema.type || 'string';

  if (enumValues.length && schema.acceptsJson) kind = 'enumText';
  else if (enumValues.length) kind = 'select';
  else if (kind === 'boolean') kind = 'checkbox';
  else if (kind === 'integer' || kind === 'number') kind = 'number';
  else if (kind === 'object' || kind === 'array') kind = 'json';
  else if (name.toLowerCase().includes('prompt') || String(defaultValue).length > 90) kind = 'textarea';
  else if (name.toLowerCase().endsWith('_url') || name.toLowerCase().includes('url')) kind = 'url';
  else kind = 'text';

  return { name, title, kind, schema, required, enumValues, defaultValue, description: descriptor, acceptsJson };
}

function simplifySchema(rawSchema, openapi) {
  const schema = resolveSchema(rawSchema, openapi) || {};
  if (!schema.anyOf && !schema.oneOf) return schema;
  const variants = (schema.anyOf || schema.oneOf).map(item => resolveSchema(item, openapi)).filter(Boolean);
  const nonNull = variants.filter(item => item.type !== 'null');
  const enumVariant = nonNull.find(item => item.enum);
  const objectVariant = nonNull.find(item => item.type === 'object');
  const simpleVariant = nonNull.find(item => item.type && item.type !== 'object');
  const chosen = enumVariant || simpleVariant || objectVariant || nonNull[0] || {};
  return { ...schema, ...chosen, acceptsJson: Boolean(objectVariant && enumVariant), enumValues: enumVariant?.enum || chosen.enum || [] };
}

function resolveSchema(schema, openapi) {
  if (!schema) return null;
  if (schema.$ref) {
    const key = schema.$ref.replace('#/components/schemas/', '');
    return openapi.components?.schemas?.[key] || schema;
  }
  return schema;
}

function fieldHtml(field) {
  const full = ['textarea', 'json'].includes(field.kind) ? ' full' : '';
  const requiredMark = field.required ? ' <span class="required-mark">*</span>' : '';
  const help = [field.description, defaultHelp(field), rangeHelp(field)].filter(Boolean).join(' ');
  const helpHtml = help ? `<div class="form-help">${escapeHtml(help)}</div>` : '';
  const value = valueToInput(field.defaultValue, field.kind);
  const baseAttrs = `id="field_${escapeAttr(field.name)}" name="${escapeAttr(field.name)}" data-kind="${escapeAttr(field.kind)}" ${field.required ? 'required' : ''}`;

  if (field.kind === 'select') {
    const options = field.enumValues.map(option => `<option value="${escapeAttr(option)}"${String(option) === String(value) ? ' selected' : ''}>${escapeHtml(option)}</option>`).join('');
    return `<div class="form-group${full}"><label for="field_${escapeAttr(field.name)}">${escapeHtml(field.title)}${requiredMark}</label><select ${baseAttrs}>${options}</select>${helpHtml}</div>`;
  }

  if (field.kind === 'enumText') {
    const listId = `list_${field.name.replace(/[^a-z0-9_-]/gi, '_')}`;
    const options = field.enumValues.map(option => `<option value="${escapeAttr(option)}"></option>`).join('');
    return `<div class="form-group${full}"><label for="field_${escapeAttr(field.name)}">${escapeHtml(field.title)}${requiredMark}</label><input ${baseAttrs} type="text" list="${escapeAttr(listId)}" value="${escapeAttr(value)}"/><datalist id="${escapeAttr(listId)}">${options}</datalist>${helpHtml}</div>`;
  }

  if (field.kind === 'checkbox') {
    return `<div class="form-group${full}"><label for="field_${escapeAttr(field.name)}">${escapeHtml(field.title)}${requiredMark}</label><select ${baseAttrs}><option value="true"${value === true || value === 'true' ? ' selected' : ''}>true</option><option value="false"${value === false || value === 'false' ? ' selected' : ''}>false</option></select>${helpHtml}</div>`;
  }

  if (field.kind === 'textarea' || field.kind === 'json') {
    return `<div class="form-group full"><label for="field_${escapeAttr(field.name)}">${escapeHtml(field.title)}${requiredMark}</label><textarea ${baseAttrs} spellcheck="false">${escapeHtml(value)}</textarea>${helpHtml}</div>`;
  }

  if (field.kind === 'number') {
    const attrs = [field.schema.minimum !== undefined && `min="${escapeAttr(field.schema.minimum)}"`, field.schema.maximum !== undefined && `max="${escapeAttr(field.schema.maximum)}"`, field.schema.type === 'integer' && 'step="1"'].filter(Boolean).join(' ');
    return `<div class="form-group${full}"><label for="field_${escapeAttr(field.name)}">${escapeHtml(field.title)}${requiredMark}</label><input ${baseAttrs} type="number" value="${escapeAttr(value)}" ${attrs}/>${helpHtml}</div>`;
  }

  return `<div class="form-group${full}"><label for="field_${escapeAttr(field.name)}">${escapeHtml(field.title)}${requiredMark}</label><input ${baseAttrs} type="${field.kind === 'url' ? 'url' : 'text'}" value="${escapeAttr(value)}" spellcheck="false"/>${helpHtml}</div>`;
}

function defaultHelp(field) {
  return field.schema.default !== undefined ? `Default: ${JSON.stringify(field.schema.default)}.` : '';
}

function rangeHelp(field) {
  const parts = [];
  if (field.schema.minimum !== undefined) parts.push(`min ${field.schema.minimum}`);
  if (field.schema.maximum !== undefined) parts.push(`max ${field.schema.maximum}`);
  if (!parts.length) return '';
  return `Range: ${parts.join(', ')}.`;
}

function collectInput() {
  const result = {};
  for (const field of state.fields) {
    const input = dynamicFields.querySelector(`[name="${cssEscape(field.name)}"]`);
    if (!input) continue;
    let raw = input.value;
    if (!field.required && raw === '' && field.schema.default === undefined) continue;

    if (field.kind === 'number') {
      if (raw === '') continue;
      result[field.name] = field.schema.type === 'integer' ? parseInt(raw, 10) : Number(raw);
    } else if (field.kind === 'checkbox') {
      result[field.name] = raw === 'true';
    } else if (field.kind === 'json' || (field.acceptsJson && /^[\[{]/.test(raw.trim()))) {
      if (raw.trim() === '') continue;
      result[field.name] = JSON.parse(raw);
    } else {
      result[field.name] = raw;
    }
  }
  return result;
}

async function callFal(event) {
  event.preventDefault();
  const apiKey = $('#apiKeyInput').value.trim();
  if (!apiKey && !hasProxyUrl()) {
    setStatus('error', '請先填入 FAL_KEY；若直連被 CORS 擋住，請改填 proxy URL。');
    return;
  }
  if (!state.inputSchema || !state.submitPath) {
    setStatus('error', '請先載入模型 schema。');
    return;
  }

  $('#sendBtn').disabled = true;
  $('#queueLog').textContent = '';
  try {
    const input = collectInput();
    const request = buildSubmitRequest(input, apiKey, false);
    writeJson('#requestPreview', sanitizeRequest(request, input));
    setStatus('loading', '正在送出 queue request...');

    const submitResponse = await fetch(request.url, request.options);
    const submitData = await parseJsonResponse(submitResponse);
    writeJson('#responsePreview', submitData);
    if (!submitResponse.ok) throw new Error(formatError(submitResponse, submitData));

    if (!submitData.request_id) {
      renderResult(submitData);
      setStatus('success', 'fal 回傳完成。');
      return;
    }

    state.requestId = submitData.request_id;
    appendLog(`request_id: ${state.requestId}`);
    await pollResult(apiKey, state.requestId);
  } catch (error) {
    setStatus('error', `fal 呼叫失敗：${error.message}`);
    appendLog(`error: ${error.message}`);
  } finally {
    $('#sendBtn').disabled = false;
  }
}

async function pollResult(apiKey, requestId) {
  const statusUrl = queueUrl(`${state.submitPath}/requests/${encodeURIComponent(requestId)}/status?logs=1`);
  const resultUrl = queueUrl(`${state.submitPath}/requests/${encodeURIComponent(requestId)}`);

  for (let attempt = 1; attempt <= 180; attempt += 1) {
    setStatus('loading', `任務執行中，正在輪詢狀態 (${attempt})...`);
    const statusResponse = await fetch(statusUrl, { headers: falHeaders(apiKey, false) });
    const statusData = await parseJsonResponse(statusResponse);
    writeJson('#responsePreview', statusData);
    appendStatusLogs(statusData);
    if (!statusResponse.ok) throw new Error(formatError(statusResponse, statusData));

    if (statusData.status === 'COMPLETED') {
      const resultResponse = await fetch(resultUrl, { headers: falHeaders(apiKey, false) });
      const resultData = await parseJsonResponse(resultResponse);
      writeJson('#responsePreview', resultData);
      if (!resultResponse.ok) throw new Error(formatError(resultResponse, resultData));
      renderResult(resultData);
      setStatus('success', '任務完成，結果已載入。');
      return;
    }

    if (['FAILED', 'ERROR', 'CANCELLED'].includes(statusData.status)) {
      throw new Error(statusData.error || statusData.status);
    }

    await delay(1600);
  }
  throw new Error('輪詢逾時，請稍後用 request_id 查詢結果。');
}

function updateRequestPreview() {
  if (!state.inputSchema || !state.submitPath) {
    writeJson('#requestPreview', { note: '尚未載入 schema' });
    return;
  }
  try {
    const input = collectInput();
    const request = buildSubmitRequest(input, '', true);
    writeJson('#requestPreview', sanitizeRequest(request, input));
  } catch (error) {
    $('#requestPreview').textContent = error.message;
  }
}

function buildSubmitRequest(input, apiKey, preview) {
  const headers = falHeaders(apiKey, preview, true);
  return {
    url: queueUrl(state.submitPath),
    options: { method: 'POST', headers, body: JSON.stringify(input, null, 2) }
  };
}

function falHeaders(apiKey, preview, json = false) {
  const headers = json ? { 'Content-Type': 'application/json' } : {};
  if (apiKey) headers.Authorization = preview ? 'Key ***' : `Key ${apiKey}`;
  return headers;
}

function renderResult(data) {
  const media = extractMedia(data);
  if (media.length) {
    renderPreviewMedia(media, 'fal result');
    return;
  }
  const text = extractText(data);
  if (text) {
    $('#previewArea').innerHTML = `<div class="text-output">${escapeHtml(text)}</div>`;
    return;
  }
  $('#previewArea').innerHTML = `<div class="text-output">${escapeHtml(JSON.stringify(data, null, 2))}</div>`;
}

function renderPreviewMedia(media, label) {
  $('#previewArea').innerHTML = `<div class="media-output" aria-label="${escapeAttr(label)}">${media.map(mediaHtml).join('')}</div>`;
}

function extractText(data) {
  const paths = [['text'], ['output'], ['message'], ['prompt'], ['data', 'text'], ['data', 'output']];
  for (const path of paths) {
    const value = getPath(data, path);
    if (typeof value === 'string' && value.trim()) return value;
  }
  return '';
}

function extractMedia(data) {
  const found = [];
  collectUrls(data).forEach(url => found.push({ type: guessMediaType(url), src: url }));
  collectDataUris(data).forEach(uri => found.push({ type: guessMediaType(uri), src: uri }));
  return dedupe(found, item => item.src).filter(item => item.type !== 'link' || found.length === 1).slice(0, 12);
}

function collectUrls(value, acc = []) {
  if (!value) return acc;
  if (typeof value === 'string' && /^https?:\/\//.test(value)) acc.push(value);
  else if (Array.isArray(value)) value.forEach(item => collectUrls(item, acc));
  else if (typeof value === 'object') Object.values(value).forEach(item => collectUrls(item, acc));
  return acc;
}

function collectDataUris(value, acc = []) {
  if (!value) return acc;
  if (typeof value === 'string' && /^data:(image|video|audio)\//.test(value)) acc.push(value);
  else if (Array.isArray(value)) value.forEach(item => collectDataUris(item, acc));
  else if (typeof value === 'object') Object.values(value).forEach(item => collectDataUris(item, acc));
  return acc;
}

function guessMediaType(url) {
  if (/^data:image\//.test(url) || /\.(png|jpg|jpeg|webp|gif)(\?|$)/i.test(url)) return 'image';
  if (/^data:video\//.test(url) || /\.(mp4|webm|mov)(\?|$)/i.test(url)) return 'video';
  if (/^data:audio\//.test(url) || /\.(mp3|wav|ogg|m4a)(\?|$)/i.test(url)) return 'audio';
  return 'link';
}

function mediaHtml(item) {
  if (item.type === 'image') return `<img src="${escapeAttr(item.src)}" alt="fal result" />`;
  if (item.type === 'video') return `<video controls src="${escapeAttr(item.src)}"></video>`;
  if (item.type === 'audio') return `<audio controls src="${escapeAttr(item.src)}"></audio>`;
  return `<div class="link-list"><a href="${escapeAttr(item.src)}" target="_blank" rel="noopener">${escapeHtml(item.src)}</a></div>`;
}

function openApiUrl(modelId) {
  if (hasProxyUrl()) return `${proxyBaseUrl()}/openapi?endpoint_id=${encodeURIComponent(modelId)}`;
  return `https://fal.ai/api/openapi/queue/openapi.json?endpoint_id=${encodeURIComponent(modelId)}`;
}

function modelsUrl() {
  return hasProxyUrl() ? `${proxyBaseUrl()}/models` : 'https://fal.ai/models';
}

function queueUrl(path) {
  return hasProxyUrl() ? `${proxyBaseUrl()}/queue${path}` : `https://queue.fal.run${path}`;
}

function proxyBaseUrl() {
  return $('#proxyUrlInput').value.trim().replace(/\/+$/, '');
}

function hasProxyUrl() {
  return Boolean(proxyBaseUrl());
}

function updateProxyState() {
  state.serverUrl = hasProxyUrl() ? `${proxyBaseUrl()}/queue` : 'https://queue.fal.run';
}

function titleFromId(id) {
  return id.split('/').slice(-2).join('/').replace(/[-_]/g, ' ').replace(/\b\w/g, char => char.toUpperCase());
}

function titleFromName(name) {
  return name.replace(/_/g, ' ').replace(/\b\w/g, char => char.toUpperCase());
}

function categoryFromId(id) {
  if (/video|veo|kling|pixverse|sora|wan|ltx|seedance|hailuo/i.test(id)) return /image-to-video|reference-to-video/i.test(id) ? 'image-to-video' : 'text-to-video';
  if (/speech|tts|audio|music|voice|sound/i.test(id)) return id.includes('text-to-speech') || /tts|speech|voice/i.test(id) ? 'text-to-speech' : 'text-to-audio';
  if (/3d|meshy|pixal/i.test(id)) return '3d';
  if (/edit|kontext|rembg|background|upscale|image-to-image|tryon/i.test(id)) return 'image-to-image';
  return 'text-to-image';
}

function firstExample(schema) {
  if (Array.isArray(schema.examples) && schema.examples.length) return schema.examples[0];
  return schema.example;
}

function valueToInput(value, kind) {
  if (value === undefined || value === null) return '';
  if (kind === 'json') return typeof value === 'string' ? value : JSON.stringify(value, null, 2);
  return value;
}

function sanitizeRequest(request, input) {
  const headers = { ...request.options.headers };
  if (headers.Authorization) headers.Authorization = 'Key ***';
  return {
    url: request.url,
    method: request.options.method,
    headers,
    body: input
  };
}

async function parseJsonResponse(response) {
  const text = await response.text();
  if (!text) return {};
  try { return JSON.parse(text); }
  catch (_) { return { text }; }
}

function appendStatusLogs(statusData) {
  const lines = [`status: ${statusData.status || 'UNKNOWN'}${statusData.queue_position !== undefined ? `, queue_position: ${statusData.queue_position}` : ''}`];
  const logs = Array.isArray(statusData.logs) ? statusData.logs : [];
  logs.forEach(log => lines.push(log.message || JSON.stringify(log)));
  appendLog(lines.join('\n'));
}

function appendLog(text) {
  const node = $('#queueLog');
  const prefix = node.textContent ? '\n' : '';
  node.textContent += `${prefix}${new Date().toLocaleTimeString()} ${text}`;
  node.scrollTop = node.scrollHeight;
}

function formatError(response, data) {
  const detail = data?.detail || data?.error || data?.message || data?.text || response.statusText;
  return `${response.status} ${response.statusText}: ${typeof detail === 'string' ? detail : JSON.stringify(detail)}`;
}

function switchCodeTab(event) {
  document.querySelectorAll('.code-tab').forEach(btn => btn.classList.toggle('active', btn === event.currentTarget));
  const tab = event.currentTarget.dataset.codeTab;
  $('#requestPreview').classList.toggle('hidden', tab !== 'request');
  $('#responsePreview').classList.toggle('hidden', tab !== 'response');
  $('#queueLog').classList.toggle('hidden', tab !== 'logs');
}

function saveKey() {
  const key = $('#apiKeyInput').value.trim();
  if (!key) {
    setStatus('error', '沒有可儲存的 FAL_KEY。');
    return;
  }
  document.cookie = `endman_fal_key=${encodeURIComponent(key)}; max-age=31536000; path=/; SameSite=Lax`;
  setStatus('success', 'FAL_KEY 已儲存到 cookie。');
}

async function saveProxyUrl() {
  const proxyUrl = proxyBaseUrl();
  if (!proxyUrl) {
    setStatus('error', '沒有可儲存的 proxy URL。');
    return;
  }
  document.cookie = `endman_fal_proxy_url=${encodeURIComponent(proxyUrl)}; max-age=31536000; path=/; SameSite=Lax`;
  updateProxyState();
  setStatus('success', 'Proxy URL 已儲存到 cookie。');
  updateRequestPreview();
  await loadRemoteCatalog();
  if (state.selectedModel?.id) await selectModel(state.selectedModel.id);
}

function clearProxyUrl() {
  document.cookie = 'endman_fal_proxy_url=; max-age=0; path=/; SameSite=Lax';
  $('#proxyUrlInput').value = '';
  updateProxyState();
  setStatus('success', '已清除 proxy URL cookie。');
  updateRequestPreview();
}

function clearKey() {
  document.cookie = 'endman_fal_key=; max-age=0; path=/; SameSite=Lax';
  $('#apiKeyInput').value = '';
  setStatus('success', '已清除 FAL_KEY cookie。');
}

function readCookie(name) {
  const found = document.cookie.split('; ').find(row => row.startsWith(`${name}=`));
  return found ? decodeURIComponent(found.slice(name.length + 1)) : '';
}

function setStatus(kind, message) {
  const badge = $('#statusBadge');
  badge.className = `status-badge ${kind}`;
  badge.textContent = kind.toUpperCase();
  $('#statusMsg').textContent = message;
}

function setCatalogStatus(message) {
  $('#catalogStatus').textContent = message;
}

function getPath(data, path) {
  return path.reduce((value, key) => value && value[key] !== undefined ? value[key] : undefined, data);
}

function dedupe(items, keyFn) {
  const seen = new Set();
  return items.filter(item => {
    const key = keyFn(item);
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function delay(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function writeJson(selector, value) {
  $(selector).textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2);
}

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
}

function escapeAttr(value) {
  return escapeHtml(value).replace(/`/g, '&#96;');
}

function cssEscape(value) {
  if (window.CSS?.escape) return CSS.escape(value);
  return String(value).replace(/(["'\\.#:[\],>+~*^$|= ])/g, '\\$1');
}
