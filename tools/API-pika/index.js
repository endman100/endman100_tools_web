const GROUPS = [
  { id: 'llm', label: 'LLM / 多模態' },
  { id: 'image', label: '生圖 / 圖像' },
  { id: 'video', label: 'AI 影片' },
  { id: 'audio', label: '語音 / STT / TTS' },
];

const commonChatFields = [
  { name: 'model', label: '模型', type: 'text', value: 'gpt-4o-mini' },
  { name: 'systemPrompt', label: 'System prompt', type: 'textarea', value: 'You are a helpful assistant.', full: true },
  { name: 'prompt', label: 'User prompt', type: 'textarea', value: '請用繁體中文簡短介紹這個 API 可以做什麼。', full: true },
  { name: 'temperature', label: 'Temperature', type: 'number', value: '0.7', min: '0', max: '2', step: '0.1' },
  { name: 'maxTokens', label: 'Max tokens', type: 'number', value: '512', min: '1', step: '1' },
];

const imageFields = [
  { name: 'model', label: '模型', type: 'text', value: 'gpt-image-1' },
  { name: 'prompt', label: 'Prompt', type: 'textarea', value: 'A clean product mockup on a workbench, realistic lighting', full: true },
  { name: 'negativePrompt', label: 'Negative prompt', type: 'textarea', value: '', full: true },
  { name: 'size', label: '尺寸 / Aspect ratio', type: 'text', value: '1024x1024' },
  { name: 'count', label: '張數', type: 'number', value: '1', min: '1', max: '4', step: '1' },
  { name: 'steps', label: 'Steps', type: 'number', value: '28', min: '1', step: '1' },
];

const videoFields = [
  { name: 'model', label: '模型', type: 'text', value: 'video-model' },
  { name: 'prompt', label: 'Prompt', type: 'textarea', value: 'A cinematic camera move through a compact creative studio', full: true },
  { name: 'imageUrl', label: '參考圖片 URL', type: 'url', value: '' },
  { name: 'duration', label: '秒數', type: 'number', value: '5', min: '1', step: '1' },
  { name: 'aspectRatio', label: 'Aspect ratio', type: 'text', value: '16:9' },
];

const ttsFields = [
  { name: 'model', label: '模型', type: 'text', value: 'tts-1' },
  { name: 'voice', label: 'Voice', type: 'text', value: 'alloy' },
  { name: 'text', label: '文字', type: 'textarea', value: '這是一段 API 語音合成測試文字。', full: true },
  { name: 'format', label: '輸出格式', type: 'text', value: 'mp3' },
];

const sttFields = [
  { name: 'model', label: '模型', type: 'text', value: 'whisper-1' },
  { name: 'audioFile', label: '音訊檔', type: 'file', accept: 'audio/*' },
  { name: 'audioUrl', label: '音訊 URL / 已上傳檔案 URL', type: 'url', value: '' },
  { name: 'language', label: '語言代碼', type: 'text', value: 'zh' },
];

const services = [
  video('pika', 'Pika API', 'https://api.pika.art/generate', 'pika-1.5', 'pikaVideo')
];

function llm(id, name, endpoint, model, build) {
  return service({ id, name, group: 'llm', task: 'chat', endpoint, model, build, summary: '文字、聊天或多模態模型的前端 API 呼叫。', fields: withModel(commonChatFields, model) });
}

function image(id, name, endpoint, model, build) {
  return service({ id, name, group: 'image', task: 'image', endpoint, model, build, summary: '文字生圖、圖像編輯或圖片生成任務。', fields: withModel(imageFields, model) });
}

function video(id, name, endpoint, model, build, headers = {}) {
  return service({ id, name, group: 'video', task: 'video', endpoint, model, build, summary: '文字或圖片轉影片，通常會回傳任務 ID 或影片 URL。', fields: withModel(videoFields, model), headers });
}

function tts(id, name, endpoint, model, build) {
  return service({ id, name, group: 'audio', task: 'tts', endpoint, model, build, summary: '文字轉語音 API，支援音訊預覽或回傳任務 JSON。', fields: withModel(ttsFields, model), responseType: 'auto' });
}

function stt(id, name, endpoint, model, build) {
  return service({ id, name, group: 'audio', task: 'stt', endpoint, model, build, summary: '語音轉文字 API，支援檔案或音訊 URL 輸入。', fields: withModel(sttFields, model) });
}

function service(config) {
  return {
    method: 'POST', auth: 'bearer', keyLabel: 'API key', credentials: [{ name: 'apiKey', label: config.keyLabel || 'API key', type: 'password' }],
    headers: {}, responseType: 'json', ...config,
  };
}

function withModel(fields, model) {
  return fields.map(field => field.name === 'model' ? { ...field, value: model } : { ...field });
}

function bedrock() {
  return service({
    id: 'aws-bedrock', name: 'AWS Bedrock', group: 'llm', task: 'chat', build: 'bedrockInvoke', auth: 'aws-sigv4', method: 'POST',
    endpoint: 'https://bedrock-runtime.{region}.amazonaws.com/model/{model}/invoke',
    summary: '使用 AWS SigV4 在前端簽署 Bedrock Runtime InvokeModel 請求。瀏覽器 CORS 與 IAM 權限需由你的 AWS 帳號允許。',
    credentials: [
      { name: 'accessKeyId', label: 'AWS access key ID', type: 'password' },
      { name: 'secretAccessKey', label: 'AWS secret access key', type: 'password' },
      { name: 'sessionToken', label: 'Session token（選填）', type: 'password' },
    ],
    fields: [
      { name: 'region', label: 'Region', type: 'text', value: 'us-east-1' },
      { name: 'model', label: 'Model ID', type: 'text', value: 'anthropic.claude-3-5-sonnet-20240620-v1:0' },
      { name: 'prompt', label: 'User prompt', type: 'textarea', value: '請用繁體中文簡短介紹 AWS Bedrock。', full: true },
      { name: 'maxTokens', label: 'Max tokens', type: 'number', value: '512', min: '1' },
    ],
  });
}

function azureFoundry() {
  return service({
    id: 'azure-ai-foundry', name: 'Azure AI Foundry', group: 'llm', task: 'chat', build: 'openaiChat', auth: 'api-key',
    endpoint: '{resourceEndpoint}/openai/deployments/{deployment}/chat/completions?api-version={apiVersion}',
    summary: 'Azure OpenAI / AI Foundry 相容聊天端點，需填 resource endpoint、deployment 與 api-version。',
    fields: [
      { name: 'resourceEndpoint', label: 'Azure resource endpoint', type: 'url', value: 'https://YOUR-RESOURCE.openai.azure.com' },
      { name: 'deployment', label: 'Deployment name', type: 'text', value: 'gpt-4o-mini' },
      { name: 'apiVersion', label: 'API version', type: 'text', value: '2024-10-21' },
      ...commonChatFields.filter(f => f.name !== 'model'),
    ],
  });
}

function vertexAi() {
  return service({
    id: 'vertex-ai', name: 'Google Cloud Vertex AI', group: 'llm', task: 'chat', build: 'vertexGemini', auth: 'bearer', keyLabel: 'OAuth access token',
    endpoint: 'https://{location}-aiplatform.googleapis.com/v1/projects/{project}/locations/{location}/publishers/google/models/{model}:generateContent',
    summary: 'Vertex AI 通常需要 OAuth access token。此頁不提供後端代換流程，只以前端 Bearer token 呼叫。',
    fields: [
      { name: 'project', label: 'Project ID', type: 'text', value: 'your-project-id' },
      { name: 'location', label: 'Location', type: 'text', value: 'us-central1' },
      ...withModel(commonChatFields, 'gemini-1.5-flash'),
    ],
  });
}

function openAiTts() {
  return service({
    id: 'openai-audio-tts', name: 'OpenAI Audio TTS API', group: 'audio', task: 'tts', build: 'openaiTts', endpoint: 'https://api.openai.com/v1/audio/speech',
    summary: 'OpenAI 文字轉語音功能工具，獨立於語音轉文字。', responseType: 'blob', fields: ttsFields,
  });
}

function openAiStt() {
  return service({
    id: 'openai-audio-stt', name: 'OpenAI Audio STT API', group: 'audio', task: 'stt', build: 'openaiStt', endpoint: 'https://api.openai.com/v1/audio/transcriptions',
    summary: 'OpenAI 語音轉文字功能工具，獨立於 TTS。', responseType: 'json', fields: sttFields,
  });
}

const state = { group: 'llm', selected: services[0], lastRequest: null, lastResponse: null };
const $ = selector => document.querySelector(selector);
const categoryTabs = $('#categoryTabs');
const serviceList = $('#serviceList');
const searchInput = $('#serviceSearch');
const apiForm = $('#apiForm');
const credentialFields = $('#credentialFields');
const dynamicFields = $('#dynamicFields');
const endpointInput = $('#endpointInput');

init();

function init() {
  const requestedTool = document.body.dataset.toolId || new URLSearchParams(location.search).get('tool');
  const selected = services.find(item => item.id === requestedTool) || services[0];
  state.group = selected.group;
  renderTabs();
  renderServiceList();
  selectService(selected.id, true);
  searchInput.addEventListener('input', renderServiceList);
  $('#saveKeyBtn').addEventListener('click', saveCredentials);
  $('#clearKeyBtn').addEventListener('click', clearCredentials);
  apiForm.addEventListener('submit', callApi);
  document.querySelectorAll('.code-tab').forEach(btn => btn.addEventListener('click', switchCodeTab));
  window.__apiToolsNoKeySmokeTest = runNoKeySmokeTest;
}

function renderTabs() {
  categoryTabs.innerHTML = GROUPS.map(group => `<button type="button" class="category-tab${group.id === state.group ? ' active' : ''}" data-group="${group.id}">${group.label}</button>`).join('');
  categoryTabs.querySelectorAll('button').forEach(btn => btn.addEventListener('click', () => {
    state.group = btn.dataset.group;
    renderTabs();
    renderServiceList();
    const first = filteredServices()[0];
    if (first) selectService(first.id);
  }));
}

function filteredServices() {
  const keyword = searchInput.value.trim().toLowerCase();
  return services.filter(item => item.group === state.group && (!keyword || `${item.name} ${item.summary}`.toLowerCase().includes(keyword)));
}

function renderServiceList() {
  const items = filteredServices();
  serviceList.innerHTML = items.map(item => `<button type="button" class="service-item${state.selected?.id === item.id ? ' active' : ''}" data-id="${item.id}"><strong>${item.name}</strong><span>${GROUPS.find(g => g.id === item.group).label}</span></button>`).join('') || '<div class="empty-state"><p>沒有符合的服務。</p></div>';
  serviceList.querySelectorAll('button').forEach(btn => btn.addEventListener('click', () => selectService(btn.dataset.id)));
}

function selectService(id, replaceUrl = false) {
  state.selected = services.find(item => item.id === id) || services[0];
  state.group = state.selected.group;
  const isPhysicalToolPage = Boolean(document.body.dataset.toolId);
  const nextUrl = `${location.pathname}?tool=${encodeURIComponent(state.selected.id)}`;
  if (!isPhysicalToolPage && location.search !== `?tool=${encodeURIComponent(state.selected.id)}`) {
    history[replaceUrl ? 'replaceState' : 'pushState']({}, '', nextUrl);
  }
  renderServiceList();
  $('#serviceTag').textContent = GROUPS.find(g => g.id === state.selected.group).label;
  $('#serviceName').textContent = state.selected.name;
  $('#serviceSummary').textContent = state.selected.summary;
  $('#methodPill').textContent = state.selected.method;
  endpointInput.value = state.selected.endpoint;
  renderCredentials();
  renderDynamicFields();
  setStatus('info', `已選擇 ${state.selected.name} tool。API key 會以 cookie 儲存在這個瀏覽器。`);
  updateRequestPreview();
}

function renderCredentials() {
  const stored = readCredentialCookie(state.selected.id);
  credentialFields.innerHTML = state.selected.credentials.map(field => fieldHtml(field, stored[field.name] || '')).join('');
  credentialFields.querySelectorAll('input').forEach(input => input.addEventListener('input', updateRequestPreview));
}

function renderDynamicFields() {
  const fields = [...(state.selected.fields || []), ...(state.selected.extraFields || [])];
  dynamicFields.innerHTML = fields.map(field => fieldHtml(field, field.value || '')).join('');
  dynamicFields.querySelectorAll('input,select,textarea').forEach(input => input.addEventListener('input', updateRequestPreview));
  endpointInput.addEventListener('input', updateRequestPreview);
}

function fieldHtml(field, value) {
  const full = field.full || field.type === 'textarea' || field.type === 'file' ? ' full' : '';
  const help = field.help ? `<div class="form-help">${escapeHtml(field.help)}</div>` : '';
  if (field.type === 'textarea') {
    return `<div class="form-group${full}"><label for="${field.name}">${field.label}</label><textarea id="${field.name}" name="${field.name}">${escapeHtml(value)}</textarea>${help}</div>`;
  }
  if (field.type === 'select') {
    const options = (field.options || []).map(option => `<option value="${escapeHtml(option.value)}"${option.value === value ? ' selected' : ''}>${escapeHtml(option.label)}</option>`).join('');
    return `<div class="form-group${full}"><label for="${field.name}">${field.label}</label><select id="${field.name}" name="${field.name}">${options}</select>${help}</div>`;
  }
  const attrs = [field.min && `min="${field.min}"`, field.max && `max="${field.max}"`, field.step && `step="${field.step}"`, field.accept && `accept="${field.accept}"`].filter(Boolean).join(' ');
  return `<div class="form-group${full}"><label for="${field.name}">${field.label}</label><input id="${field.name}" name="${field.name}" type="${field.type || 'text'}" value="${field.type === 'file' ? '' : escapeHtml(value)}" ${attrs}/>${help}</div>`;
}

function saveCredentials() {
  const credentials = collectCredentials();
  if (!Object.values(credentials).some(Boolean)) {
    setStatus('error', '沒有可儲存的 API key / 認證欄位。');
    return;
  }
  document.cookie = `${cookieName(state.selected.id)}=${encodeURIComponent(JSON.stringify(credentials))}; max-age=31536000; path=/; SameSite=Lax`;
  setStatus('success', `${state.selected.name} tool 的 API key 已儲存到 cookie。`);
}

function clearCredentials() {
  document.cookie = `${cookieName(state.selected.id)}=; max-age=0; path=/; SameSite=Lax`;
  renderCredentials();
  setStatus('success', `已清除 ${state.selected.name} tool 的 API key cookie。`);
}

function readCredentialCookie(id) {
  const name = `${cookieName(id)}=`;
  const found = document.cookie.split('; ').find(row => row.startsWith(name));
  if (!found) return {};
  try { return JSON.parse(decodeURIComponent(found.slice(name.length))) || {}; } catch (_) { return {}; }
}

function cookieName(id) { return `endman_api_tool_${id}`; }

function collectCredentials(mask = false) {
  const result = {};
  state.selected.credentials.forEach(field => {
    const value = credentialFields.querySelector(`[name="${field.name}"]`)?.value || '';
    result[field.name] = mask && value ? maskSecret(value) : value;
  });
  return result;
}

function collectValues() {
  const values = { endpoint: endpointInput.value };
  dynamicFields.querySelectorAll('input,select,textarea').forEach(input => {
    values[input.name] = input.type === 'file' ? input.files[0] : input.value;
  });
  return values;
}

async function callApi(event) {
  event.preventDefault();
  const credentials = collectCredentials();
  if (!Object.values(credentials).filter(Boolean).length) {
    setStatus('error', '請先填入這個工具的 API key / 認證欄位。');
    return;
  }

  setStatus('loading', `正在呼叫 ${state.selected.name}...`);
  $('#sendBtn').disabled = true;
  try {
    const request = await buildRequest(state.selected, collectValues(), credentials, false);
    state.lastRequest = sanitizeRequest(request);
    writeJson('#requestPreview', state.lastRequest);

    const response = await fetch(request.url, request.options);
    const parsed = await parseResponse(response, state.selected.responseType);
    state.lastResponse = { ok: response.ok, status: response.status, statusText: response.statusText, headers: Object.fromEntries(response.headers.entries()), body: parsed.preview };
    writeJson('#responsePreview', state.lastResponse);
    renderPreview(state.selected, parsed);

    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    setStatus('success', `${state.selected.name} 回應完成。`);
  } catch (error) {
    setStatus('error', `${state.selected.name} 呼叫失敗：${error.message}`);
  } finally {
    $('#sendBtn').disabled = false;
  }
}

async function updateRequestPreview() {
  if (!state.selected) return;
  try {
    const request = await buildRequest(state.selected, collectValues(), collectCredentials(true), true);
    state.lastRequest = sanitizeRequest(request);
    writeJson('#requestPreview', state.lastRequest);
  } catch (error) {
    $('#requestPreview').textContent = error.message;
  }
}

async function buildRequest(service, values, credentials, preview) {
  let url = interpolate(values.endpoint || service.endpoint, values);
  const headers = { ...(service.headers || {}) };
  let body;

  if (!['formData', 'raw'].includes(service.build) && !hasHeader(headers, 'Content-Type')) headers['Content-Type'] = 'application/json';
  applyAuth(service, url, headers, credentials);
  url = applyQueryAuth(service, url, credentials);

  body = await buildBody(service, values, preview);
  if (body instanceof FormData) delete headers['Content-Type'];
  if (body instanceof Blob) headers['Content-Type'] = body.type || 'application/octet-stream';

  const options = { method: service.method, headers, body };
  if (service.auth === 'aws-sigv4' && !preview && credentials.accessKeyId && credentials.secretAccessKey) {
    options.headers = await signAwsRequest(url, options, credentials, values.region || 'us-east-1', 'bedrock');
  }
  return { url, options };
}

function applyAuth(service, url, headers, credentials) {
  const key = credentials.apiKey || '';
  if (!key && service.auth !== 'playht') return;
  if (service.auth === 'bearer') headers.Authorization = `Bearer ${key}`;
  if (service.auth === 'token') headers.Authorization = `Token ${key}`;
  if (service.auth === 'fal-key') headers.Authorization = `Key ${key}`;
  if (service.auth === 'x-api-key') headers['x-api-key'] = key;
  if (service.auth === 'x-api-key-title') headers['X-Api-Key'] = key;
  if (service.auth === 'api-key') headers['api-key'] = key;
  if (service.auth === 'api-key-title') headers['Api-Key'] = key;
  if (service.auth === 'x-key') headers['x-key'] = key;
  if (service.auth === 'xi-api-key') headers['xi-api-key'] = key;
  if (service.auth === 'x-hume-api-key') headers['X-Hume-Api-Key'] = key;
  if (service.auth === 'plain-authorization') headers.Authorization = key;
  if (service.auth === 'ocp-key') headers['Ocp-Apim-Subscription-Key'] = key;
  if (service.auth === 'playht') {
    if (credentials.apiKey) headers.Authorization = `Bearer ${credentials.apiKey}`;
    if (credentials.userId) headers['X-USER-ID'] = credentials.userId;
  }
}

function applyQueryAuth(service, url, credentials) {
  if (service.auth !== 'query-key') return url;
  if (!credentials.apiKey) return url;
  const parsed = new URL(url);
  parsed.searchParams.set('key', credentials.apiKey || '');
  return parsed.toString();
}

function hasHeader(headers, target) {
  return Object.keys(headers).some(key => key.toLowerCase() === target.toLowerCase());
}

async function buildBody(service, values, preview) {
  const number = (name, fallback) => Number(values[name] || fallback);
  const prompt = values.prompt || values.text || '';
  switch (service.build) {
    case 'openaiChat':
      return json({ model: values.model || values.deployment, messages: messages(values), temperature: number('temperature', 0.7), max_tokens: number('maxTokens', 512) });
    case 'anthropic':
      return json({ model: values.model, system: values.systemPrompt || '', messages: [{ role: 'user', content: prompt }], temperature: number('temperature', 0.7), max_tokens: number('maxTokens', 512) });
    case 'gemini':
    case 'vertexGemini': {
      const body = {
        contents: [{ role: 'user', parts: [{ text: prompt }] }],
        generationConfig: { temperature: number('temperature', 0.7), maxOutputTokens: number('maxTokens', 512) },
      };
      if (values.systemPrompt) body.systemInstruction = { parts: [{ text: values.systemPrompt }] };
      return json(body);
    }
    case 'cohere':
      return json({ stream: false, model: values.model, messages: [{ role: 'user', content: prompt }], temperature: number('temperature', 0.7), max_tokens: number('maxTokens', 512) });
    case 'minimaxChat':
      return json({ model: values.model, messages: messages(values), temperature: number('temperature', 0.7), max_tokens: number('maxTokens', 512) });
    case 'bedrockInvoke':
      return json({ anthropic_version: 'bedrock-2023-05-31', max_tokens: number('maxTokens', 512), messages: [{ role: 'user', content: [{ type: 'text', text: prompt }] }] });
    case 'openaiImage':
      return json({ model: values.model, prompt, size: values.size, n: number('count', 1) });
    case 'googleImagen':
      return json({ instances: [{ prompt }], parameters: { sampleCount: number('count', 1), aspectRatio: values.size } });
    case 'geminiImage':
      return json({
        contents: [{ role: 'user', parts: [{ text: prompt }] }],
        generationConfig: {
          responseModalities: ['TEXT', 'IMAGE'],
          imageConfig: { aspectRatio: ratioForGoogleImage(values.size), imageSize: sizeForGoogleImage(values.size) },
        },
      });
    case 'stabilityCore': {
      const form = new FormData();
      form.append('prompt', prompt);
      if (values.negativePrompt) form.append('negative_prompt', values.negativePrompt);
      form.append('aspect_ratio', ratioForStability(values.size));
      form.append('output_format', 'png');
      return form;
    }
    case 'bflFlux':
      return json({ prompt, width: 1024, height: 1024, steps: number('steps', 28), prompt_upsampling: false });
    case 'falGeneric':
      return json({ prompt, image_url: values.imageUrl || undefined, duration: values.duration || undefined, aspect_ratio: values.aspectRatio || values.size });
    case 'replicatePrediction':
      return json({ version: values.version || undefined, input: { prompt, image: values.imageUrl || undefined, num_outputs: number('count', 1), aspect_ratio: values.aspectRatio || values.size } });
    case 'ideogram':
      return json({ image_request: { prompt, model: values.model, aspect_ratio: normalizeAspect(values.size), num_images: number('count', 1) } });
    case 'lumaGeneration':
      return json({ prompt, model: values.model, aspect_ratio: values.aspectRatio || values.size, duration: `${values.duration || 5}s` });
    case 'runwareImage':
      return json([{ taskType: 'imageInference', taskUUID: crypto.randomUUID(), positivePrompt: prompt, negativePrompt: values.negativePrompt || undefined, model: values.model, width: 1024, height: 1024, numberResults: number('count', 1), steps: number('steps', 28) }]);
    case 'segmindGeneric':
      return json({ prompt, negative_prompt: values.negativePrompt || '', samples: number('count', 1), scheduler: 'dpmpp_2m', num_inference_steps: number('steps', 28), guidance_scale: 7.5, img_width: 1024, img_height: 1024 });
    case 'runwayVideo':
      return json({ model: values.model, prompt_text: prompt, prompt_image: values.imageUrl || undefined, ratio: values.aspectRatio, duration: number('duration', 5) });
    case 'klingVideo':
      return json({ model_name: values.model, prompt, image: values.imageUrl || undefined, duration: String(values.duration || 5), aspect_ratio: values.aspectRatio });
    case 'pikaVideo':
      return json({ prompt, model: values.model, image_url: values.imageUrl || undefined, duration: number('duration', 5), aspect_ratio: values.aspectRatio });
    case 'heygenVideo':
      return json({ video_inputs: [{ character: { type: 'avatar', avatar_id: values.model }, voice: { type: 'text', input_text: prompt } }], dimension: { width: 1280, height: 720 } });
    case 'hedraVideo':
      return json({ prompt, model: values.model, image_url: values.imageUrl || undefined, duration: number('duration', 5), aspect_ratio: values.aspectRatio });
    case 'minimaxVideo':
      return json({ model: values.model, prompt, first_frame_image: values.imageUrl || undefined, duration: number('duration', 5) });
    case 'openaiTts':
      return json({ model: values.model || 'tts-1', voice: values.voice || 'alloy', input: values.text || prompt, response_format: values.format || 'mp3' });
    case 'openaiStt': {
      if (preview) return '[FormData: model, file, language]';
      const form = new FormData();
      form.append('model', values.model || 'whisper-1');
      form.append('language', values.language || 'zh');
      if (values.audioFile) form.append('file', values.audioFile);
      return form;
    }
    case 'elevenTts':
      return json({ text: values.text || prompt, model_id: values.model, voice_settings: { stability: 0.5, similarity_boost: 0.75 } });
    case 'deepgramStt':
      if (preview) return '[Binary audio file body]';
      return values.audioFile || new Blob([], { type: 'audio/wav' });
    case 'assemblyTranscript':
      return json({ audio_url: values.audioUrl, language_code: values.language || 'zh' });
    case 'humeTts':
      return json({ utterances: [{ text: values.text || prompt, voice: { name: values.voice || 'ITO' } }], format: { type: values.format || 'mp3' } });
    case 'cartesiaTts':
      return json({ model_id: values.model, transcript: values.text || prompt, voice: { mode: 'id', id: values.voice || 'a0e99841-438c-4a64-b679-ae501e7d6091' }, output_format: { container: values.format || 'mp3', encoding: 'mp3', sample_rate: 44100 } });
    case 'playhtTts':
      return json({ text: values.text || prompt, voice: values.voice || 's3://voice-cloning-zero-shot/default-male', output_format: values.format || 'mp3', quality: 'medium' });
    case 'googleStt': {
      const content = preview || !values.audioFile ? 'BASE64_AUDIO_CONTENT' : await fileToBase64(values.audioFile);
      return json({ config: { languageCode: values.language || 'zh-TW', model: values.model || 'latest_long' }, audio: { content } });
    }
    case 'googleTts':
      return json({ input: { text: values.text || prompt }, voice: { languageCode: values.languageCode || 'zh-TW', name: values.voice }, audioConfig: { audioEncoding: (values.format || 'mp3').toUpperCase() === 'MP3' ? 'MP3' : 'LINEAR16' } });
    case 'azureSpeechTts':
      return `<speak version="1.0" xml:lang="zh-TW"><voice name="${escapeXml(values.voice || 'zh-TW-HsiaoChenNeural')}">${escapeXml(values.text || prompt)}</voice></speak>`;
    default:
      return json({ prompt });
  }
}

function messages(values) {
  const list = [];
  if (values.systemPrompt) list.push({ role: 'system', content: values.systemPrompt });
  list.push({ role: 'user', content: values.prompt || values.text || '' });
  return list;
}

function json(value) { return JSON.stringify(value); }
function interpolate(template, values) {
  return template.replace(/\{([a-zA-Z0-9_]+)\}/g, (_, key) => encodePathValue(values[key] || ''));
}

function encodePathValue(value) {
  return String(value).split('/').map(part => encodeURIComponent(part)).join('/');
}
function ratioForStability(size) { return size && size.includes(':') ? size : '1:1'; }
function normalizeAspect(size) { return (size || '1:1').replace(':', 'x'); }
function ratioForGoogleImage(size) {
  if (!size) return '1:1';
  if (size.includes(':')) return size;
  const match = String(size).match(/^(\d+)x(\d+)$/i);
  if (!match) return '1:1';
  const width = Number(match[1]);
  const height = Number(match[2]);
  if (!width || !height) return '1:1';
  const divisor = gcd(width, height);
  return (width / divisor) + ':' + (height / divisor);
}
function sizeForGoogleImage(size) {
  if (/^512$/i.test(String(size))) return '512';
  if (/^(2k|4k)$/i.test(String(size))) return String(size).toUpperCase();
  return '1K';
}
function gcd(a, b) { return b ? gcd(b, a % b) : Math.abs(a); }

async function parseResponse(response, responseType) {
  const contentType = response.headers.get('content-type') || '';
  if (responseType === 'blob' || /audio|image|video/.test(contentType)) {
    const blob = await response.blob();
    return { kind: 'blob', blob, contentType, preview: `[Blob ${blob.type || contentType || 'application/octet-stream'} ${blob.size} bytes]` };
  }
  const text = await response.text();
  try { return { kind: 'json', json: JSON.parse(text), preview: JSON.parse(text) }; }
  catch (_) { return { kind: 'text', text, preview: text }; }
}

function renderPreview(service, parsed) {
  const area = $('#previewArea');
  if (parsed.kind === 'blob') {
    const url = URL.createObjectURL(parsed.blob);
    if (parsed.contentType.includes('image')) area.innerHTML = `<div class="media-output"><img src="${url}" alt="API image result" /></div>`;
    else if (parsed.contentType.includes('video')) area.innerHTML = `<div class="media-output"><video controls src="${url}"></video></div>`;
    else area.innerHTML = `<div class="media-output"><audio controls src="${url}"></audio><a class="btn btn-ghost" href="${url}" download="api-result">下載結果</a></div>`;
    return;
  }
  if (parsed.kind === 'text') {
    area.innerHTML = `<div class="text-output">${escapeHtml(parsed.text)}</div>`;
    return;
  }

  const data = parsed.json;
  const text = extractText(data);
  const media = extractMedia(data);
  if (media.length) {
    area.innerHTML = `<div class="media-output">${media.map(item => mediaHtml(item)).join('')}</div>`;
    return;
  }
  if (text) {
    area.innerHTML = `<div class="text-output">${escapeHtml(text)}</div>`;
    return;
  }
  area.innerHTML = `<div class="text-output">${escapeHtml(JSON.stringify(data, null, 2))}</div>`;
}

function extractText(data) {
  const paths = [
    ['choices', 0, 'message', 'content'], ['content', 0, 'text'], ['message', 'content', 0, 'text'],
    ['candidates', 0, 'content', 'parts', 0, 'text'], ['output', 0, 'content', 0, 'text'],
    ['text'], ['transcript'], ['results', 'channels', 0, 'alternatives', 0, 'transcript'], ['audioContent'],
  ];
  for (const path of paths) {
    const value = getPath(data, path);
    if (typeof value === 'string' && value.trim()) return value;
  }
  return '';
}

function extractMedia(data) {
  const found = [];
  const b64Paths = [['data',0,'b64_json'], ['artifacts',0,'base64'], ['image','base64'], ['audioContent']];
  b64Paths.forEach(path => {
    const value = getPath(data, path);
    if (typeof value === 'string' && value.length > 100) found.push({ type: path.join('.').includes('audio') ? 'audio' : 'image', src: value.startsWith('data:') ? value : `data:${path.join('.').includes('audio') ? 'audio/mpeg' : 'image/png'};base64,${value}` });
  });
  collectUrls(data).forEach(url => found.push({ type: guessMediaType(url), src: url }));
  return dedupe(found, item => item.src).slice(0, 8);
}

function collectUrls(value, acc = []) {
  if (!value) return acc;
  if (typeof value === 'string' && /^https?:\/\//.test(value)) acc.push(value);
  else if (Array.isArray(value)) value.forEach(item => collectUrls(item, acc));
  else if (typeof value === 'object') Object.values(value).forEach(item => collectUrls(item, acc));
  return acc;
}

function guessMediaType(url) {
  if (/\.(png|jpg|jpeg|webp|gif)(\?|$)/i.test(url)) return 'image';
  if (/\.(mp4|webm|mov)(\?|$)/i.test(url)) return 'video';
  if (/\.(mp3|wav|ogg|m4a)(\?|$)/i.test(url)) return 'audio';
  return 'link';
}

function mediaHtml(item) {
  if (item.type === 'image') return `<img src="${escapeAttr(item.src)}" alt="API result" />`;
  if (item.type === 'video') return `<video controls src="${escapeAttr(item.src)}"></video>`;
  if (item.type === 'audio') return `<audio controls src="${escapeAttr(item.src)}"></audio>`;
  return `<div class="link-list"><a href="${escapeAttr(item.src)}" target="_blank" rel="noopener">${escapeHtml(item.src)}</a></div>`;
}

function sanitizeRequest(request) {
  const headers = { ...(request.options.headers || {}) };
  Object.keys(headers).forEach(key => {
    if (/authorization|api-key|x-key|subscription|token/i.test(key)) headers[key] = maskSecret(headers[key]);
  });
  let body = request.options.body;
  if (body instanceof FormData) body = '[FormData]';
  return { url: request.url.replace(/key=([^&]+)/, 'key=***'), method: request.options.method, headers, body: tryParseJson(body) };
}

function tryParseJson(body) { try { return typeof body === 'string' && body.trim().startsWith('{') || typeof body === 'string' && body.trim().startsWith('[') ? JSON.parse(body) : body; } catch (_) { return body; } }
function writeJson(selector, value) { $(selector).textContent = typeof value === 'string' ? value : JSON.stringify(value, null, 2); }
function switchCodeTab(event) { document.querySelectorAll('.code-tab').forEach(btn => btn.classList.remove('active')); event.currentTarget.classList.add('active'); $('#requestPreview').classList.toggle('hidden', event.currentTarget.dataset.codeTab !== 'request'); $('#responsePreview').classList.toggle('hidden', event.currentTarget.dataset.codeTab !== 'response'); }
function setStatus(type, message) { $('#statusBadge').className = `status-badge ${type}`; $('#statusBadge').textContent = type.toUpperCase(); $('#statusMsg').textContent = message; }
function getPath(obj, path) { return path.reduce((current, key) => current == null ? undefined : current[key], obj); }
function dedupe(items, keyFn) { const seen = new Set(); return items.filter(item => { const key = keyFn(item); if (seen.has(key)) return false; seen.add(key); return true; }); }
function maskSecret(value) { return value ? `${String(value).slice(0, 6)}...${String(value).slice(-4)}` : ''; }
function escapeHtml(str) { return String(str ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'); }
function escapeAttr(str) { return escapeHtml(str).replace(/'/g, '&#39;'); }
function escapeXml(str) { return String(str ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&apos;'); }
function fileToBase64(file) { return new Promise((resolve, reject) => { const reader = new FileReader(); reader.onload = () => resolve(String(reader.result).split(',')[1] || ''); reader.onerror = reject; reader.readAsDataURL(file); }); }

async function runNoKeySmokeTest(options = {}) {
  const timeoutMs = options.timeoutMs || 4500;
  const concurrency = options.concurrency || 4;
  const queue = services.map(item => ({ service: item, values: defaultValues(item) }));
  const results = [];

  async function worker() {
    while (queue.length) {
      const next = queue.shift();
      if (!next) return;
      results.push(await smokeOneTool(next.service, next.values, timeoutMs));
    }
  }

  await Promise.all(Array.from({ length: Math.min(concurrency, queue.length) }, worker));
  const summary = results.reduce((acc, item) => {
    acc[item.result] = (acc[item.result] || 0) + 1;
    return acc;
  }, {});
  return { total: services.length, summary, results: results.sort((a, b) => a.name.localeCompare(b.name)) };
}

function defaultValues(service) {
  const values = { endpoint: service.endpoint };
  [...(service.fields || []), ...(service.extraFields || [])].forEach(field => {
    if (field.type !== 'file') values[field.name] = field.value || '';
  });
  return values;
}

async function smokeOneTool(service, values, timeoutMs) {
  try {
    const request = await buildRequest(service, values, {}, false);
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(request.url, { ...request.options, signal: controller.signal });
      clearTimeout(timer);
      return { id: service.id, name: service.name, result: 'http-response', status: response.status, ok: response.ok, request: sanitizeRequest(request) };
    } catch (error) {
      clearTimeout(timer);
      const aborted = error.name === 'AbortError';
      return { id: service.id, name: service.name, result: aborted ? 'timeout' : 'fetch-blocked', error: error.message, request: sanitizeRequest(request) };
    }
  } catch (error) {
    return { id: service.id, name: service.name, result: 'build-error', error: error.message };
  }
}

async function signAwsRequest(url, options, credentials, region, awsService) {
  const parsed = new URL(url);
  const method = options.method || 'POST';
  const body = options.body || '';
  const now = new Date();
  const amzDate = now.toISOString().replace(/[:-]|\.\d{3}/g, '');
  const dateStamp = amzDate.slice(0, 8);
  const payloadHash = await sha256Hex(body);
  const headers = lowercaseHeaders(options.headers || {});
  headers.host = parsed.host;
  headers['x-amz-date'] = amzDate;
  headers['x-amz-content-sha256'] = payloadHash;
  if (credentials.sessionToken) headers['x-amz-security-token'] = credentials.sessionToken;

  const canonicalHeaders = Object.keys(headers).sort().map(key => `${key}:${String(headers[key]).trim()}\n`).join('');
  const signedHeaders = Object.keys(headers).sort().join(';');
  const canonicalQuery = [...parsed.searchParams.entries()].sort(([a], [b]) => a.localeCompare(b)).map(([key, value]) => `${encodeURIComponent(key)}=${encodeURIComponent(value)}`).join('&');
  const canonicalRequest = [method, parsed.pathname, canonicalQuery, canonicalHeaders, signedHeaders, payloadHash].join('\n');
  const credentialScope = `${dateStamp}/${region}/${awsService}/aws4_request`;
  const stringToSign = ['AWS4-HMAC-SHA256', amzDate, credentialScope, await sha256Hex(canonicalRequest)].join('\n');
  const signingKey = await getSignatureKey(credentials.secretAccessKey, dateStamp, region, awsService);
  const signature = await hmacHex(signingKey, stringToSign);
  headers.authorization = `AWS4-HMAC-SHA256 Credential=${credentials.accessKeyId}/${credentialScope}, SignedHeaders=${signedHeaders}, Signature=${signature}`;
  return headers;
}

function lowercaseHeaders(headers) { return Object.fromEntries(Object.entries(headers).map(([key, value]) => [key.toLowerCase(), value])); }
async function sha256Hex(message) { const data = typeof message === 'string' ? new TextEncoder().encode(message) : await message.arrayBuffer?.() || new TextEncoder().encode(String(message)); return hex(await crypto.subtle.digest('SHA-256', data)); }
async function hmacRaw(key, message) { const cryptoKey = await crypto.subtle.importKey('raw', key, { name: 'HMAC', hash: 'SHA-256' }, false, ['sign']); return new Uint8Array(await crypto.subtle.sign('HMAC', cryptoKey, new TextEncoder().encode(message))); }
async function hmacHex(key, message) { return hex(await hmacRaw(key, message)); }
async function getSignatureKey(secret, dateStamp, region, serviceName) { let key = new TextEncoder().encode(`AWS4${secret}`); key = await hmacRaw(key, dateStamp); key = await hmacRaw(key, region); key = await hmacRaw(key, serviceName); return hmacRaw(key, 'aws4_request'); }
function hex(buffer) { return [...new Uint8Array(buffer)].map(byte => byte.toString(16).padStart(2, '0')).join(''); }
