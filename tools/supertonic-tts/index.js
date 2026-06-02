/**
 * Supertonic TTS — Pure Frontend WebGPU Tool
 * Models loaded from HuggingFace CDN, cached in IndexedDB.
 * Engine ported from https://github.com/supertone-inc/supertonic/tree/main/web
 * License: MIT (code) / OpenRAIL-M (model weights)
 */
(function () {
  'use strict';

  // ─── Configuration ────────────────────────────────────────────────────────
  const HF_BASE = 'https://huggingface.co/Supertone/supertonic-3/resolve/main';
  const ONNX_BASE = `${HF_BASE}/onnx`;
  const VS_BASE   = `${HF_BASE}/voice_styles`;

  const DB_NAME    = 'supertonic-v3-cache';
  const STORE_NAME = 'models';
  const CUSTOM_VOICES_KEY = 'supertonic-v3-custom-voices';
  const WEBGPU_ERROR =
    'This tool is WebGPU-only. Use a WebGPU-capable Chrome/Edge browser and make sure GPU acceleration is enabled.';

  const MODEL_DEFS = [
    { id: 'dp', name: 'Duration Predictor', url: `${ONNX_BASE}/duration_predictor.onnx`, size: '3.7 MB'  },
    { id: 'te', name: 'Text Encoder',        url: `${ONNX_BASE}/text_encoder.onnx`,        size: '36 MB'   },
    { id: 've', name: 'Vector Estimator',    url: `${ONNX_BASE}/vector_estimator.onnx`,    size: '257 MB'  },
    { id: 'vc', name: 'Vocoder',             url: `${ONNX_BASE}/vocoder.onnx`,             size: '101 MB'  },
  ];

  const AVAILABLE_LANGS = [
    'en','ko','ja','ar','bg','cs','da','de','el','es',
    'et','fi','fr','hi','hr','hu','id','it','lt','lv',
    'nl','pl','pt','ro','ru','sk','sl','sv','tr','uk','vi','na'
  ];

  // ─── Global State ─────────────────────────────────────────────────────────
  let db = null;
  let tts = null;
  let currentStyle = null;
  let currentAudioUrl = null;
  let currentReferenceUrl = null;
  let currentReferenceBlob = null;
  let currentReferenceName = '';
  let mediaRecorder = null;
  let recordedChunks = [];

  // ─── IndexedDB Cache ──────────────────────────────────────────────────────
  function openDB() {
    return new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, 1);
      req.onupgradeneeded = e => e.target.result.createObjectStore(STORE_NAME);
      req.onsuccess  = e => resolve(e.target.result);
      req.onerror    = e => reject(e.target.error);
    });
  }

  function dbGet(key) {
    if (!db) return Promise.resolve(null);
    return new Promise(resolve => {
      const req = db.transaction(STORE_NAME, 'readonly').objectStore(STORE_NAME).get(key);
      req.onsuccess = () => resolve(req.result ?? null);
      req.onerror   = () => resolve(null);
    });
  }

  function dbPut(key, value) {
    if (!db) return Promise.resolve();
    return new Promise(resolve => {
      try {
        const tx = db.transaction(STORE_NAME, 'readwrite');
        tx.objectStore(STORE_NAME).put(value, key);
        tx.oncomplete = resolve;
        tx.onerror    = resolve; // non-fatal
      } catch (e) { resolve(); }
    });
  }

  async function dbClear() {
    if (!db) return;
    return new Promise(resolve => {
      const tx = db.transaction(STORE_NAME, 'readwrite');
      tx.objectStore(STORE_NAME).clear();
      tx.oncomplete = resolve;
      tx.onerror    = resolve;
    });
  }

  /**
   * Fetch a file, caching its ArrayBuffer in IndexedDB.
   * onProgress(loaded, total, fromCache)
   */
  async function fetchFile(url, onProgress) {
    const cached = await dbGet(url);
    if (cached !== null) {
      onProgress && onProgress(1, 1, true);
      return cached; // ArrayBuffer
    }

    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status} fetching ${url}`);

    const contentLength = parseInt(res.headers.get('content-length') || '0');

    // Stream with progress
    const reader = res.body.getReader();
    const chunks = [];
    let loaded = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      loaded += value.length;
      if (onProgress && contentLength > 0) onProgress(loaded, contentLength, false);
    }

    const merged = new Uint8Array(loaded);
    let offset = 0;
    for (const c of chunks) { merged.set(c, offset); offset += c.length; }

    const buf = merged.buffer;
    await dbPut(url, buf).catch(() => {});
    return buf;
  }

  // ─── Text Processor ───────────────────────────────────────────────────────
  class UnicodeProcessor {
    constructor(indexer) { this.indexer = indexer; }

    call(textList, langList) {
      const processed = textList.map((t, i) => this._preprocess(t, langList[i]));
      const lengths   = processed.map(t => t.length);
      const maxLen    = Math.max(...lengths);

      const textIds = processed.map(text => {
        const row = new Array(maxLen).fill(0);
        for (let j = 0; j < text.length; j++) {
          const cp = text.codePointAt(j);
          row[j] = cp < this.indexer.length ? this.indexer[cp] : -1;
        }
        return row;
      });

      return { textIds, textMask: this._mask(lengths) };
    }

    _preprocess(text, lang) {
      text = text.normalize('NFKD');
      // Remove emoji
      text = text.replace(
        /[\u{1F600}-\u{1F64F}\u{1F300}-\u{1F5FF}\u{1F680}-\u{1F6FF}\u{1F700}-\u{1F77F}\u{1F780}-\u{1F7FF}\u{1F800}-\u{1F8FF}\u{1F900}-\u{1F9FF}\u{1FA00}-\u{1FA6F}\u{1FA70}-\u{1FAFF}\u{2600}-\u{26FF}\u{2700}-\u{27BF}\u{1F1E6}-\u{1F1FF}]+/gu, '');
      // Symbol replacements
      const reps = {
        '–':'-','‑':'-','—':'-','_':' ',
        '\u201C':'"','\u201D':'"','\u2018':"'",'\u2019':"'",
        '´':"'",'`':"'",'[':' ',']':' ','|':' ','/':" ",'#':' ','→':' ','←':' '
      };
      for (const [k, v] of Object.entries(reps)) text = text.replaceAll(k, v);
      text = text.replace(/[♥☆♡©\\]/g, '');
      text = text.replaceAll('@', ' at ').replaceAll('e.g.,', 'for example, ').replaceAll('i.e.,', 'that is, ');
      // Fix spacing
      text = text.replace(/ ,/g,',').replace(/ \./g,'.').replace(/ !/g,'!')
                 .replace(/ \?/g,'?').replace(/ ;/g,';').replace(/ :/g,':').replace(/ '/g,"'");
      // Remove duplicate quotes
      while (text.includes('""')) text = text.replace('""', '"');
      while (text.includes("''")) text = text.replace("''", "'");
      text = text.replace(/\s+/g, ' ').trim();
      if (!/[.!?;:,'\"')\]}…。」』】〉》›»]$/.test(text)) text += '.';
      if (!AVAILABLE_LANGS.includes(lang)) throw new Error(`Invalid language: ${lang}`);
      return `<${lang}>${text}`;
    }

    _mask(lengths) {
      const maxLen = Math.max(...lengths);
      return lengths.map(len => {
        const row = new Array(maxLen).fill(0.0);
        for (let j = 0; j < Math.min(len, maxLen); j++) row[j] = 1.0;
        return [row];
      });
    }
  }

  // ─── Style Container ──────────────────────────────────────────────────────
  class Style {
    constructor(ttl, dp) { this.ttl = ttl; this.dp = dp; }
  }

  // ─── TTS Engine ───────────────────────────────────────────────────────────
  class TextToSpeech {
    constructor(cfgs, proc, dpOrt, teOrt, veOrt, vcOrt) {
      this.cfgs = cfgs;
      this.proc = proc;
      this.dpOrt = dpOrt;
      this.teOrt = teOrt;
      this.veOrt = veOrt;
      this.vcOrt = vcOrt;
      this.sampleRate = cfgs.ae.sample_rate;
    }

    async _infer(textList, langList, style, totalStep, speed, progressCb) {
      const bsz = textList.length;
      const { textIds, textMask } = this.proc.call(textList, langList);

      // Save raw data for all tensors used in multiple run() calls.
      // WebGPU runs reuse these values across sessions and denoising steps, so every
      // run() call receives freshly constructed tensors.
      const tiRaw   = new BigInt64Array(textIds.flat().map(x => BigInt(x)));
      const tiShape = [bsz, textIds[0].length];
      const tmRaw   = new Float32Array(textMask.flat(2));
      const tmShape = [bsz, 1, textMask[0][0].length];
      const dpRaw   = new Float32Array(style.dp.data);
      const dpDims  = [...style.dp.dims];
      const ttlRaw  = new Float32Array(style.ttl.data);
      const ttlDims = [...style.ttl.dims];

      const mkTi  = () => new ort.Tensor('int64',   new BigInt64Array(tiRaw),  tiShape);
      const mkTm  = () => new ort.Tensor('float32', new Float32Array(tmRaw),   tmShape);
      const mkDp  = () => new ort.Tensor('float32', new Float32Array(dpRaw),   dpDims);
      const mkTtl = () => new ort.Tensor('float32', new Float32Array(ttlRaw),  ttlDims);

      // Duration predictor
      const dpOut = await this.dpOrt.run({ text_ids: mkTi(), style_dp: mkDp(), text_mask: mkTm() });
      const duration = Array.from(dpOut.duration.data).map(d => d / speed);

      // Text encoder
      const teOut = await this.teOrt.run({ text_ids: mkTi(), style_ttl: mkTtl(), text_mask: mkTm() });
      const textEmbRaw  = new Float32Array(teOut.text_emb.data);
      const textEmbDims = [...teOut.text_emb.dims];
      const mkTe  = () => new ort.Tensor('float32', new Float32Array(textEmbRaw), textEmbDims);

      // Sample noisy latent
      let { xt, latentMask } = this._sampleNoisy(duration);
      const lmRaw   = new Float32Array(latentMask.flat(2));
      const lmShape = [bsz, 1, latentMask[0][0].length];
      const tsRaw   = new Float32Array(bsz).fill(totalStep);
      const mkLm  = () => new ort.Tensor('float32', new Float32Array(lmRaw),  lmShape);
      const mkTs  = () => new ort.Tensor('float32', new Float32Array(tsRaw),  [bsz]);

      // Denoising loop — every tensor must be freshly created each iteration
      for (let step = 0; step < totalStep; step++) {
        progressCb && progressCb(step + 1, totalStep);
        const csT = new ort.Tensor('float32', new Float32Array(bsz).fill(step), [bsz]);
        const xtT = new ort.Tensor('float32',
          new Float32Array(xt.flat(2)),
          [bsz, xt[0].length, xt[0][0].length]);

        const veOut = await this.veOrt.run({
          noisy_latent: xtT, text_emb: mkTe(), style_ttl: mkTtl(),
          latent_mask: mkLm(), text_mask: mkTm(), current_step: csT, total_step: mkTs()
        });

        const denoised = Array.from(veOut.denoised_latent.data);
        const ld = xt[0].length, ll = xt[0][0].length;
        xt = []; let idx = 0;
        for (let b = 0; b < bsz; b++) {
          const batch = [];
          for (let d = 0; d < ld; d++) {
            const row = [];
            for (let t = 0; t < ll; t++) row.push(denoised[idx++]);
            batch.push(row);
          }
          xt.push(batch);
        }
      }

      // Vocoder
      const fxt = new ort.Tensor('float32',
        new Float32Array(xt.flat(2)),
        [bsz, xt[0].length, xt[0][0].length]);
      const vcOut = await this.vcOrt.run({ latent: fxt });
      return { wav: Array.from(vcOut.wav_tts.data), duration };
    }

    async call(text, lang, style, totalStep, speed, silenceSec, progressCb) {
      const maxLen = (lang === 'ko' || lang === 'ja') ? 120 : 300;
      const chunks = chunkText(text, maxLen);
      let wavCat = [], durCat = 0;
      const silenceLen = Math.floor(silenceSec * this.sampleRate);

      for (const chunk of chunks) {
        const { wav, duration } = await this._infer([chunk], [lang], style, totalStep, speed, progressCb);
        if (wavCat.length === 0) {
          wavCat = wav; durCat = duration[0];
        } else {
          wavCat = [...wavCat, ...new Array(silenceLen).fill(0), ...wav];
          durCat += duration[0] + silenceSec;
        }
      }
      return { wav: wavCat, duration: [durCat] };
    }

    _sampleNoisy(duration) {
      const bsz = duration.length;
      const { ae: { sample_rate: sr, base_chunk_size: bcs }, ttl: { chunk_compress_factor: ccf, latent_dim: ld } } = this.cfgs;
      const chunkSize  = bcs * ccf;
      const maxDur     = Math.max(...duration);
      const wavLenMax  = Math.floor(maxDur * sr);
      const wavLengths = duration.map(d => Math.floor(d * sr));
      const latentLen  = Math.floor((wavLenMax + chunkSize - 1) / chunkSize);
      const latentDim  = ld * ccf;

      const xt = [];
      for (let b = 0; b < bsz; b++) {
        const batch = [];
        for (let d = 0; d < latentDim; d++) {
          const row = [];
          for (let t = 0; t < latentLen; t++) {
            const u1  = Math.max(0.0001, Math.random()), u2 = Math.random();
            row.push(Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2));
          }
          batch.push(row);
        }
        xt.push(batch);
      }

      const latentLengths = wavLengths.map(l => Math.floor((l + chunkSize - 1) / chunkSize));
      const latentMask = latentLengths.map(len => {
        const row = new Array(latentLen).fill(0.0);
        for (let j = 0; j < Math.min(len, latentLen); j++) row[j] = 1.0;
        return [row];
      });

      for (let b = 0; b < bsz; b++)
        for (let d = 0; d < latentDim; d++)
          for (let t = 0; t < latentLen; t++)
            xt[b][d][t] *= latentMask[b][0][t];

      return { xt, latentMask };
    }
  }

  // ─── Helpers ──────────────────────────────────────────────────────────────
  function chunkText(text, maxLen) {
    const paragraphs = text.trim().split(/\n\s*\n+/).filter(p => p.trim());
    const chunks = [];
    for (let para of paragraphs) {
      para = para.trim();
      if (!para) continue;
      let sentences;
      try {
        sentences = para.split(
          /(?<!Mr\.|Mrs\.|Ms\.|Dr\.|Prof\.|Sr\.|Jr\.|Ph\.D\.|etc\.|e\.g\.|i\.e\.|vs\.|Inc\.|Ltd\.|Co\.|Corp\.|St\.|Ave\.|Blvd\.)(?<!\b[A-Z]\.)(?<=[.!?])\s+/
        );
      } catch (e) {
        // Fallback for browsers without lookbehind
        sentences = para.split(/(?<=[.!?])\s+/);
      }
      let cur = '';
      for (const s of sentences) {
        if (cur.length + s.length + 1 <= maxLen) {
          cur += (cur ? ' ' : '') + s;
        } else {
          if (cur) chunks.push(cur.trim());
          cur = s;
        }
      }
      if (cur) chunks.push(cur.trim());
    }
    return chunks.length ? chunks : [text.trim()];
  }

  function writeWav(audioData, sampleRate) {
    const nc = 1, bps = 16;
    const byteRate   = sampleRate * nc * bps / 8;
    const blockAlign = nc * bps / 8;
    const dataSize   = audioData.length * 2;
    const buf  = new ArrayBuffer(44 + dataSize);
    const view = new DataView(buf);
    const ws = (off, str) => { for (let i = 0; i < str.length; i++) view.setUint8(off + i, str.charCodeAt(i)); };
    ws(0,'RIFF'); view.setUint32(4, 36 + dataSize, true);
    ws(8,'WAVE'); ws(12,'fmt '); view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); view.setUint16(22, nc, true);
    view.setUint32(24, sampleRate, true); view.setUint32(28, byteRate, true);
    view.setUint16(32, blockAlign, true); view.setUint16(34, bps, true);
    ws(36,'data'); view.setUint32(40, dataSize, true);
    const i16 = new Int16Array(audioData.length);
    for (let i = 0; i < audioData.length; i++)
      i16[i] = Math.floor(Math.max(-1, Math.min(1, audioData[i])) * 32767);
    new Uint8Array(buf, 44).set(new Uint8Array(i16.buffer));
    return buf;
  }

  function fmtBytes(b) {
    if (b < 1024) return b + ' B';
    if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';
    return (b / 1048576).toFixed(1) + ' MB';
  }

  function escapeHtml(value) {
    return String(value).replace(/[&<>"']/g, ch => ({
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }[ch]));
  }

  function safeFileStem(value, fallback = 'reference') {
    const stem = String(value || fallback).replace(/\.[^.]+$/, '').trim();
    return (stem || fallback).replace(/[^a-z0-9_-]+/gi, '-').replace(/^-+|-+$/g, '') || fallback;
  }

  function blobToDataUrl(blob) {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = () => resolve(String(reader.result));
      reader.onerror = () => reject(reader.error || new Error('Failed to read audio blob.'));
      reader.readAsDataURL(blob);
    });
  }

  function dataUrlToBlob(dataUrl) {
    const match = /^data:([^;,]+)?(;base64)?,(.*)$/i.exec(String(dataUrl || ''));
    if (!match || !match[2]) throw new Error('Reference JSON audioData must be a base64 data URL.');
    const mimeType = match[1] || 'audio/wav';
    const binary = atob(match[3]);
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    return new Blob([bytes], { type: mimeType });
  }

  async function readReferenceAudioPackage(file) {
    const pkg = JSON.parse(await file.text());
    if (pkg?.kind !== 'supertonic-reference-audio' || !pkg.audioData) {
      throw new Error('這不是參考音 JSON。Voice Builder 產出的 Voice JSON 請用下方 Voice JSON 區塊匯入。');
    }
    return {
      blob: dataUrlToBlob(pkg.audioData),
      sourceName: pkg.sourceName || file.name.replace(/\.json$/i, '.wav') || 'reference.wav'
    };
  }

  function containsHanText(text) {
    return /[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]/u.test(text);
  }

  // ─── UI Helpers ───────────────────────────────────────────────────────────
  const $ = id => document.getElementById(id);

  function setStatus(msg, type = 'info') {
    const badge = $('statusBadge');
    const text  = $('statusMsg');
    if (!badge || !text) return;
    const labels = { loading: 'LOADING', success: 'READY', error: 'ERROR', info: 'INFO' };
    badge.className = `status-badge ${type}`;
    badge.textContent = labels[type] || type.toUpperCase();
    text.textContent = msg;
  }

  function setProgress(loaded, total, label) {
    const wrap  = $('progressWrap');
    const fill  = $('progressFill');
    const pLabel = $('progressLabel');
    const pMeta  = $('progressMeta');
    if (!wrap) return;
    if (loaded === 0 && total === 0) { wrap.classList.add('hidden'); return; }
    wrap.classList.remove('hidden');
    const pct = total > 0 ? Math.min(100, (loaded / total) * 100) : 0;
    fill.style.width = pct.toFixed(1) + '%';
    if (pLabel) pLabel.textContent = label || '';
    if (pMeta)  pMeta.textContent  = `${fmtBytes(loaded)} / ${fmtBytes(total)} (${pct.toFixed(0)}%)`;
  }

  function setModelDot(id, state) {
    const el = $(`dot-${id}`);
    if (el) el.className = `dot ${state}`;
  }

  function setBackendBadge(text, type = 'webgpu') {
    const backendEl = $('backendBadge');
    if (!backendEl) return;
    backendEl.className = `backend-badge ${type}`;
    backendEl.textContent = text;
    backendEl.classList.remove('hidden');
  }

  async function requireWebGpu() {
    if (!navigator.gpu) {
      setBackendBadge('WebGPU unavailable', 'error');
      throw new Error(WEBGPU_ERROR);
    }

    const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
    if (!adapter) {
      setBackendBadge('No WebGPU adapter', 'error');
      throw new Error(WEBGPU_ERROR);
    }

    if (ort.env.webgpu) {
      ort.env.webgpu.powerPreference = 'high-performance';
      ort.env.webgpu.forceFallbackAdapter = false;
      ort.env.webgpu.adapter = adapter;
    }

    setBackendBadge('WebGPU only', 'webgpu');
  }

  // ─── Voice Style Loader ───────────────────────────────────────────────────
  function createStyleFromDefinition(vs) {
    if (!vs || !vs.style_ttl || !vs.style_dp || !vs.style_ttl.data || !vs.style_dp.data || !vs.style_ttl.dims || !vs.style_dp.dims) {
      throw new Error('Voice JSON must include style_ttl and style_dp data/dims.');
    }
    const ttlFlat = new Float32Array(vs.style_ttl.data.flat(Infinity));
    const dpFlat  = new Float32Array(vs.style_dp.data.flat(Infinity));
    return new Style(
      new ort.Tensor('float32', ttlFlat, vs.style_ttl.dims),
      new ort.Tensor('float32', dpFlat,  vs.style_dp.dims)
    );
  }

  async function loadVoiceStyle(name) {
    const url = `${VS_BASE}/${name}.json`;
    const buf = await fetchFile(url);
    const vs  = JSON.parse(new TextDecoder().decode(buf));
    return createStyleFromDefinition(vs);
  }

  function getCustomVoices() {
    try {
      const raw = localStorage.getItem(CUSTOM_VOICES_KEY);
      const voices = raw ? JSON.parse(raw) : [];
      return Array.isArray(voices) ? voices : [];
    } catch (e) {
      return [];
    }
  }

  function saveCustomVoices(voices) {
    localStorage.setItem(CUSTOM_VOICES_KEY, JSON.stringify(voices));
  }

  function loadCustomVoiceStyle(id) {
    const voice = getCustomVoices().find(item => item.id === id);
    if (!voice) throw new Error('Custom voice not found.');
    return createStyleFromDefinition(voice.definition);
  }

  async function loadSelectedVoiceStyle(value) {
    if (value.startsWith('custom:')) return loadCustomVoiceStyle(value.slice(7));
    return loadVoiceStyle(value);
  }

  function renderCustomVoices() {
    const select = $('voiceSel');
    const list = $('customVoiceList');
    if (!select || !list) return;

    const existing = select.querySelector('optgroup[data-custom-voices]');
    if (existing) existing.remove();

    const voices = getCustomVoices();
    if (voices.length) {
      const group = document.createElement('optgroup');
      group.label = 'Custom Voices';
      group.dataset.customVoices = 'true';
      voices.forEach(voice => {
        const option = document.createElement('option');
        option.value = `custom:${voice.id}`;
        option.textContent = voice.name;
        group.appendChild(option);
      });
      select.appendChild(group);
      list.innerHTML = voices.map(voice => `<span class="voice-chip">${escapeHtml(voice.name)}</span>`).join('');
    } else {
      list.textContent = '尚未匯入自訂 voice。';
    }
  }

  async function importCustomVoice() {
    const fileInput = $('customVoiceJson');
    const nameInput = $('customVoiceName');
    const file = fileInput?.files?.[0];
    if (!file) {
      setReferenceStatus('請先選擇 Supertonic Voice JSON。', true);
      return;
    }

    try {
      const definition = JSON.parse(await file.text());
      if (definition?.kind === 'supertonic-reference-audio') {
        throw new Error('這是參考音 JSON，不是 Voice Builder style_ttl/style_dp Voice JSON。請改用上方聲音參考區塊匯入。');
      }
      createStyleFromDefinition(definition);
      const fallbackName = file.name.replace(/\.json$/i, '') || 'Custom Voice';
      const name = (nameInput.value || fallbackName).trim();
      const voices = getCustomVoices();
      const id = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
      voices.push({ id, name, definition });
      saveCustomVoices(voices);
      renderCustomVoices();

      $('voiceSel').value = `custom:${id}`;
      if (tts) {
        currentStyle = loadCustomVoiceStyle(id);
        setStatus(`Custom voice "${name}" ready.`, 'success');
      }
      fileInput.value = '';
      nameInput.value = '';
    } catch (e) {
      setStatus('Failed to import custom voice: ' + e.message, 'error');
      setReferenceStatus('匯入失敗：' + e.message, true);
    }
  }

  function clearCustomVoices() {
    if (!confirm('Clear all imported custom voices?')) return;
    localStorage.removeItem(CUSTOM_VOICES_KEY);
    const voiceSel = $('voiceSel');
    if (voiceSel && voiceSel.value.startsWith('custom:')) voiceSel.value = 'M1';
    renderCustomVoices();
    setStatus('Custom voices cleared.', 'info');
  }

  function setReferenceStatus(message, isError = false) {
    const el = $('referenceStatus');
    if (!el) return;
    el.textContent = message;
    el.className = `reference-status${isError ? ' error' : ''}`;
  }

  function isWavAudio(blob, sourceName = '') {
    return /wav/i.test(blob.type || '') || /\.wav$/i.test(sourceName);
  }

  async function audioBlobToWav(blob, sourceName = '') {
    if (isWavAudio(blob, sourceName)) {
      return blob.type ? blob : new Blob([blob], { type: 'audio/wav' });
    }

    const AudioContextCtor = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextCtor) throw new Error('This browser cannot decode audio files.');
    let ctx;
    try {
      ctx = new AudioContextCtor({ sampleRate: 44100 });
    } catch (e) {
      ctx = new AudioContextCtor();
    }
    try {
      const audioBuffer = await ctx.decodeAudioData(await blob.arrayBuffer());
      const mono = new Float32Array(audioBuffer.length);
      for (let channel = 0; channel < audioBuffer.numberOfChannels; channel++) {
        const data = audioBuffer.getChannelData(channel);
        for (let i = 0; i < data.length; i++) mono[i] += data[i] / audioBuffer.numberOfChannels;
      }
      return new Blob([writeWav(Array.from(mono), audioBuffer.sampleRate)], { type: 'audio/wav' });
    } finally {
      if (ctx.close) ctx.close();
    }
  }

  async function setReferenceBlob(blob, sourceName) {
    const audio = $('referenceAudio');
    const downloadBtn = $('downloadReferenceBtn');
    const downloadJsonBtn = $('downloadReferenceJsonBtn');
    setReferenceStatus('正在整理參考音並轉成 WAV…');

    try {
      currentReferenceBlob = await audioBlobToWav(blob, sourceName);
      currentReferenceName = sourceName || 'reference.wav';
      if (currentReferenceUrl) URL.revokeObjectURL(currentReferenceUrl);
      currentReferenceUrl = URL.createObjectURL(currentReferenceBlob);
      audio.src = currentReferenceUrl;
      audio.classList.remove('hidden');
      downloadBtn.disabled = false;
      downloadJsonBtn.disabled = false;
      setReferenceStatus(`參考音已就緒：${sourceName} · ${fmtBytes(currentReferenceBlob.size)} WAV`);
    } catch (e) {
      currentReferenceBlob = blob;
      currentReferenceName = sourceName || 'reference-audio';
      if (currentReferenceUrl) URL.revokeObjectURL(currentReferenceUrl);
      currentReferenceUrl = URL.createObjectURL(blob);
      audio.src = currentReferenceUrl;
      audio.classList.remove('hidden');
      downloadBtn.disabled = false;
      downloadJsonBtn.disabled = false;
      setReferenceStatus(`無法轉 WAV，已保留原始音檔：${e.message}`, true);
    }
  }

  async function startRecording() {
    if (!navigator.mediaDevices?.getUserMedia) {
      setReferenceStatus('此瀏覽器不支援麥克風錄音。', true);
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      recordedChunks = [];
      mediaRecorder = new MediaRecorder(stream);
      mediaRecorder.addEventListener('dataavailable', event => {
        if (event.data.size > 0) recordedChunks.push(event.data);
      });
      mediaRecorder.addEventListener('stop', () => {
        stream.getTracks().forEach(track => track.stop());
        const blob = new Blob(recordedChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
        setReferenceBlob(blob, 'recording');
      });
      mediaRecorder.start();
      $('startRecordBtn').disabled = true;
      $('stopRecordBtn').disabled = false;
      setReferenceStatus('錄音中。建議錄 5–10 秒乾淨人聲。');
    } catch (e) {
      setReferenceStatus('無法開始錄音：' + e.message, true);
    }
  }

  function stopRecording() {
    if (!mediaRecorder || mediaRecorder.state === 'inactive') return;
    mediaRecorder.stop();
    $('startRecordBtn').disabled = false;
    $('stopRecordBtn').disabled = true;
  }

  function downloadReference() {
    if (!currentReferenceBlob) return;
    const a = document.createElement('a');
    a.href = currentReferenceUrl;
    a.download = `${safeFileStem(currentReferenceName, 'supertonic-reference')}.wav`;
    a.click();
  }

  async function downloadReferenceJson() {
    if (!currentReferenceBlob) return;
    const payload = {
      kind: 'supertonic-reference-audio',
      version: 1,
      createdAt: new Date().toISOString(),
      sourceName: currentReferenceName || 'reference.wav',
      mimeType: currentReferenceBlob.type || 'audio/wav',
      sizeBytes: currentReferenceBlob.size,
      notVoiceStyle: true,
      note: 'Portable reference-audio package for this tool. This is not a Supertonic Voice Builder style_ttl/style_dp Voice JSON.',
      audioData: await blobToDataUrl(currentReferenceBlob)
    };
    const json = JSON.stringify(payload, null, 2);
    const url = URL.createObjectURL(new Blob([json], { type: 'application/json' }));
    const a = document.createElement('a');
    a.href = url;
    a.download = `${safeFileStem(currentReferenceName, 'supertonic-reference')}.reference.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  // ─── Model Loader ─────────────────────────────────────────────────────────
  async function loadModels() {
    // Open IndexedDB (non-fatal)
    try { db = await openDB(); } catch (e) { console.warn('IndexedDB unavailable:', e); }

    setStatus('Checking WebGPU support…', 'loading');
    await requireWebGpu();

    setStatus('Loading configuration…', 'loading');

    // Config files
    const cfgBuf     = await fetchFile(`${ONNX_BASE}/tts.json`);
    const cfgs       = JSON.parse(new TextDecoder().decode(cfgBuf));
    const idxBuf     = await fetchFile(`${ONNX_BASE}/unicode_indexer.json`);
    const indexer    = JSON.parse(new TextDecoder().decode(idxBuf));
    const textProc   = new UnicodeProcessor(indexer);

    const sessionOpts = {
      graphOptimizationLevel: 'all',
      executionProviders: ['webgpu']
    };

    // Load ONNX sessions
    const sessions = {};
    for (const m of MODEL_DEFS) {
      setModelDot(m.id, 'loading');
      setStatus(`Downloading ${m.name} (${m.size})…`, 'loading');

      let fromCache = false;
      const buf = await fetchFile(m.url, (loaded, total, cached) => {
        fromCache = cached;
        if (!cached) setProgress(loaded, total, `${m.name} — ${m.size}`);
      });

      setProgress(0, 0, '');
      if (!fromCache) setStatus(`Initializing ${m.name}…`, 'loading');

      try {
        sessions[m.id] = await ort.InferenceSession.create(buf, sessionOpts);
        setModelDot(m.id, 'done');
      } catch (err) {
        setModelDot(m.id, 'error');
        throw err;
      }
    }

    tts = new TextToSpeech(cfgs, textProc,
      sessions.dp, sessions.te, sessions.ve, sessions.vc);

    // Load selected voice style
    const selectedVoice = $('voiceSel')?.value || 'M1';
    setStatus(`Loading voice style ${selectedVoice.replace('custom:', '')}…`, 'loading');
    currentStyle = await loadSelectedVoiceStyle(selectedVoice);

    setStatus('Ready! WebGPU models loaded — click Generate Speech.', 'success');
    $('generateBtn').disabled = false;
    await updateCacheSize();
  }

  // ─── Generate Speech ──────────────────────────────────────────────────────
  async function generate() {
    const text = $('textInput').value.trim();
    if (!text) { showResultError('Please enter some text.'); return; }
    if (!tts || !currentStyle) { showResultError('Models not loaded yet.'); return; }

    const lang      = $('langSel').value;
    const totalStep = parseInt($('stepsInput').value) || 8;
    const speed     = parseFloat($('speedInput').value) || 1.05;

    if (containsHanText(text) && lang !== 'ja' && lang !== 'ko') {
      showResultError('Supertonic 3 does not support Chinese (zh). Please use supported-language text, or choose Japanese/Korean only when the input is actually ja/ko.');
      setStatus('Chinese (zh) is not supported by this model.', 'error');
      return;
    }

    $('generateBtn').disabled = true;
    const startMs = Date.now();

    try {
      setStatus(`Generating speech…`, 'loading');
      $('resultsArea').innerHTML = `
        <div class="results-placeholder">
          <span class="icon">⏳</span>
          <p>Generating speech…</p>
        </div>`;

      const { wav, duration } = await tts.call(
        text, lang, currentStyle, totalStep, speed, 0.3,
        (step, total) => setStatus(`Denoising: step ${step} / ${total}`, 'loading')
      );

      const wavLen = Math.floor(tts.sampleRate * duration[0]);
      const wavOut = wav.slice(0, wavLen);
      const wavBuf = writeWav(wavOut, tts.sampleRate);
      const blob   = new Blob([wavBuf], { type: 'audio/wav' });

      // Revoke previous
      if (currentAudioUrl) URL.revokeObjectURL(currentAudioUrl);
      currentAudioUrl = URL.createObjectURL(blob);

      const elapsed   = ((Date.now() - startMs) / 1000).toFixed(2);
      const audioDur  = duration[0].toFixed(2);
      const charCount = text.length;

      $('resultsArea').innerHTML = `
        <div class="result-item">
          <div class="result-meta">
            <span>🎵 Audio: ${audioDur}s</span>
            <span>⏱ Generated in: ${elapsed}s</span>
            <span>📝 ${charCount} chars · ${lang.toUpperCase()}</span>
          </div>
          <audio controls src="${currentAudioUrl}"></audio>
          <div class="result-actions">
            <button class="btn btn-primary" onclick="window._sttsDownload()">⬇ Download WAV</button>
          </div>
        </div>`;

      setStatus(`Done! Audio ${audioDur}s generated in ${elapsed}s.`, 'success');
    } catch (e) {
      console.error(e);
      setStatus('Error: ' + e.message, 'error');
      showResultError(e.message);
    } finally {
      $('generateBtn').disabled = false;
    }
  }

  function showResultError(msg) {
    $('resultsArea').innerHTML = `
      <div class="results-placeholder">
        <span class="icon">❌</span>
        <p style="color:#f87171">${msg}</p>
      </div>`;
  }

  // ─── Cache Size Display ───────────────────────────────────────────────────
  async function updateCacheSize() {
    const el = $('cacheSize');
    if (!el) return;
    try {
      if ('storage' in navigator && 'estimate' in navigator.storage) {
        const { usage } = await navigator.storage.estimate();
        el.textContent = `Cache: ~${fmtBytes(usage)}`;
      }
    } catch (e) { /* ignore */ }
  }

  // ─── Global exports (called from onclick) ────────────────────────────────
  window._sttsDownload = function () {
    if (!currentAudioUrl) return;
    const a = document.createElement('a');
    a.href = currentAudioUrl;
    a.download = 'supertonic-speech.wav';
    a.click();
  };

  window._sttsClearCache = async function () {
    if (!confirm('Clear all cached models (~400 MB)?\nModels will be re-downloaded on next load.')) return;
    await dbClear();
    $('cacheSize').textContent = 'Cache cleared';
    setTimeout(() => location.reload(), 800);
  };

  // ─── Init ─────────────────────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', () => {
    // Configure ORT
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/';
    ort.env.wasm.numThreads = 1; // Keep the JSEP/WASM support files single-threaded.
    ort.env.wasm.proxy = false;  // WebGPU EP cannot use ORT's WASM proxy worker.

    const genBtn = $('generateBtn');
    genBtn.addEventListener('click', generate);
    renderCustomVoices();

    $('startRecordBtn').addEventListener('click', startRecording);
    $('stopRecordBtn').addEventListener('click', stopRecording);
    $('downloadReferenceBtn').addEventListener('click', downloadReference);
    $('downloadReferenceJsonBtn').addEventListener('click', () => {
      downloadReferenceJson().catch(e => setReferenceStatus('無法下載參考音 JSON：' + e.message, true));
    });
    $('importVoiceBtn').addEventListener('click', importCustomVoice);
    $('clearVoicesBtn').addEventListener('click', clearCustomVoices);
    $('referenceAudioFile').addEventListener('change', async e => {
      const file = e.target.files?.[0];
      if (!file) return;
      try {
        if (/json/i.test(file.type || '') || /\.json$/i.test(file.name)) {
          const reference = await readReferenceAudioPackage(file);
          await setReferenceBlob(reference.blob, reference.sourceName);
        } else {
          await setReferenceBlob(file, file.name);
        }
      } catch (err) {
        setReferenceStatus('參考音匯入失敗：' + err.message, true);
      } finally {
        e.target.value = '';
      }
    });

    // Voice style change
    $('voiceSel').addEventListener('change', async (e) => {
      if (!tts) return;
      const name = e.target.value;
      genBtn.disabled = true;
      setStatus(`Loading voice style ${name.replace('custom:', '')}…`, 'loading');
      try {
        currentStyle = await loadSelectedVoiceStyle(name);
        setStatus(`Voice style ${name.replace('custom:', '')} ready.`, 'success');
      } catch (err) {
        setStatus('Failed to load voice style: ' + err.message, 'error');
      }
      genBtn.disabled = false;
    });

    // Load models
    loadModels().catch(e => {
      console.error(e);
      setStatus('Failed to load models: ' + e.message, 'error');
      MODEL_DEFS.forEach(m => {
        const dot = $(`dot-${m.id}`);
        if (dot && dot.className.includes('loading')) setModelDot(m.id, 'error');
      });
    });
  });
})();
