(() => {
  const script = document.currentScript;
  const toolName = script?.dataset.toolName || 'fal.ai API';
  const runtimeBase = new URL('.', script.src);
  const escapeHtml = value => String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));

  document.documentElement.lang = 'zh-Hant';
  document.title = `${toolName} — Endman100 Tools`;
  document.body.dataset.toolId = script?.dataset.toolId || 'fal-ai';
  document.body.innerHTML = `
    <div class="bg-glow"></div>

    <nav class="navbar">
      <div class="navbar-inner">
        <a href="../../index.html" class="navbar-brand">
          <span class="brand-icon">API</span>
          Endman100 Tools
        </a>
        <div class="navbar-links">
          <a href="../../index.html">首頁</a>
          <a href="../../index.html#api-tools" class="active">API 類工具</a>
        </div>
      </div>
    </nav>

    <main class="fal-shell">
      <a href="../../index.html#api-tools" class="back-link">返回 API 工具列表</a>

      <header class="tool-page-header">
        <div class="icon">fal</div>
        <div>
          <h1>${escapeHtml(toolName)}</h1>
          <p>單一面板選擇 fal.ai 模型，讀取 OpenAPI schema 產生輸入表單，送出 queue request 並預覽圖片、影片、音訊或結果 JSON。</p>
        </div>
      </header>

      <section class="notice-card">
        <strong>瀏覽器直連提醒</strong>
        <span>這個工具維持純前端架構。fal.ai 的 models、OpenAPI schema 與 queue API 可能被瀏覽器 CORS 擋住；若要完整自動載入，請填入你自己的 proxy URL。</span>
      </section>

      <section class="workspace-grid">
        <aside class="model-panel card">
          <div class="card-title">模型下拉搜尋</div>
          <label class="search-box" for="modelSearch">
            <span>搜尋或輸入 model ID</span>
            <input id="modelSearch" type="search" list="modelOptions" placeholder="flux/dev、kling、veo、tts..." autocomplete="off" spellcheck="false" role="combobox" aria-controls="modelList" aria-autocomplete="list" />
            <datalist id="modelOptions"></datalist>
          </label>
          <div class="button-row tight">
            <button type="button" class="btn btn-ghost" id="refreshCatalogBtn">重新載入模型目錄</button>
            <button type="button" class="btn btn-ghost" id="loadSchemaBtn">載入 schema</button>
          </div>
          <div class="catalog-status" id="catalogStatus">正在準備模型清單...</div>
          <div class="model-list" id="modelList"></div>
        </aside>

        <section class="tool-panel card">
          <div class="service-heading">
            <div>
              <span class="tag" id="modelCategory">fal.ai</span>
              <h2 id="modelTitle">選擇模型</h2>
              <p id="modelSummary">選擇模型後會自動讀取 schema 並產生欄位。</p>
            </div>
            <a class="method-pill" id="docsLink" href="https://fal.ai/models" target="_blank" rel="noopener">Docs</a>
          </div>

          <div class="status-card" id="statusCard">
            <span class="status-badge info" id="statusBadge">READY</span>
            <span class="status-msg" id="statusMsg">請選擇 fal.ai 模型。</span>
          </div>

          <form id="falForm" class="api-form">
            <section class="form-section">
              <div class="card-title">Proxy URL</div>
              <div class="form-grid">
                <div class="form-group full">
                  <label for="proxyUrlInput">Proxy base URL</label>
                  <input id="proxyUrlInput" name="proxyUrl" type="url" autocomplete="off" placeholder="https://your-proxy.example.com/fal" spellcheck="false" />
                  <div class="form-help">預期路徑：GET /models、GET /openapi?endpoint_id=...、POST/GET /queue/{modelId}。留空時會嘗試直連 fal.ai，但多數瀏覽器會被 CORS 擋住。</div>
                </div>
              </div>
              <div class="button-row">
                <button type="button" class="btn btn-ghost" id="saveProxyBtn">儲存 proxy URL</button>
                <button type="button" class="btn btn-danger" id="clearProxyBtn">清除 proxy URL</button>
              </div>
            </section>

            <section class="form-section">
              <div class="card-title">API key</div>
              <div class="form-grid">
                <div class="form-group full">
                  <label for="apiKeyInput">FAL_KEY</label>
                  <input id="apiKeyInput" name="apiKey" type="password" autocomplete="off" placeholder="fal key" />
                  <div class="form-help">送出任務時使用 Authorization: Key。若 proxy 已在後端注入 FAL_KEY，這裡可留空。</div>
                </div>
              </div>
              <div class="button-row">
                <button type="button" class="btn btn-ghost" id="saveKeyBtn">儲存到 cookie</button>
                <button type="button" class="btn btn-danger" id="clearKeyBtn">清除 key</button>
              </div>
            </section>

            <section class="form-section">
              <div class="card-title">Input</div>
              <div id="schemaMeta" class="schema-meta"></div>
              <div class="form-grid" id="dynamicFields"></div>
            </section>

            <section class="action-card">
              <div>
                <div class="card-title compact">呼叫 fal queue</div>
                <p id="requestHint">表單會依目前 schema 產生 JSON body。</p>
              </div>
              <button type="submit" class="btn btn-primary" id="sendBtn">送出任務</button>
            </section>
          </form>
        </section>
      </section>

      <section class="results-layout">
        <div class="card preview-card">
          <div class="card-title">結果預覽</div>
          <div id="previewArea" class="preview-area">
            <div class="empty-state">
              <span>Preview</span>
              <p>選擇模型後會先顯示 fal metadata 圖片；任務完成後顯示輸出媒體。</p>
            </div>
          </div>
        </div>

        <div class="card request-card">
          <div class="card-title">Request / Response</div>
          <div class="code-tabs">
            <button type="button" class="code-tab active" data-code-tab="request">Request</button>
            <button type="button" class="code-tab" data-code-tab="response">Response</button>
            <button type="button" class="code-tab" data-code-tab="logs">Logs</button>
          </div>
          <pre id="requestPreview" class="code-block"></pre>
          <pre id="responsePreview" class="code-block hidden"></pre>
          <pre id="queueLog" class="code-block hidden"></pre>
        </div>
      </section>
    </main>
  `;

  const css = document.createElement('link');
  css.rel = 'stylesheet';
  css.href = `${new URL('style.css', runtimeBase).href}?v=fal-unified`;
  document.head.appendChild(css);

  const runtime = document.createElement('script');
  runtime.src = new URL('index.js', runtimeBase).href;
  document.body.appendChild(runtime);
})();
