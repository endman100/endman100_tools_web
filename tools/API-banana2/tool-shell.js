(() => {
  const script = document.currentScript;
  const toolId = script?.dataset.toolId || '';
  const toolName = script?.dataset.toolName || 'API 工具';
  const browserNote = script?.dataset.browserNote || '';
  const runtimeBase = new URL('.', script.src);
  const escapeHtml = value => String(value).replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
  const browserNotice = browserNote ? `
      <section class="notice-card browser-note">
        <strong>瀏覽器直連限制</strong>
        <span>${escapeHtml(browserNote)}</span>
      </section>` : '';

  document.documentElement.lang = 'zh-Hant';
  document.title = `${toolName} — Endman100 Tools`;
  document.body.dataset.toolId = toolId;
  document.body.innerHTML = `
    <div class="bg-glow"></div>

    <nav class="navbar">
      <div class="navbar-inner">
        <a href="../../index.html" class="navbar-brand">
          <span class="brand-icon">🛠</span>
          Endman100 Tools
        </a>
        <div class="navbar-links">
          <a href="../../index.html">首頁</a>
          <a href="../../index.html#api-tools" class="active">API 類工具</a>
        </div>
      </div>
    </nav>

    <main class="api-shell">
      <a href="../../index.html#api-tools" class="back-link">← 返回 API 工具列表</a>

      <header class="tool-page-header">
        <div class="icon">🔌</div>
        <div>
          <h1>${escapeHtml(toolName)}</h1>
          <p>純前端 API 測試頁。API key 只存在這個瀏覽器的 cookie，不會經過本站後端。</p>
        </div>
      </header>

      ${browserNotice}

      <section class="workspace-grid">
        <aside class="service-panel">
          <div class="panel-title">工具分類</div>
          <div class="category-tabs" id="categoryTabs"></div>
          <label class="search-box" for="serviceSearch">
            <span>搜尋</span>
            <input id="serviceSearch" type="search" placeholder="OpenAI、Gemini、Speech..." autocomplete="off" />
          </label>
          <div class="service-list" id="serviceList"></div>
        </aside>

        <section class="tool-panel">
          <div class="service-heading">
            <div>
              <span class="tag" id="serviceTag">API</span>
              <h2 id="serviceName">選擇 API 工具</h2>
              <p id="serviceSummary">從左側選一個 API 工具開始。</p>
            </div>
            <span class="method-pill" id="methodPill">POST</span>
          </div>

          <div class="status-card" id="statusCard">
            <span class="status-badge info" id="statusBadge">READY</span>
            <span class="status-msg" id="statusMsg">請先選擇 API 工具並填入 API key。</span>
          </div>

          <form id="apiForm" class="api-form">
            <section class="card credential-card">
              <div class="card-title">① API key / 認證</div>
              <div class="form-grid" id="credentialFields"></div>
              <div class="button-row">
                <button type="button" class="btn btn-ghost" id="saveKeyBtn">儲存到 cookie</button>
                <button type="button" class="btn btn-danger" id="clearKeyBtn">清除這個工具的 API key</button>
              </div>
            </section>

            <section class="card">
              <div class="card-title">② Endpoint</div>
              <div class="form-group">
                <label for="endpointInput">請求網址，可使用欄位值替換 {model}、{region}、{project} 等變數</label>
                <input id="endpointInput" name="endpoint" type="url" spellcheck="false" />
              </div>
            </section>

            <section class="card">
              <div class="card-title">③ Input</div>
              <div class="form-grid" id="dynamicFields"></div>
            </section>

            <section class="card action-card">
              <div>
                <div class="card-title compact">④ 呼叫服務</div>
                <p id="requestHint">不填 API key 時不會送出請求。</p>
              </div>
              <button type="submit" class="btn btn-primary" id="sendBtn">呼叫 API</button>
            </section>
          </form>
        </section>
      </section>

      <section class="results-layout">
        <div class="card preview-card">
          <div class="card-title">結果預覽</div>
          <div id="previewArea" class="preview-area">
            <div class="empty-state">
              <span>🔎</span>
              <p>呼叫 API 後會在這裡預覽文字、圖片、音訊、影片或任務 JSON。</p>
            </div>
          </div>
        </div>

        <div class="card request-card">
          <div class="card-title">Request / Response</div>
          <div class="code-tabs">
            <button type="button" class="code-tab active" data-code-tab="request">Request</button>
            <button type="button" class="code-tab" data-code-tab="response">Response</button>
          </div>
          <pre id="requestPreview" class="code-block"></pre>
          <pre id="responsePreview" class="code-block hidden"></pre>
        </div>
      </section>
    </main>
  `;

  const css = document.createElement('link');
  css.rel = 'stylesheet';
  css.href = `${new URL('style.css', runtimeBase).href}?v=single-tool-ui`;
  document.head.appendChild(css);

  const runtime = document.createElement('script');
  runtime.src = new URL('index.js', runtimeBase).href;
  document.body.appendChild(runtime);
})();
