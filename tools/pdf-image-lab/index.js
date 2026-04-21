/* global pdfjsLib, JSZip, jspdf */

pdfjsLib.GlobalWorkerOptions.workerSrc =
  'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

let currentFile = null;
let convertedImages = [];
let currentModalIndex = 0;

// ===== Init =====
(function init() {
  const dropZone = document.getElementById('drop-zone');
  let dragCounter = 0;

  dropZone.addEventListener('dragenter', (e) => {
    e.preventDefault();
    dragCounter++;
    dropZone.classList.add('dragover');
  });

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
  });

  dropZone.addEventListener('dragleave', () => {
    dragCounter--;
    if (dragCounter <= 0) {
      dragCounter = 0;
      dropZone.classList.remove('dragover');
    }
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dragCounter = 0;
    dropZone.classList.remove('dragover');
    const files = Array.from(e.dataTransfer.files);
    if (files.length === 0) return;
    // If no existing results, use first PDF as base; otherwise append
    const firstPdf = files.find(f => f.type === 'application/pdf');
    if (convertedImages.length === 0 && firstPdf) {
      const rest = files.filter(f => f !== firstPdf);
      startConversion(firstPdf).then(() => { if (rest.length) appendFiles(rest); });
    } else if (convertedImages.length === 0) {
      // all images
      appendFiles(files);
    } else {
      appendFiles(files);
    }
  });

  // Click to open file picker
  dropZone.addEventListener('click', () => document.getElementById('file-input').click());

  // Results section drag drop (append mode)
  const resultsSection = document.getElementById('results-section');
  resultsSection.addEventListener('dragover', (e) => { e.preventDefault(); });
  resultsSection.addEventListener('drop', (e) => {
    e.preventDefault();
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) appendFiles(files);
  });

  const fileInput = document.getElementById('file-input');
  fileInput.addEventListener('change', (e) => {
    const files = Array.from(e.target.files);
    if (!files.length) return;
    const firstPdf = files.find(f => f.type === 'application/pdf');
    if (convertedImages.length === 0 && firstPdf) {
      const rest = files.filter(f => f !== firstPdf);
      startConversion(firstPdf).then(() => { if (rest.length) appendFiles(rest); });
    } else {
      appendFiles(files);
    }
    e.target.value = '';
  });

  const replaceInput = document.getElementById('replace-input');
  replaceInput.addEventListener('change', (e) => {
    const files = Array.from(e.target.files);
    if (files.length > 0) appendFiles(files);
    e.target.value = '';
  });

  // Track drag position globally for auto-scroll
  document.addEventListener('dragover', (e) => { currentDragY = e.clientY; });

  document.addEventListener('keydown', (e) => {
    const modal = document.getElementById('image-modal');
    if (!modal.classList.contains('hidden')) {
      if (e.key === 'Escape') closeModal();
      if (e.key === 'ArrowLeft') prevImage();
      if (e.key === 'ArrowRight') nextImage();
    }
  });
})();

// ===== Start Conversion (entry point) =====
async function startConversion(file) {
  if (!file) return showToast('請先選擇 PDF 檔案', 'error');
  currentFile = file;

  document.getElementById('upload-section').style.display = 'none';
  document.getElementById('progress-section').classList.remove('hidden');
  document.getElementById('results-section').classList.add('hidden');
  convertedImages = [];
  document.getElementById('images-grid').innerHTML = '';

  const progressBar = document.getElementById('progress-bar');
  const progressPercent = document.getElementById('progress-percent');
  const progressText = document.getElementById('progress-text');
  const progressPages = document.getElementById('progress-pages');

  try {
    const buf = await currentFile.arrayBuffer();
    const pdf = await pdfjsLib.getDocument({ data: buf }).promise;
    const totalPages = pdf.numPages;

    progressText.textContent = '正在高品質渲染頁面…';

    for (let i = 1; i <= totalPages; i++) {
      progressPages.textContent = `${i} / ${totalPages} 頁`;
      const pct = Math.round((i / totalPages) * 100);
      progressBar.style.width = pct + '%';
      progressPercent.textContent = pct + '%';

      const page = await pdf.getPage(i);
      const viewport = page.getViewport({ scale: 2.5 });
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d', { alpha: false });
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      await page.render({ canvasContext: ctx, viewport }).promise;

      const dataURL = canvas.toDataURL('image/png', 0.92);
      const imgData = {
        page: i,
        dataURL,
        width: canvas.width,
        height: canvas.height,
        filename: `page_${String(i).padStart(2, '0')}.png`,
      };
      convertedImages.push(imgData);
      addCardToGrid(imgData, convertedImages.length - 1);
      await new Promise(r => setTimeout(r, 20));
    }

    progressBar.style.width = '100%';
    progressPercent.textContent = '100%';
    progressText.textContent = '拆解完成！';
    progressPages.textContent = `${totalPages} / ${totalPages} 頁`;

    await new Promise(r => setTimeout(r, 500));
    document.getElementById('progress-section').classList.add('hidden');
    document.getElementById('results-section').classList.remove('hidden');
    document.getElementById('results-summary').textContent =
      `成功拆解 ${totalPages} 頁 · 可拖曳調整順序`;
    showToast(`已成功拆解 ${totalPages} 頁高品質圖片！`, 'success');

  } catch (err) {
    showToast('拆解失敗：' + err.message, 'error');
    document.getElementById('progress-section').classList.add('hidden');
    document.getElementById('upload-section').style.display = 'block';
  }
}

// ===== Grid rendering =====
function buildCardHTML(img, index) {
  const sizeMB = (img.dataURL.length / 1024 / 1024).toFixed(1);
  return `
    <div class="pdf-page-card" draggable="true" data-index="${index}">
      <div class="pdf-card-header">
        <div class="pdf-card-left">
          <span class="pdf-drag-handle" title="拖曳排序">⠿</span>
          <span class="pdf-page-badge">第 ${index + 1} 頁</span>
        </div>
        <div class="pdf-card-actions">
          <button class="pdf-card-icon-btn" title="下載" onclick="event.stopPropagation();downloadSingle(${index})">⬇</button>
          <button class="pdf-card-icon-btn danger" title="刪除" onclick="event.stopPropagation();deleteImage(${index})">🗑</button>
        </div>
      </div>
      <div class="pdf-card-img-wrap" onclick="openModal(${index})">
        <img src="${img.dataURL}" alt="PDF 第 ${index + 1} 頁" />
      </div>
      <div class="pdf-card-footer">
        <span>PNG</span>
        <span class="pdf-card-size">${sizeMB} MB</span>
      </div>
    </div>`;
}

function addCardToGrid(img, index) {
  const grid = document.getElementById('images-grid');
  grid.insertAdjacentHTML('beforeend', buildCardHTML(img, index));
  attachDrag(grid.lastElementChild);
}

function renderGrid() {
  const grid = document.getElementById('images-grid');
  grid.innerHTML = '';
  convertedImages.forEach((img, i) => {
    grid.insertAdjacentHTML('beforeend', buildCardHTML(img, i));
    attachDrag(grid.lastElementChild);
  });
}

// ===== Drag & drop reordering =====
let dragSrcCard = null;
let autoScrollRAF = null;
let currentDragY = 0;

function startAutoScroll() {
  cancelAutoScroll();
  function step() {
    const threshold = 80;
    const speed = 10;
    const h = window.innerHeight;
    if (currentDragY < threshold) window.scrollBy(0, -speed);
    else if (currentDragY > h - threshold) window.scrollBy(0, speed);
    autoScrollRAF = requestAnimationFrame(step);
  }
  autoScrollRAF = requestAnimationFrame(step);
}

function cancelAutoScroll() {
  if (autoScrollRAF) { cancelAnimationFrame(autoScrollRAF); autoScrollRAF = null; }
}

function attachDrag(card) {
  card.addEventListener('dragstart', (e) => {
    dragSrcCard = card;
    e.dataTransfer.effectAllowed = 'move';
    e.dataTransfer.setData('text/plain', card.dataset.index);
    card.classList.add('dragging');
    setTimeout(() => (card.style.opacity = '0.4'), 0);
  });

  card.addEventListener('dragend', () => {
    dragSrcCard = null;
    cancelAutoScroll();
    card.classList.remove('dragging');
    card.style.opacity = '1';
    document.querySelectorAll('.pdf-page-card').forEach(c => c.classList.remove('drag-over'));
    // sync convertedImages order from DOM
    const grid = document.getElementById('images-grid');
    const cards = Array.from(grid.querySelectorAll('.pdf-page-card'));
    const newOrder = cards.map(c => convertedImages[parseInt(c.dataset.index)]);
    convertedImages.length = 0;
    newOrder.forEach(img => convertedImages.push(img));
    // re-index dataset and badges
    cards.forEach((c, i) => {
      c.dataset.index = i;
      const badge = c.querySelector('.pdf-page-badge');
      if (badge) badge.textContent = `第 ${i + 1} 頁`;
    });
    showToast('順序已更新', 'success', 1200);
  });

  card.addEventListener('dragover', (e) => {
    e.preventDefault();
    currentDragY = e.clientY;
    if (!dragSrcCard || dragSrcCard === card) return;
    startAutoScroll();
    const grid = document.getElementById('images-grid');
    const cards = Array.from(grid.querySelectorAll('.pdf-page-card'));
    const srcIdx = cards.indexOf(dragSrcCard);
    const tgtIdx = cards.indexOf(card);
    if (srcIdx < tgtIdx) {
      card.insertAdjacentElement('afterend', dragSrcCard);
    } else {
      card.insertAdjacentElement('beforebegin', dragSrcCard);
    }
    card.classList.add('drag-over');
  });

  card.addEventListener('dragleave', () => card.classList.remove('drag-over'));
  card.addEventListener('drop', (e) => { e.preventDefault(); card.classList.remove('drag-over'); });
}

// ===== Actions =====
function deleteImage(index) {
  if (!confirm('確定要刪除這一頁嗎？')) return;
  convertedImages.splice(index, 1);
  renderGrid();
  if (convertedImages.length === 0) {
    document.getElementById('results-section').classList.add('hidden');
    document.getElementById('upload-section').style.display = 'block';
  }
  showToast('已刪除該頁', 'success', 1500);
}

function downloadSingle(index) {
  const img = convertedImages[index];
  triggerDownload(img.dataURL, img.filename);
  showToast(`已下載 ${img.filename}`, 'success');
}

async function downloadAllAsZip() {
  if (!convertedImages.length) return showToast('沒有可下載的檔案', 'error');
  const toast = showToast('正在打包 ZIP…', 'info', 0);
  try {
    const zip = new JSZip();
    const folder = zip.folder('PDF_圖片結果');
    convertedImages.forEach(img => {
      folder.file(img.filename, img.dataURL.split(',')[1], { base64: true });
    });
    const blob = await zip.generateAsync({ type: 'blob', compression: 'DEFLATE', compressionOptions: { level: 6 } });
    triggerDownload(URL.createObjectURL(blob), `PDF圖片_${today()}.zip`);
    toast.remove();
    showToast(`已成功打包 ${convertedImages.length} 張圖片！`, 'success');
  } catch (err) {
    toast.remove();
    showToast('打包失敗，請重試', 'error');
  }
}

function exportToPDF() {
  if (!convertedImages.length) return showToast('沒有可匯出的圖片', 'error');
  const { jsPDF } = window.jspdf;
  const doc = new jsPDF({ orientation: 'portrait', unit: 'pt', format: 'a4' });
  const pw = doc.internal.pageSize.getWidth();
  const ph = doc.internal.pageSize.getHeight();
  const margin = 25;

  const toast = showToast('正在生成 PDF…', 'info', 0);
  setTimeout(() => {
    try {
      convertedImages.forEach((img, idx) => {
        if (idx > 0) doc.addPage();
        const scale = Math.min((pw - margin * 2) / img.width, (ph - margin * 2) / img.height);
        const w = img.width * scale;
        const h = img.height * scale;
        doc.addImage(img.dataURL, 'PNG', (pw - w) / 2, (ph - h) / 2, w, h);
      });
      doc.save(`重排後_PDF_${today()}.pdf`);
      toast.remove();
      showToast('PDF 匯出成功！', 'success');
    } catch {
      toast.remove();
      showToast('PDF 匯出失敗', 'error');
    }
  }, 200);
}

// ===== Modal =====
function openModal(index) {
  currentModalIndex = index;
  updateModal();
  document.getElementById('image-modal').classList.remove('hidden');
}

function closeModal() {
  document.getElementById('image-modal').classList.add('hidden');
}

function prevImage() {
  currentModalIndex = (currentModalIndex - 1 + convertedImages.length) % convertedImages.length;
  updateModal();
}

function nextImage() {
  currentModalIndex = (currentModalIndex + 1) % convertedImages.length;
  updateModal();
}

function updateModal() {
  const img = convertedImages[currentModalIndex];
  document.getElementById('modal-image').src = img.dataURL;
  document.getElementById('modal-page-text').textContent =
    `第 ${currentModalIndex + 1} 頁 / 共 ${convertedImages.length} 頁`;
  document.getElementById('modal-counter').textContent =
    `${currentModalIndex + 1} / ${convertedImages.length}`;
}

function downloadCurrentModal() {
  const img = convertedImages[currentModalIndex];
  triggerDownload(img.dataURL, img.filename);
  showToast(`已下載 ${img.filename}`, 'success');
}

// ===== Misc helpers =====
function pickNewPdf() {
  document.getElementById('replace-input').click();
}

// ===== Append multiple files (PDF or images) =====
async function appendFiles(files) {
  for (const file of files) {
    if (file.type === 'application/pdf') {
      await appendConversion(file);
    } else if (file.type.startsWith('image/')) {
      await appendImage(file);
    } else {
      showToast(`不支援的檔案格式：${file.name}`, 'error', 2000);
    }
  }
}

// ===== Append a single image file =====
async function appendImage(file) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const dataURL = e.target.result;
      const img = new Image();
      img.onload = () => {
        const index = convertedImages.length;
        const imgData = {
          page: index + 1,
          dataURL,
          width: img.naturalWidth,
          height: img.naturalHeight,
          filename: `page_${String(index + 1).padStart(2, '0')}.png`,
        };
        convertedImages.push(imgData);
        addCardToGrid(imgData, convertedImages.length - 1);
        document.getElementById('results-summary').textContent =
          `共 ${convertedImages.length} 頁 · 可拖曳調整順序`;
        resolve();
      };
      img.src = dataURL;
    };
    reader.readAsDataURL(file);
  });
}

// ===== Append a single PDF =====
async function appendConversion(file) {
  if (!file) return;
  const toast = showToast('正在追加頁面…', 'info', 0);
  try {
    const buf = await file.arrayBuffer();
    const pdf = await pdfjsLib.getDocument({ data: buf }).promise;
    const totalPages = pdf.numPages;
    const startIndex = convertedImages.length;

    for (let i = 1; i <= totalPages; i++) {
      const page = await pdf.getPage(i);
      const viewport = page.getViewport({ scale: 2.5 });
      const canvas = document.createElement('canvas');
      const ctx = canvas.getContext('2d', { alpha: false });
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      await page.render({ canvasContext: ctx, viewport }).promise;

      const dataURL = canvas.toDataURL('image/png', 0.92);
      const imgData = {
        page: startIndex + i,
        dataURL,
        width: canvas.width,
        height: canvas.height,
        filename: `page_${String(startIndex + i).padStart(2, '0')}.png`,
      };
      convertedImages.push(imgData);
      addCardToGrid(imgData, convertedImages.length - 1);
      await new Promise(r => setTimeout(r, 20));
    }

    toast.remove();
    document.getElementById('results-summary').textContent =
      `共 ${convertedImages.length} 頁 · 可拖曳調整順序`;
    showToast(`已追加 ${totalPages} 頁！`, 'success');
  } catch (err) {
    toast.remove();
    showToast('追加失敗：' + err.message, 'error');
  }
}

function resetAll() {
  currentFile = null;
  convertedImages = [];
  currentModalIndex = 0;
  document.getElementById('upload-section').style.display = 'block';
  document.getElementById('progress-section').classList.add('hidden');
  document.getElementById('results-section').classList.add('hidden');
  document.getElementById('drop-zone').classList.remove('hidden');
  document.getElementById('images-grid').innerHTML = '';
  document.getElementById('file-input').value = '';
  closeModal();
}

function formatFileSize(bytes) {
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(1024));
  return (bytes / Math.pow(1024, i)).toFixed(1) + ' ' + units[i];
}

function today() {
  return new Date().toISOString().slice(0, 10);
}

function triggerDownload(url, filename) {
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
}

function showToast(message, type = 'success', duration = 2800) {
  const el = document.createElement('div');
  el.className = `pdf-toast ${type}`;
  el.textContent = message;
  document.body.appendChild(el);
  if (duration > 0) {
    setTimeout(() => {
      el.style.transition = 'opacity 0.3s, transform 0.3s';
      el.style.opacity = '0';
      el.style.transform = 'translateY(12px)';
      setTimeout(() => el.remove(), 300);
    }, duration);
  }
  return el;
}
