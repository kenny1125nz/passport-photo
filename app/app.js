import { detectRotationAngle, rotateImageOnCanvas } from './face-detect.js';
import { computeMask, applyMask } from './background-remove.js';
import { getPreset, PRESETS } from './presets.js';

// --- Config (computed from active preset) ---
let cfg = {};

function applyPreset(preset) {
  const MM = preset.dpi / 25.4;
  const W = Math.round(preset.w * MM);
  const H = Math.round(preset.h * MM);
  cfg = {
    preset, W, H, MM,
    MIN_SCALE: 0.1,
    MAX_SCALE: 10,
    CX: W / 2,
    CY: H / 2,
    OVAL_INNER_RX: Math.round(preset.headW[0] * MM / 2),
    OVAL_INNER_RY: Math.round(preset.headH[0] * MM / 2),
    OVAL_OUTER_RX: Math.round(preset.headW[1] * MM / 2),
    OVAL_OUTER_RY: Math.round(preset.headH[1] * MM / 2),
    CROWN_TOP: Math.round(preset.crown[0] * MM),
    CROWN_BOT: Math.round(preset.crown[1] * MM),
  };
  cfg.OVAL_OUTER_CY = cfg.CROWN_TOP + cfg.OVAL_OUTER_RY;
  cfg.OVAL_INNER_CY = cfg.CROWN_BOT + cfg.OVAL_INNER_RY;
  document.documentElement.style.setProperty('--ar', preset.w + '/' + preset.h);
}

applyPreset(getPreset('china'));

// --- DOM refs ---
const uploadSection = document.getElementById('upload-section');
const uploadZone = document.getElementById('upload-zone');
const fileInput = document.getElementById('file-input');
const countrySelect = document.getElementById('country-select');
const cameraBtn = document.getElementById('camera-btn');
const editorSection = document.getElementById('editor-section');
const canvas = document.getElementById('display-canvas');
const ctx = canvas.getContext('2d');
const guideCanvas = document.getElementById('guide-canvas');
const guideCtx = guideCanvas.getContext('2d');
const hud = document.getElementById('hud');
const zoomLabel = document.getElementById('zoom-label');
const zoomInBtn = document.getElementById('zoom-in-btn');
const zoomOutBtn = document.getElementById('zoom-out-btn');
const resetBtn = document.getElementById('reset-btn');
const cropBtn = document.getElementById('crop-btn');
const reselectBtn = document.getElementById('reselect-btn');
const resultSection = document.getElementById('result-section');
const resultCanvas = document.getElementById('result-canvas');
const resultInfo = document.getElementById('result-info');
const downloadBtn = document.getElementById('download-btn');
const backBtn = document.getElementById('back-btn');
const bgThreshold = document.getElementById('bg-threshold');
const bgThresholdLabel = document.getElementById('bg-threshold-label');
const bgThresholdGroup = document.getElementById('bg-threshold-group');
const toast = document.getElementById('toast');
const loadingOverlay = document.getElementById('loading-overlay');
const loadingLabel = document.getElementById('loading-label');
const rotateSlider = document.getElementById('rotate-slider');
const rotateValue = document.getElementById('rotate-value');
const rotateReset = document.getElementById('rotate-reset');

// --- Country selects ---
function populateSelect(sel) {
  for (const p of PRESETS) {
    const opt = document.createElement('option');
    opt.value = p.id; opt.textContent = p.name;
    if (p.id === 'china') opt.selected = true;
    sel.appendChild(opt);
  }
}
populateSelect(countrySelect);


function onCountryChange(sel) {
  const p = getPreset(sel.value);
  applyPreset(p);
  if (resultSection.style.display === 'flex') {
    if (!sourceImage) return;
    const oc = cropImage();
    if (oc) showResult(oc);
  } else {
    sizeCanvases();
    drawGuides();
    if (sourceImage) render();
  }
  countrySelect.value = sel.value;
}

countrySelect.addEventListener('change', function() { onCountryChange(this); });

// --- State ---
let sourceImage = null;
let originalImage = null;
let rotationAngle = 0;
let scale = 1;
let offsetX = cfg.CX;
let offsetY = cfg.CY;
let croppedDataURL = null;
let cachedMask = null;
let cropCanvas = null;
let bgThresholdValue = 0.08;
let toastTimer = null;
let isDragging = false;
let dragStartX = 0, dragStartY = 0, dragOffsetX = 0, dragOffsetY = 0;
let initialPinchDist = 0, initialPinchScale = 1, pinchCenterX = 0, pinchCenterY = 0;
let touchPoints = [];
let resizeTimer;

// --- Helpers ---
function showToast(msg) {
  toast.textContent = msg;
  toast.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => toast.classList.remove('show'), 2500);
}
function showLoading(msg) { loadingLabel.textContent = msg; loadingOverlay.classList.add('active'); }
function hideLoading() { loadingOverlay.classList.remove('active'); }

// --- Canvas sizing ---
function sizeCanvases() {
  const rect = canvas.parentElement.getBoundingClientRect();
  const w = rect.width, h = rect.height;
  if (w === 0 || h === 0) return;
  canvas.width = cfg.W;
  canvas.height = cfg.H;
  canvas.style.width = w + 'px';
  canvas.style.height = h + 'px';
  guideCanvas.width = cfg.W;
  guideCanvas.height = cfg.H;
  guideCanvas.style.width = w + 'px';
  guideCanvas.style.height = h + 'px';
}

// --- Drawing ---
function drawGuides() {
  const g = guideCtx;
  g.clearRect(0, 0, cfg.W, cfg.H);

  g.save();
  g.strokeStyle = '#f59e0b';
  g.lineWidth = 2;
  g.setLineDash([6, 4]);
  g.beginPath(); g.moveTo(0, cfg.CROWN_TOP); g.lineTo(cfg.W, cfg.CROWN_TOP); g.stroke();
  g.beginPath(); g.moveTo(0, cfg.CROWN_BOT); g.lineTo(cfg.W, cfg.CROWN_BOT); g.stroke();
  g.fillStyle = 'rgba(245, 158, 11, 0.08)';
  g.fillRect(0, cfg.CROWN_TOP, cfg.W, cfg.CROWN_BOT - cfg.CROWN_TOP);
  g.restore();

  g.save();
  g.font = '10px -apple-system, sans-serif';
  g.textAlign = 'right';
  g.textBaseline = 'middle';
  g.fillStyle = '#f59e0b';
  g.fillText('CROWN', cfg.W - 12, cfg.CROWN_TOP - 10);
  g.font = '9px -apple-system, sans-serif';
  g.fillText(cfg.preset.crown[0] + '-' + cfg.preset.crown[1] + 'mm', cfg.W - 12, cfg.CROWN_BOT + 12);
  g.restore();

  const chinInner = cfg.OVAL_INNER_CY + cfg.OVAL_INNER_RY;
  const chinOuter = cfg.OVAL_OUTER_CY + cfg.OVAL_OUTER_RY;
  const bx = cfg.W - 24;
  g.save();
  g.strokeStyle = '#8b5cf6';
  g.lineWidth = 1.5;
  g.setLineDash([]);
  g.beginPath(); g.moveTo(bx, chinInner); g.lineTo(bx, chinOuter); g.stroke();
  g.beginPath(); g.moveTo(bx - 6, chinInner); g.lineTo(bx + 6, chinInner); g.stroke();
  g.beginPath(); g.moveTo(bx - 6, chinOuter); g.lineTo(bx + 6, chinOuter); g.stroke();
  g.restore();
  g.save();
  g.font = '9px -apple-system, sans-serif';
  g.textAlign = 'left';
  g.textBaseline = 'middle';
  g.fillStyle = '#8b5cf6';
  g.fillText('CHIN', bx + 10, (chinInner + chinOuter) / 2);
  g.restore();

  g.save();
  g.strokeStyle = '#3b82f6';
  g.lineWidth = 2;
  g.setLineDash([8, 6]);
  g.beginPath();
  g.ellipse(cfg.CX, cfg.OVAL_OUTER_CY, cfg.OVAL_OUTER_RX, cfg.OVAL_OUTER_RY, 0, 0, Math.PI * 2);
  g.stroke();
  g.restore();

  g.save();
  g.strokeStyle = '#10b981';
  g.lineWidth = 2;
  g.setLineDash([8, 6]);
  g.beginPath();
  g.ellipse(cfg.CX, cfg.OVAL_INNER_CY, cfg.OVAL_INNER_RX, cfg.OVAL_INNER_RY, 0, 0, Math.PI * 2);
  g.stroke();
  g.restore();

  g.save();
  g.font = '10px -apple-system, sans-serif';
  g.textAlign = 'center';
  g.fillStyle = '#3b82f6';
  g.fillText('MAX WIDTH ' + cfg.preset.headW[1] + 'mm', cfg.CX, cfg.OVAL_OUTER_CY + cfg.OVAL_OUTER_RY + 18);
  g.fillStyle = '#10b981';
  g.fillText('MIN WIDTH ' + cfg.preset.headW[0] + 'mm', cfg.CX, cfg.OVAL_INNER_CY + cfg.OVAL_INNER_RY - 10);
  g.restore();

  const m = 18, len = 28;
  g.save();
  g.strokeStyle = 'rgba(0,0,0,0.2)';
  g.lineWidth = 2;
  g.setLineDash([]);
  g.beginPath(); g.moveTo(m, m + len); g.lineTo(m, m); g.lineTo(m + len, m); g.stroke();
  g.beginPath(); g.moveTo(cfg.W - m - len, m); g.lineTo(cfg.W - m, m); g.lineTo(cfg.W - m, m + len); g.stroke();
  g.beginPath(); g.moveTo(m, cfg.H - m - len); g.lineTo(m, cfg.H - m); g.lineTo(m + len, cfg.H - m); g.stroke();
  g.beginPath(); g.moveTo(cfg.W - m - len, cfg.H - m); g.lineTo(cfg.W - m, cfg.H - m); g.lineTo(cfg.W - m, cfg.H - m - len); g.stroke();
  g.restore();
}

function render() {
  ctx.clearRect(0, 0, cfg.W, cfg.H);
  if (!sourceImage) return;
  ctx.save();
  ctx.translate(offsetX, offsetY);
  ctx.scale(scale, scale);
  ctx.drawImage(sourceImage, -sourceImage.width / 2, -sourceImage.height / 2);
  ctx.restore();

  const pct = Math.round(scale * 100);
  zoomLabel.textContent = pct + '%';
  hud.classList.add('visible');
  clearTimeout(hud._hideTimer);
  hud._hideTimer = setTimeout(() => hud.classList.remove('visible'), 3000);
}

// --- Rotation ---
function setRotation(angleDeg) {
  rotationAngle = angleDeg * Math.PI / 180;
  rotateSlider.value = angleDeg;
  rotateValue.textContent = angleDeg.toFixed(1) + '\u00B0';
  sourceImage = rotateImageOnCanvas(originalImage, rotationAngle);
  resetTransform();
  render();
}

// --- Transform ---
function resetTransform() {
  if (!sourceImage) return;
  const iA = sourceImage.width / sourceImage.height;
  const cA = cfg.W / cfg.H;
  scale = iA > cA ? cfg.W / sourceImage.width : cfg.H / sourceImage.height;
  offsetX = cfg.CX;
  offsetY = cfg.CY;
}

function zoomAtPoint(newScale, cx, cy) {
  const clamped = Math.min(cfg.MAX_SCALE, Math.max(cfg.MIN_SCALE, newScale));
  offsetX = cx - (cx - offsetX) * (clamped / scale);
  offsetY = cy - (cy - offsetY) * (clamped / scale);
  scale = clamped;
  render();
}

// --- Auto-position ---
function autoPosition(info) {
  const targetHeadPx = cfg.preset.headW[1] * cfg.MM;
  const estimatedHeadW = info.faceW * sourceImage.width * 1.35;
  scale = targetHeadPx / estimatedHeadW + 0.08;

  const faceCenterPx = info.faceCenterX * sourceImage.width;
  offsetX = cfg.CX - (faceCenterPx - sourceImage.width / 2) * scale;

  const chinInner = cfg.OVAL_INNER_CY + cfg.OVAL_INNER_RY;
  const chinOuter = cfg.OVAL_OUTER_CY + cfg.OVAL_OUTER_RY;
  const targetChinY = (chinInner + chinOuter) / 2;
  const chinPx = info.chinY * sourceImage.height;
  offsetY = targetChinY - (chinPx - sourceImage.height / 2) * scale;
}

// --- Image loading ---
function loadImage(file) {
  const reader = new FileReader();
  reader.onload = function(e) {
    const img = new Image();
    img.onload = async function() {
      originalImage = img;
      rotationAngle = 0;
      rotateSlider.value = 0;
      rotateValue.textContent = '0.0\u00B0';

      showLoading('Detecting face...');
      const result = await detectRotationAngle(img);

      if (result) {
        const angleDeg = result.angle * 180 / Math.PI;
        rotationAngle = result.angle;
        rotateSlider.value = +angleDeg.toFixed(1);
        rotateValue.textContent = angleDeg.toFixed(1) + '\u00B0';

        sourceImage = rotateImageOnCanvas(originalImage, rotationAngle);

        showLoading('Positioning head...');
        const rotatedInfo = await detectRotationAngle(sourceImage);
        if (rotatedInfo) {
          autoPosition(rotatedInfo);
        } else {
          resetTransform();
        }
      } else {
        sourceImage = img;
        resetTransform();
        showToast('No face detected, you can adjust manually');
      }

      hideLoading();
      uploadSection.style.display = 'none';
      editorSection.style.display = 'flex';
      resultSection.style.display = 'none';
      sizeCanvases();
      drawGuides();
      render();
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

// --- Crop ---
function cropImage() {
  if (!sourceImage) return null;
  const srcX = -offsetX / scale + sourceImage.width / 2;
  const srcY = -offsetY / scale + sourceImage.height / 2;
  const srcW = cfg.W / scale;
  const srcH = cfg.H / scale;
  const outCanvas = document.createElement('canvas');
  outCanvas.width = cfg.W;
  outCanvas.height = cfg.H;
  const outCtx = outCanvas.getContext('2d');
  outCtx.fillStyle = '#ffffff';
  outCtx.fillRect(0, 0, cfg.W, cfg.H);
  outCtx.drawImage(sourceImage, srcX, srcY, srcW, srcH, 0, 0, cfg.W, cfg.H);
  return outCanvas;
}

function redrawBg() {
  if (!cropCanvas || !cachedMask) return;
  const result = applyMask(cropCanvas, cachedMask, bgThresholdValue);
  croppedDataURL = result.toDataURL('image/jpeg', 0.95);
  const ctx = resultCanvas.getContext('2d');
  ctx.drawImage(result, 0, 0);
}

// --- Show result ---
async function showResult(outputCanvas) {
  cropCanvas = outputCanvas;
  croppedDataURL = outputCanvas.toDataURL('image/jpeg', 0.95);
  const rCanvas = resultCanvas;
  rCanvas.width = cfg.W;
  rCanvas.height = cfg.H;
  const rCtx = rCanvas.getContext('2d');
  rCtx.drawImage(outputCanvas, 0, 0);
  const p = cfg.preset;
  resultInfo.textContent = p.w + ' \u00D7 ' + p.h + ' mm \u00B7 ' + p.dpi + ' DPI \u00B7 ' + cfg.W + ' \u00D7 ' + cfg.H + ' px \u00B7 100% free';
  editorSection.style.display = 'none';
  resultSection.style.display = 'flex';
  bgThresholdGroup.style.display = 'none';
  showLoading('Removing background...');
  try {
    cachedMask = await computeMask(outputCanvas);
    bgThresholdValue = parseFloat(bgThreshold.value) / 100 * 0.1;
    redrawBg();
    bgThresholdGroup.style.display = 'flex';
  } catch (e) {
    console.error(e);
    showToast('Background removal failed');
  }
  hideLoading();
}

function download() {
  if (!croppedDataURL) return;
  const link = document.createElement('a');
  link.download = 'passport_photo.jpg';
  link.href = croppedDataURL;
  link.click();
}

// === Upload handlers ===
uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', (e) => { e.preventDefault(); uploadZone.classList.add('dragover'); });
uploadZone.addEventListener('dragleave', () => { uploadZone.classList.remove('dragover'); });
uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file && file.type.startsWith('image/')) loadImage(file);
  else showToast('Please drop an image file');
});
fileInput.addEventListener('change', () => { if (fileInput.files[0]) loadImage(fileInput.files[0]); });
cameraBtn.addEventListener('click', () => {
  const input = document.createElement('input');
  input.type = 'file';
  input.accept = 'image/*';
  input.capture = 'environment';
  input.onchange = () => { if (input.files[0]) loadImage(input.files[0]); };
  input.click();
});

// === Editor controls ===
zoomInBtn.addEventListener('click', () => zoomAtPoint(scale * 1.25, cfg.CX, cfg.CY));
zoomOutBtn.addEventListener('click', () => zoomAtPoint(scale / 1.25, cfg.CX, cfg.CY));
resetBtn.addEventListener('click', () => { if (!sourceImage) return; resetTransform(); render(); });
cropBtn.addEventListener('click', () => { if (!sourceImage) return; const oc = cropImage(); if (oc) showResult(oc); });
reselectBtn.addEventListener('click', () => {
  sourceImage = null; originalImage = null; cachedMask = null; cropCanvas = null;
  editorSection.style.display = 'none'; uploadSection.style.display = 'flex';
});

// === Rotation slider ===
rotateSlider.addEventListener('input', () => {
  const deg = parseFloat(rotateSlider.value);
  setRotation(deg);
});
rotateReset.addEventListener('click', () => setRotation(0));

// === Result controls ===
downloadBtn.addEventListener('click', download);
backBtn.addEventListener('click', () => {
  resultSection.style.display = 'none';
  bgThresholdGroup.style.display = 'none';
  cachedMask = null;
  cropCanvas = null;
  editorSection.style.display = 'flex';
  sizeCanvases();
  drawGuides();
  render();
});
bgThreshold.addEventListener('input', () => {
  const val = parseInt(bgThreshold.value);
  bgThresholdValue = val / 100 * 0.1;
  bgThresholdLabel.textContent = Math.round(bgThresholdValue * 100) + '%';
  redrawBg();
});

// === Mouse ===
canvas.addEventListener('mousedown', (e) => {
  if (!sourceImage) return;
  isDragging = true;
  dragStartX = e.clientX; dragStartY = e.clientY;
  dragOffsetX = offsetX; dragOffsetY = offsetY;
  canvas.style.cursor = 'grabbing';
});
window.addEventListener('mousemove', (e) => {
  if (!isDragging || !sourceImage) return;
  const rect = canvas.getBoundingClientRect();
  offsetX = dragOffsetX + (e.clientX - dragStartX) / rect.width * cfg.W;
  offsetY = dragOffsetY + (e.clientY - dragStartY) / rect.height * cfg.H;
  render();
});
window.addEventListener('mouseup', () => {
  if (isDragging) { isDragging = false; canvas.style.cursor = 'grab'; }
});
canvas.addEventListener('wheel', (e) => {
  e.preventDefault();
  if (!sourceImage) return;
  const rect = canvas.getBoundingClientRect();
  const cx = (e.clientX - rect.left) / rect.width * cfg.W;
  const cy = (e.clientY - rect.top) / rect.height * cfg.H;
  zoomAtPoint(scale * (e.deltaY < 0 ? 1.08 : 1 / 1.08), cx, cy);
}, { passive: false });
canvas.addEventListener('dragstart', (e) => e.preventDefault());

// === Touch ===
canvas.addEventListener('touchstart', (e) => {
  if (!sourceImage) return;
  const touches = Array.from(e.touches);
  if (touches.length === 1) {
    isDragging = true;
    dragStartX = touches[0].clientX; dragStartY = touches[0].clientY;
    dragOffsetX = offsetX; dragOffsetY = offsetY;
  } else if (touches.length === 2) {
    isDragging = false;
    const t1 = touches[0], t2 = touches[1];
    initialPinchDist = Math.hypot(t2.clientX - t1.clientX, t2.clientY - t1.clientY);
    initialPinchScale = scale;
    const rect = canvas.getBoundingClientRect();
    const mx = (t1.clientX + t2.clientX) / 2, my = (t1.clientY + t2.clientY) / 2;
    pinchCenterX = (mx - rect.left) / rect.width * cfg.W;
    pinchCenterY = (my - rect.top) / rect.height * cfg.H;
  }
}, { passive: true });

canvas.addEventListener('touchmove', (e) => {
  e.preventDefault();
  if (!sourceImage) return;
  const touches = Array.from(e.touches);
  if (touches.length === 1 && isDragging) {
    const rect = canvas.getBoundingClientRect();
    offsetX = dragOffsetX + (touches[0].clientX - dragStartX) / rect.width * cfg.W;
    offsetY = dragOffsetY + (touches[0].clientY - dragStartY) / rect.height * cfg.H;
    render();
  } else if (touches.length === 2 && initialPinchDist > 0) {
    const t1 = touches[0], t2 = touches[1];
    const dist = Math.hypot(t2.clientX - t1.clientX, t2.clientY - t1.clientY);
    const factor = dist / initialPinchDist;
    const ns = Math.min(cfg.MAX_SCALE, Math.max(cfg.MIN_SCALE, initialPinchScale * factor));
    zoomAtPoint(ns, pinchCenterX, pinchCenterY);
  }
}, { passive: false });

canvas.addEventListener('touchend', () => {
  isDragging = false;
  touchPoints = [];
  initialPinchDist = 0;
}, { passive: true });

// === Resize ===
window.addEventListener('resize', () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (editorSection.style.display !== 'none') { sizeCanvases(); drawGuides(); render(); }
  }, 150);
});

// === Init ===
canvas.style.cursor = 'grab';
sizeCanvases();
drawGuides();
