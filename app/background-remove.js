let ortRef = null;
let ortSession = null;
let sessionLoading = null;

async function loadORT() {
  if (!ortRef) {
    const m = await import(
      'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.0/dist/ort.bundle.min.mjs'
    );
    ortRef = m;
  }
  return ortRef;
}

async function getSession() {
  if (sessionLoading) return sessionLoading;
  sessionLoading = (async () => {
    const ort = await loadORT();
    ortSession = await ort.InferenceSession.create('u2netp.onnx', {
      executionProviders: ['wasm']
    });
    return ortSession;
  })();
  return sessionLoading;
}

function preprocess(img) {
  const W = 320, H = 320;
  const canvas = document.createElement('canvas');
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(img, 0, 0, W, H);
  const imageData = ctx.getImageData(0, 0, W, H);
  const pixels = imageData.data;

  const float32 = new Float32Array(3 * W * H);
  for (let i = 0; i < W * H; i++) {
    const offset = i * 4;
    float32[i] = pixels[offset] / 255.0;
    float32[W * H + i] = pixels[offset + 1] / 255.0;
    float32[2 * W * H + i] = pixels[offset + 2] / 255.0;
  }
  return float32;
}

function postprocess(maskData, origW, origH) {
  const W = 320, H = 320;
  const canvas = document.createElement('canvas');
  canvas.width = origW;
  canvas.height = origH;
  const ctx = canvas.getContext('2d');

  const tempCanvas = document.createElement('canvas');
  tempCanvas.width = W;
  tempCanvas.height = H;
  const tempCtx = tempCanvas.getContext('2d');
  const imageData = tempCtx.createImageData(W, H);

  for (let i = 0; i < W * H; i++) {
    const v = Math.round(maskData[i] * 255);
    imageData.data[i * 4] = v;
    imageData.data[i * 4 + 1] = v;
    imageData.data[i * 4 + 2] = v;
    imageData.data[i * 4 + 3] = 255;
  }
  tempCtx.putImageData(imageData, 0, 0);

  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(tempCanvas, 0, 0, origW, origH);
  return canvas;
}

export function applyMask(origCanvas, maskCanvas, threshold) {
  const w = origCanvas.width, h = origCanvas.height;
  const output = document.createElement('canvas');
  output.width = w; output.height = h;
  const ctx = output.getContext('2d');
  ctx.drawImage(origCanvas, 0, 0);

  const imgData = ctx.getImageData(0, 0, w, h);
  const px = imgData.data;
  const mCtx = maskCanvas.getContext('2d');
  const mData = mCtx.getImageData(0, 0, w, h);
  const mp = mData.data;

  const feather = threshold * 1.5;

  for (let i = 0; i < w * h; i++) {
    const o = i * 4;
    const m = mp[o] / 255;
    if (m < threshold) {
      px[o] = 255; px[o+1] = 255; px[o+2] = 255;
    } else if (m < threshold + feather) {
      const b = (m - threshold) / feather;
      px[o] = Math.round(px[o] * b + 255 * (1 - b));
      px[o+1] = Math.round(px[o+1] * b + 255 * (1 - b));
      px[o+2] = Math.round(px[o+2] * b + 255 * (1 - b));
    }
  }
  ctx.putImageData(imgData, 0, 0);
  return output;
}

export async function computeMask(src) {
  const w = src.width, h = src.height;
  const origCanvas = document.createElement('canvas');
  origCanvas.width = w; origCanvas.height = h;
  const origCtx = origCanvas.getContext('2d');
  origCtx.drawImage(src, 0, 0);

  const float32 = preprocess(origCanvas);
  const session = await getSession();

  const feeds = {};
  feeds[session.inputNames[0]] = new ortRef.Tensor('float32', float32, [1, 3, 320, 320]);

  const result = await session.run(feeds);
  const outputName = session.outputNames[0];
  const outputData = result[outputName].data;

  return postprocess(outputData, w, h);
}

export async function removeBackground(src, threshold) {
  const mask = await computeMask(src);
  return applyMask(src, mask, threshold || 0.08);
}
