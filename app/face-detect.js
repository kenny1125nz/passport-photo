let faceLandmarkerPromise = null;

async function getFaceLandmarker() {
  if (faceLandmarkerPromise) return faceLandmarkerPromise;
  const CDN = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35';
  faceLandmarkerPromise = (async () => {
    const { FaceLandmarker, FilesetResolver } = await import(CDN + '/vision_bundle.mjs');
    const vision = await FilesetResolver.forVisionTasks(CDN + '/wasm/');
    return await FaceLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
        delegate: 'GPU'
      },
      runningMode: 'IMAGE',
      numFaces: 1
    });
  })();
  return faceLandmarkerPromise;
}

export async function detectRotationAngle(img) {
  let faceLandmarker;
  try {
    faceLandmarker = await getFaceLandmarker();
  } catch (e) {
    console.warn('MediaPipe load failed:', e);
    return null;
  }
  let result;
  try {
    result = faceLandmarker.detect(img);
  } catch (e) {
    console.warn('Face detection failed:', e);
    return null;
  }
  if (!result.faceLandmarks || result.faceLandmarks.length === 0) {
    return null;
  }
  const lm = result.faceLandmarks[0];
  const lx = (lm[33].x + lm[133].x) / 2;
  const ly = (lm[33].y + lm[133].y) / 2;
  const rx = (lm[362].x + lm[263].x) / 2;
  const ry = (lm[362].y + lm[263].y) / 2;
  const angle = Math.atan2(ry - ly, rx - lx);

  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const p of lm) {
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }

  return {
    angle,
    faceCenterX: (minX + maxX) / 2,
    faceW: maxX - minX,
    faceH: maxY - minY,
    chinY: lm[152].y
  };
}

export function rotateImageOnCanvas(img, angleRad) {
  if (!angleRad || Math.abs(angleRad) < 0.0087) return img;
  const rad = Math.abs(angleRad);
  const w = img.width;
  const h = img.height;
  const newW = Math.ceil(Math.abs(w * Math.cos(rad)) + Math.abs(h * Math.sin(rad)));
  const newH = Math.ceil(Math.abs(w * Math.sin(rad)) + Math.abs(h * Math.cos(rad)));
  const canvas = document.createElement('canvas');
  canvas.width = newW;
  canvas.height = newH;
  const c = canvas.getContext('2d');
  c.imageSmoothingEnabled = true;
  c.imageSmoothingQuality = 'high';
  c.translate(newW / 2, newH / 2);
  c.rotate(-angleRad);
  c.drawImage(img, -w / 2, -h / 2);
  return canvas;
}
