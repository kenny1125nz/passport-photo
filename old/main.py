import argparse
import sys
import urllib.request
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks.python.vision import FaceLandmarker
from PIL import Image

from u2net import U2NetInference

MODEL_DIR = Path.home() / ".cache" / "passport-photo"
FACE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
FACE_MODEL_PATH = MODEL_DIR / "face_landmarker.task"

MM_PER_INCH = 25.4
DPI = 600
PX_PER_MM = DPI / MM_PER_INCH

TARGET_W_PX = round(33 * PX_PER_MM)
TARGET_H_PX = round(48 * PX_PER_MM)

HEAD_W_MIN_PX = round(15 * PX_PER_MM)
HEAD_W_MAX_PX = round(22 * PX_PER_MM)
HEAD_H_MIN_PX = round(28 * PX_PER_MM)
HEAD_H_MAX_PX = round(33 * PX_PER_MM)

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

LEFT_EAR = 234
RIGHT_EAR = 454
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263
LEFT_EYE_INNER = 133
RIGHT_EYE_INNER = 362


def _ensure_face_model():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if not FACE_MODEL_PATH.exists():
        print("Downloading face landmark model …")
        urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)
        print("Download complete.")
    return str(FACE_MODEL_PATH)


class FaceDetector:
    def __init__(self):
        model_path = _ensure_face_model()
        self.landmarker = FaceLandmarker.create_from_model_path(model_path)

    def detect(self, image_rgb: np.ndarray):
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        result = self.landmarker.detect(mp_img)
        if not result or not result.face_landmarks:
            return None
        return result.face_landmarks[0]

    @staticmethod
    def get_head_bbox(landmarks, img_w, img_h):
        pts = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in FACE_OVAL]
        xs, ys = zip(*pts)
        return min(xs), min(ys), max(xs), max(ys)

    @staticmethod
    def get_head_dimensions(landmarks, img_w, img_h):
        x1, y1, x2, y2 = FaceDetector.get_head_bbox(landmarks, img_w, img_h)
        return x2 - x1, y2 - y1

    @staticmethod
    def get_eye_y(landmarks, img_h):
        ly = (landmarks[LEFT_EYE_OUTER].y + landmarks[LEFT_EYE_INNER].y) / 2
        ry = (landmarks[RIGHT_EYE_OUTER].y + landmarks[RIGHT_EYE_INNER].y) / 2
        return ((ly + ry) / 2) * img_h

    @staticmethod
    def get_face_center(landmarks, img_w, img_h):
        x1, y1, x2, y2 = FaceDetector.get_head_bbox(landmarks, img_w, img_h)
        return (x1 + x2) / 2, (y1 + y2) / 2

    @staticmethod
    def check_ears(landmarks, margin=0.02):
        left = landmarks[LEFT_EAR].x > margin and landmarks[LEFT_EAR].x < 1 - margin
        right = landmarks[RIGHT_EAR].x > margin and landmarks[RIGHT_EAR].x < 1 - margin
        return left, right

    @staticmethod
    def estimate_tilt(landmarks):
        lx, ly = landmarks[LEFT_EYE_OUTER].x, landmarks[LEFT_EYE_OUTER].y
        rx, ry = landmarks[RIGHT_EYE_OUTER].x, landmarks[RIGHT_EYE_OUTER].y
        return np.degrees(np.arctan2(ry - ly, rx - lx))

    @staticmethod
    def mouth_open(landmarks, threshold=0.04):
        upper = landmarks[13]
        lower = landmarks[14]
        return abs(upper.y - lower.y) > threshold


# ---- Reusable functional API ----

def detect_face(image_rgb: np.ndarray):
    """Return landmarks for the first detected face, or None."""
    d = FaceDetector()
    return d.detect(image_rgb)


def compute_initial_crop(landmarks, img_w, img_h):
    """Compute crop box (cx, cy, scale) that fits the head to spec."""
    head_w, head_h = FaceDetector.get_head_dimensions(landmarks, img_w, img_h)
    eye_y = FaceDetector.get_eye_y(landmarks, img_h)
    cx, cy = FaceDetector.get_face_center(landmarks, img_w, img_h)

    sh_min = HEAD_H_MIN_PX / head_h
    sh_max = HEAD_H_MAX_PX / head_h
    scale = (sh_min + sh_max) / 2

    return cx, cy, eye_y, scale, head_w, head_h


def auto_rotate(image_rgb: np.ndarray, landmarks) -> tuple:
    """Rotate image so both eyes are level. Returns (rotated_img, angle_deg)."""
    h, w = image_rgb.shape[:2]

    left_x = (landmarks[LEFT_EYE_OUTER].x + landmarks[LEFT_EYE_INNER].x) / 2 * w
    left_y = (landmarks[LEFT_EYE_OUTER].y + landmarks[LEFT_EYE_INNER].y) / 2 * h
    right_x = (landmarks[RIGHT_EYE_OUTER].x + landmarks[RIGHT_EYE_INNER].x) / 2 * w
    right_y = (landmarks[RIGHT_EYE_OUTER].y + landmarks[RIGHT_EYE_INNER].y) / 2 * h

    angle = np.degrees(np.arctan2(right_y - left_y, right_x - left_x))
    if abs(angle) < 0.5:
        return image_rgb, 0.0

    center = (w / 2, h / 2)
    rot = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        image_rgb, rot, (w, h),
        flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REFLECT,
    )
    return rotated, angle


def crop_to_canvas(image_rgb: np.ndarray, cx, cy, eye_y, scale) -> np.ndarray:
    """Crop a region from the scaled image and place on a 780x1134 white canvas."""
    h, w = image_rgb.shape[:2]
    scaled = cv2.resize(image_rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
    sh, sw = scaled.shape[:2]

    target_eye_y = TARGET_H_PX * 0.32
    target_cx = TARGET_W_PX / 2

    offset_x = int(target_cx - cx * scale)
    offset_y = int(target_eye_y - eye_y * scale)

    canvas = np.full((TARGET_H_PX, TARGET_W_PX, 3), 255, dtype=np.uint8)

    sx = max(0, -offset_x)
    sy = max(0, -offset_y)
    dx = max(0, offset_x)
    dy = max(0, offset_y)
    cw = min(sw - sx, TARGET_W_PX - dx)
    ch = min(sh - sy, TARGET_H_PX - dy)

    if cw > 0 and ch > 0:
        canvas[dy : dy + ch, dx : dx + cw] = scaled[sy : sy + ch, sx : sx + cw]

    return canvas


def compute_face_weight_mask(h, w, landmarks, img_w, img_h):
    """Create a spatial weight map: 1.0 at face center, decaying to 0.0 at edges."""
    bbox = FaceDetector.get_head_bbox(landmarks, img_w, img_h)
    fc_x = (bbox[0] + bbox[2]) / 2
    fc_y = (bbox[1] + bbox[3]) / 2
    face_w = bbox[2] - bbox[0]
    face_h = bbox[3] - bbox[1]
    face_r = max(face_w, face_h) * 1.5

    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.sqrt((xx - fc_x) ** 2 + (yy - fc_y) ** 2)
    weight = np.clip(1.0 - dist / face_r, 0.0, 1.0)
    return weight.astype(np.float32)


def compute_mask_face_aware(
    image_rgb: np.ndarray,
    landmarks,
    bg_model: U2NetInference,
    base_threshold: float = 0.3,
    feather_radius: int = 5,
    face_weight: float = 0.5,
) -> np.ndarray:
    """Compute a segmentation mask using U2Net + face-aware spatial prior."""
    h, w = image_rgb.shape[:2]
    img_w, img_h = w, h

    mask_pil = bg_model.predict(Image.fromarray(image_rgb))
    saliency = np.array(mask_pil, dtype=np.float32) / 255.0

    fw = compute_face_weight_mask(h, w, landmarks, img_w, img_h)
    effective_threshold = base_threshold * (1.0 - face_weight * fw)

    mask = (saliency > effective_threshold).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    if feather_radius > 0:
        mask = mask.astype(np.float32)
        mask = cv2.GaussianBlur(mask, (feather_radius * 2 + 1, feather_radius * 2 + 1), 0)
        mask = np.clip(mask, 0, 255).astype(np.uint8)

    return mask


def composite(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply mask to image and set background to white."""
    mask_f = mask.astype(np.float32) / 255.0
    result = image_rgb.astype(np.float32) * mask_f[..., None] + 255.0 * (1.0 - mask_f[..., None])
    return np.clip(result, 0, 255).astype(np.uint8)


# ---- CLI (kept for backward compatibility) ----

class Validator:
    @staticmethod
    def validate(image: np.ndarray, head_w: float, head_h: float) -> list[str]:
        issues = []
        h, w = image.shape[:2]
        if h != TARGET_H_PX or w != TARGET_W_PX:
            issues.append(f"Dimensions: {w}x{h}, expected {TARGET_W_PX}x{TARGET_H_PX}")
        if not (HEAD_W_MIN_PX <= head_w <= HEAD_W_MAX_PX):
            issues.append(f"Head width: {head_w:.0f}px, range {HEAD_W_MIN_PX}-{HEAD_W_MAX_PX}")
        if not (HEAD_H_MIN_PX <= head_h <= HEAD_H_MAX_PX):
            issues.append(f"Head height: {head_h:.0f}px, range {HEAD_H_MIN_PX}-{HEAD_H_MAX_PX}")
        bg_mask = np.all(image > 250, axis=2)
        if bg_mask.any() and image[bg_mask].mean() < 250:
            issues.append("Background not pure white")
        return issues


def process(
    input_path: str,
    output_path: str,
    validate_only: bool,
    verbose: bool,
    device: str,
) -> bool:
    img_bgr = cv2.imread(str(input_path))
    if img_bgr is None:
        print(f"Error: cannot read '{input_path}'")
        return False

    orig_h, orig_w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    landmarks = detect_face(img_rgb)
    if landmarks is None:
        print("Error: no face detected.")
        return False

    left_ear, right_ear = FaceDetector.check_ears(landmarks)
    if not left_ear:
        print("Warning: left ear may not be visible")
    if not right_ear:
        print("Warning: right ear may not be visible")

    tilt = FaceDetector.estimate_tilt(landmarks)
    if abs(tilt) > 10:
        print(f"Warning: head tilted ({tilt:.1f}°)")

    if FaceDetector.mouth_open(landmarks):
        print("Warning: mouth appears open")

    cx, cy, eye_y, scale, head_w, head_h = compute_initial_crop(landmarks, orig_w, orig_h)

    if validate_only:
        print(f"Source: {orig_w}x{orig_h}px")
        print(f"Head size in source: {head_w:.0f} x {head_h:.0f}px")
        print(f"Output at {DPI}dpi: {TARGET_W_PX}x{TARGET_H_PX}px")
        print(f"Required head: {HEAD_W_MIN_PX}-{HEAD_W_MAX_PX}px wide, {HEAD_H_MIN_PX}-{HEAD_H_MAX_PX}px tall")
        sw_min, sw_max = HEAD_W_MIN_PX / head_w, HEAD_W_MAX_PX / head_w
        sh_min, sh_max = HEAD_H_MIN_PX / head_h, HEAD_H_MAX_PX / head_h
        feasible = max(sw_min, sh_min) <= min(sw_max, sh_max)
        print(f"Scale range: w=[{sw_min:.2f}, {sw_max:.2f}]  h=[{sh_min:.2f}, {sh_max:.2f}]")
        print(f"Compliant: {'YES' if feasible else 'NO'}")
        return True

    if verbose:
        print("Processing …")

    bg_model = U2NetInference(device=device)

    cropped = crop_to_canvas(img_rgb, cx, cy, eye_y, scale)
    mask = compute_mask_face_aware(img_rgb, landmarks, bg_model)
    mask_cropped = crop_to_canvas(
        np.stack([mask] * 3, axis=-1), cx, cy, eye_y, scale
    )[..., 0]
    result = composite(cropped, mask_cropped)

    final_hw = head_w * scale
    final_hh = head_h * scale

    if verbose:
        print(f"Scale: {scale:.4f}")
        print(f"Head size: {final_hw:.0f} x {final_hh:.0f}px")

    Image.fromarray(result).save(output_path, dpi=(DPI, DPI))
    print(f"Saved: {output_path}")

    issues = Validator.validate(result, final_hw, final_hh)
    if issues:
        print("Warnings:")
        for i in issues:
            print(f"  - {i}")

    return True


def main():
    parser = argparse.ArgumentParser(description="Passport photo compliance tool")
    parser.add_argument("-i", "--input", required=True)
    parser.add_argument("-o", "--output", default="output.jpg")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA not available, falling back to CPU")
        args.device = "cpu"

    ok = process(
        input_path=args.input,
        output_path=args.output,
        validate_only=args.validate_only,
        verbose=args.verbose,
        device=args.device,
    )
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
