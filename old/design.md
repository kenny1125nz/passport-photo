# Passport Photo Compliance Application - Design Document

## 1. Requirements Summary

| Parameter | Specification |
|-----------|--------------|
| Print size | 33 mm (width) x 48 mm (height) |
| Resolution | 600 dpi |
| Head width | 15 mm - 22 mm |
| Head height | 28 mm - 33 mm |
| Background | Pure white |
| Expression | Neutral |
| Ears | Both visible |
| Face coverage | Crown to chin, clearly visible |
| Prohibited | Heavy jewelry, uniforms, distracting objects |

**Pixel math at 600 dpi (rounded):**
- Photo: 780 x 1134 px
- Head width: 354-520 px | Head height: 661-780 px

---

## 2. Architecture Overview

```
                      ┌─────────────────────────────┐
  Upload Photo ──────▶  Step 1: Interactive Crop     │
                      │  - Auto-detect face          │
                      │  - Show crop frame overlay   │
                      │  - User adjusts zoom/pan     │
                      │  - Realtime guide overlay    │
                      └─────────────┬───────────────┘
                                    │ cropped 780x1134
                                    ▼
                      ┌─────────────────────────────┐
                      │  Step 2: Background Removal  │
                      │  - Face-aware U2Net seg      │
                      │  - Sliders: threshold,       │
                      │    feather, face_weight      │
                      │  - Realtime preview          │
                      │  - User confirms result      │
                      └─────────────┬───────────────┘
                                    │ final JPEG
                                    ▼
                           Saved at 600dpi
```

### Why Interactive?

The automated one-shot pipeline fails because:
1. **Crop is ambiguous** — the exact crop/zoom that satisfies the spec depends on the photo's composition; automatic heuristics often produce wrong head size
2. **U2Net alone misses shoulders** — salient object detection segments the face/head but not shoulders; needs human-tuned parameters
3. **Face-aware prior is essential** — using face location as a spatial prior dramatically improves segmentation quality

---

## 3. Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **UI Framework** | Gradio (web-based, runs locally) | Image upload, sliders, real-time preview, cross-platform |
| **Face Detection** | MediaPipe FaceLandmarker | 468 landmarks, fast, CPU |
| **Background Removal** | U2Net (PyTorch, CUDA) | Pretrained salient object detection |
| **Face-Aware Mask** | OpenCV distance transform + custom weighting | Uses face bounding box to modulate threshold spatially |
| **Image Processing** | OpenCV + Pillow | Resize, composite, saving with DPI metadata |
| **GPU** | RTX 3060 + CUDA (via PyTorch) | U2Net inference |

---

## 4. Interactive Workflow

### Step 1 — Interactive Crop (Gradio ImageEditor)

1. User uploads a photo
2. App runs MediaPipe face detection
3. App computes ideal crop frame from face landmarks (head at ~15-22mm wide, eyes at ~32% from top)
4. Crop frame is shown as an overlay on the image
5. User can:
   - **Pan**: click and drag to move the frame
   - **Zoom**: scroll to resize (frame corresponds to 780x1134 output)
6. Guide overlay shows:
   - Green rectangle where the head should fit
   - Eye position line
7. The cropped preview updates in real-time

**Output of Step 1:** 780x1134 px image (cropped and resized)

### Step 2 — Interactive Background Removal (Gradio)

1. The cropped image is displayed
2. U2Net mask is computed on the full-resolution uncropped image (for best context), then cropped and scaled to match
3. Three sliders let the user tune the result in real-time:
   - **Mask Threshold** (0.0-1.0): controls how aggressive the U2Net mask is (lower = more inclusive)
   - **Feather** (0-20px): smooths the mask edge transition
   - **Face Weight** (0.0-1.0): how much to favor face-area pixels (higher = face is more likely to stay opaque)
4. A live preview shows the composited result (person on white background)
5. User clicks "Confirm & Save" to export

---

## 5. Face-Aware Background Removal

The U2Net saliency map is combined with a face-location prior:

1. **Face bounding box** from MediaPipe landmarks
2. **Distance transform** — create a heatmap where values are 1.0 at the face center, decaying to 0.0 at the image edges
3. **Adaptive threshold** — for each pixel, the effective threshold is:
   ```
   effective_threshold = base_threshold * (1 - face_weight * distance_value)
   ```
   - Near face: lower threshold → more likely to be kept as "person"
   - Far from face: higher threshold → more likely to be removed as "background"
4. The final mask is: `(model_saliency > effective_threshold) | body_prior`

This ensures:
- The face is always fully preserved
- The shoulders are more likely to be included (they're near the face)
- The background far from the face is cleanly removed

---

## 6. File Structure

```
passport-photo/
  app.py               # Gradio web interface (entry point)
  main.py              # Core processing functions (refactored for reuse)
  u2net.py             # U2Net model definition + inference
  requirements.txt     # dependencies
  spec.md              # Requirements specification
  design.md            # This file
```

---

## 7. Dependencies

```
torch>=2.0.0
mediapipe>=0.10.35
opencv-python>=4.8.0
Pillow>=10.0.0
numpy>=1.24.0
gradio>=5.0.0
```

### GPU Setup

```
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

---

## 8. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Gradio over tkinter/Qt | Cross-platform, no native deps, real-time preview is built-in, runs in browser |
| U2Net on full image, then crop mask | Model sees full context (better body segmentation), cropping happens in image space |
| Face-aware thresholding | Simple, fast, no extra model needed; uses spatial prior from face detection |
| Body prior (luminance-based) | Recovers shoulders that U2Net misses; combined with face-aware masking |
| Sliders for threshold/feather/weight | Gives user control over the trade-off between coverage and edge quality |

---

## 9. Limitations

- **One face only** — multiple faces are rejected
- **Expression check** — MediaPipe can check mouth openness but not subtle expressions
- **Jewelry/uniform detection** — not automated
- **U2Net model download** — ~170 MB on first run
- **Crop frame accuracy** — depends on user's manual adjustment
