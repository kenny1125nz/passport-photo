import numpy as np
import cv2
import gradio as gr
from PIL import Image

from main import (
    detect_face, compute_initial_crop, crop_to_canvas, composite, auto_rotate,
    FaceDetector, DPI, TARGET_W_PX, TARGET_H_PX,
    HEAD_W_MIN_PX, HEAD_W_MAX_PX, HEAD_H_MIN_PX, HEAD_H_MAX_PX,
    FACE_OVAL,
    LEFT_EYE_OUTER, LEFT_EYE_INNER, RIGHT_EYE_OUTER, RIGHT_EYE_INNER,
)

HEAD_REF_W = int((HEAD_W_MIN_PX + HEAD_W_MAX_PX) / 2)
HEAD_REF_H = int((HEAD_H_MIN_PX + HEAD_H_MAX_PX) / 2)


def shoulder_segs_from_points(n_x, n_y, l_x, l_y, r_x, r_y):
    cw = TARGET_W_PX
    left_seg = ((l_x, l_y), (n_x, n_y))
    right_seg = ((n_x, n_y), (r_x, r_y))
    above_poly = np.array([
        [0, 0], [l_x, l_y], [n_x, n_y], [r_x, r_y], [cw, 0],
    ], dtype=np.int32)
    return left_seg, right_seg, above_poly


def default_shoulder_points(chin_cy, face_h, tilt_deg=0.0):
    cw, ch = TARGET_W_PX, TARGET_H_PX
    n_x = int(cw / 2)
    n_y = min(ch, max(0, int(chin_cy + face_h * 0.2)))
    drop = int(face_h * 0.25)
    slope_rad = np.radians(tilt_deg) * 0.4
    half_w = int(face_h * 0.9)
    l_x = max(0, n_x - half_w)
    l_y = min(ch, max(0, n_y + drop - int(half_w * np.sin(slope_rad))))
    r_x = min(cw, n_x + half_w)
    r_y = min(ch, max(0, n_y + drop + int(half_w * np.sin(slope_rad))))
    return n_x, n_y, l_x, l_y, r_x, r_y


def crop_and_make_mask(orig_img_rgb, landmarks, cx, eye_y, scale,
                       h_min, h_max, s_min, s_max, v_min, v_max, feather,
                       n_x, n_y, l_x, l_y, r_x, r_y):
    from main import crop_to_canvas

    cropped = crop_to_canvas(orig_img_rgb, cx, 0, eye_y, scale)

    h, w = orig_img_rgb.shape[:2]
    def tc(xo, yo):
        return (int((xo - cx) * scale + TARGET_W_PX / 2),
                int((yo - eye_y) * scale + TARGET_H_PX * 0.32))
    face_pts = np.array(
        [[tc(landmarks[i].x * w, landmarks[i].y * h)] for i in FACE_OVAL],
        dtype=np.int32,
    )
    face_mask = np.zeros((TARGET_H_PX, TARGET_W_PX), dtype=np.uint8)
    cv2.fillPoly(face_mask, face_pts, 255)

    _, _, above_poly = shoulder_segs_from_points(n_x, n_y, l_x, l_y, r_x, r_y)

    always1 = np.zeros((TARGET_H_PX, TARGET_W_PX), dtype=np.uint8)
    cv2.fillPoly(always1, [above_poly], 255)
    always1 = cv2.bitwise_not(always1)
    always1 = cv2.bitwise_or(always1, face_mask)

    hsv = cv2.cvtColor(cropped, cv2.COLOR_RGB2HSV)
    h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    if h_min <= h_max:
        h_ok = (h_ch >= h_min) & (h_ch <= h_max)
    else:
        h_ok = (h_ch >= h_min) | (h_ch <= h_max)
    hsv_ok = h_ok & (s_ch >= s_min) & (s_ch <= s_max) & (v_ch >= v_min) & (v_ch <= v_max)
    hsv_mask = hsv_ok.astype(np.uint8) * 255
    hsv_mask = cv2.bitwise_and(hsv_mask, cv2.bitwise_not(always1))

    mask = cv2.bitwise_or(always1, hsv_mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    if feather > 0:
        mask = cv2.GaussianBlur(mask.astype(np.float32),
                                (feather * 2 + 1, feather * 2 + 1), 0)
        mask = np.clip(mask, 0, 255).astype(np.uint8)
    mask = cv2.bitwise_or(mask, face_mask)

    result = composite(cropped, mask)
    return cropped, mask, result


def save_fn(img_rgb):
    if img_rgb is None:
        return None
    path = "passport_photo_output.jpg"
    Image.fromarray(img_rgb).save(path, dpi=(600, 600))
    return path


CSS = """
.gr-box {border-radius: 12px !important;}
"""

with gr.Blocks(title="Passport Photo Tool") as app:
    gr.Markdown("# Passport Photo Tool")
    gr.Markdown("Upload a photo. Adjust **Pan** / **Zoom** sliders (no preview needed — just estimate), then click **Confirm**. The server will detect the face, crop, and remove the background in one step.")

    photo = gr.Image(label="Photo", type="numpy", height=540)

    with gr.Row():
        zoom = gr.Slider(0.3, 2.5, 1.0, step=0.05, label="Zoom", scale=1)
        pan_x = gr.Slider(-500, 500, 0, step=5, label="Pan X", scale=1)
        pan_y = gr.Slider(-500, 500, 0, step=5, label="Pan Y", scale=1)
        mode = gr.Markdown("### 🟢 Waiting for upload", scale=2)

    confirm = gr.Button("✅ Confirm Crop & Remove Background", variant="primary")

    with gr.Group(visible=False) as bg_group:
        gr.Markdown("### Fine-tune Background")
        with gr.Row():
            with gr.Column(scale=1):
                h_min = gr.Number(0, label="H min", minimum=0, maximum=179)
                h_max = gr.Number(25, label="H max", minimum=0, maximum=179)
                s_min = gr.Number(20, label="S min", minimum=0, maximum=255)
                s_max = gr.Number(255, label="S max", minimum=0, maximum=255)
                v_min = gr.Number(30, label="V min", minimum=0, maximum=255)
                v_max = gr.Number(255, label="V max", minimum=0, maximum=255)
                feather = gr.Slider(0, 15, 3, step=1, label="Feather")
            with gr.Column(scale=1):
                n_x = gr.Number(390, label="Neck X", minimum=0, maximum=780)
                n_y = gr.Number(600, label="Neck Y", minimum=0, maximum=1134)
                l_x = gr.Number(100, label="Left X", minimum=0, maximum=780)
                l_y = gr.Number(700, label="Left Y", minimum=0, maximum=1134)
                r_x = gr.Number(680, label="Right X", minimum=0, maximum=780)
                r_y = gr.Number(700, label="Right Y", minimum=0, maximum=1134)
        save = gr.Button("💾 Save (600 dpi JPEG)", variant="primary")
        saved = gr.Text(label="", visible=False)

    st_img = gr.State(None)

    # ── Upload: store raw image only, no server processing ──
    def on_upload(img):
        if img is None:
            return None, gr.update(mode="### 🟢 Waiting for upload")
        return img, gr.update(mode="### 🟢 Adjust sliders then click **Confirm**")

    photo.upload(
        fn=on_upload,
        inputs=[photo],
        outputs=[st_img, mode],
    )

    # ── Confirm: all server processing happens here ──
    def on_confirm(img, z, px, py):
        if img is None:
            return None, gr.update(), None, None, None, None, None, None, None, None, None, None, None, None, None, None, gr.update()

        h, w = img.shape[:2]

        # Face detection + auto-rotate
        landmarks = detect_face(img)
        if landmarks is None:
            return img, gr.update(), None, None, None, None, None, None, None, None, None, None, None, None, None, None, gr.update(mode="### ❌ No face detected")

        rotated, rot_angle = auto_rotate(img, landmarks)
        if abs(rot_angle) > 0.5:
            landmarks = detect_face(rotated)
            if landmarks is None:
                rotated = img

        # Compute initial crop from face
        cx, cy, eye_y, scale_s, hw, hh = compute_initial_crop(landmarks, w, h)

        # Apply user pan/zoom
        acx = cx + px
        aeye = eye_y + py
        ascale = scale_s * z

        # Default HSV + shoulder params
        chin_y = landmarks[152].y * h
        chin_cy = int((chin_y - aeye) * ascale + TARGET_H_PX * 0.32)
        fh_y = landmarks[10].y * h
        fh_cy = int((fh_y - aeye) * ascale + TARGET_H_PX * 0.32)
        face_h = chin_cy - fh_cy
        tilt = FaceDetector.estimate_tilt(landmarks)
        sx = default_shoulder_points(chin_cy, face_h, tilt)
        nxp, nyp, lxp, lyp, rxp, ryp = sx

        hm, hx, sm, sx2, vm, vx, fe = 0, 25, 20, 255, 30, 255, 3

        _, _, result = crop_and_make_mask(
            rotated, landmarks, acx, aeye, ascale,
            hm, hx, sm, sx2, vm, vx, fe,
            nxp, nyp, lxp, lyp, rxp, ryp,
        )

        return [
            result,
            gr.update(visible=True),
            hm, hx, sm, sx2, vm, vx, fe,
            nxp, nyp, lxp, lyp, rxp, ryp,
            gr.update(mode="### ✅ Done — fine-tune below if needed"),
        ]

    confirm.click(
        fn=on_confirm,
        inputs=[st_img, zoom, pan_x, pan_y],
        outputs=[photo, bg_group,
                 h_min, h_max, s_min, s_max, v_min, v_max, feather,
                 n_x, n_y, l_x, l_y, r_x, r_y,
                 mode],
    )

    # ── BG fine-tune ──
    def on_bg(img, z, px, py,
              hm, hx, sm, sx2, vm, vx, fe,
              nx, ny, lx, ly, rx, ry):
        if img is None:
            return None

        h, w = img.shape[:2]
        landmarks = detect_face(img)
        if landmarks is None:
            return None

        rotated, rot_angle = auto_rotate(img, landmarks)
        if abs(rot_angle) > 0.5:
            landmarks = detect_face(rotated)
            if landmarks is None:
                rotated = img

        cx, cy, eye_y, scale_s, hw, hh = compute_initial_crop(landmarks, w, h)
        acx = cx + px
        aeye = eye_y + py
        ascale = scale_s * z

        _, _, result = crop_and_make_mask(
            rotated, landmarks, acx, aeye, ascale,
            hm, hx, sm, sx2, vm, vx, fe,
            nx, ny, lx, ly, rx, ry,
        )
        return result

    bg_inputs = [st_img, zoom, pan_x, pan_y,
                 h_min, h_max, s_min, s_max, v_min, v_max, feather,
                 n_x, n_y, l_x, l_y, r_x, r_y]

    for ctrl in [h_min, h_max, s_min, s_max, v_min, v_max, feather,
                 n_x, n_y, l_x, l_y, r_x, r_y]:
        ctrl.change(fn=on_bg, inputs=bg_inputs, outputs=[photo])

    save.click(fn=save_fn, inputs=[photo], outputs=[saved])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, css=CSS)
