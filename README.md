# Passport Photo

A progressive web app for generating compliant passport photos in the browser.

## Features

- Face detection and automatic cropping
- Background removal using a bundled U2Net ONNX model
- Country-specific size and head-position presets (China, US, UK, Canada, Australia, France, Germany, Japan, India, Brazil, EU/Schengen, and more)
- Export at the required dimensions and DPI
- Installable PWA with offline support

## Running

The app uses a service worker and models, so serve it over HTTP rather than opening the file directly:

```bash
cd app
python3 -m http.server 8000
```

Then visit <http://localhost:8000>.

## Layout

- `app/index.html` — UI
- `app/app.js` — application logic
- `app/face-detect.js` — face detection
- `app/background-remove.js` — background removal
- `app/presets.js` — country presets
- `app/u2netp.onnx` — segmentation model
- `passport_*.json` — per-country requirement data

## Privacy

Personal input/output images are intentionally excluded from version control.
