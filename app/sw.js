const CACHE = 'passport-photo-v2';
const FILES = [
  '.',
  'index.html',
  'styles.css',
  'app.js',
  'face-detect.js',
  'background-remove.js',
  'presets.js',
  'icon-192.png',
  'icon-512.png'
];

self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE).then((c) => c.addAll(FILES))
  );
  self.skipWaiting();
});

self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.map((k) => { if (k !== CACHE) return caches.delete(k); }))
    )
  );
  self.clients.claim();
});

self.addEventListener('fetch', (e) => {
  if (e.request.url.startsWith(self.location.origin)) {
    e.respondWith(
      caches.match(e.request).then((r) => r || fetch(e.request).catch(() => r))
    );
  }
});
