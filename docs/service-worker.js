// Minimal service worker so EveryCam installs as an app and works offline.
const CACHE = "everycam-v1";
const ASSETS = [
  "./", "index.html", "record.html", "manifest.webmanifest",
  "icon-192.png", "icon-512.png", "assets/pipeline.svg",
];

self.addEventListener("install", (e) => {
  e.waitUntil(
    caches.open(CACHE).then((c) => c.addAll(ASSETS).catch(() => {})).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (e) => {
  e.waitUntil(
    caches.keys()
      .then((ks) => Promise.all(ks.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (e) => {
  if (e.request.method !== "GET") return;
  e.respondWith(
    caches.match(e.request).then(
      (r) =>
        r ||
        fetch(e.request)
          .then((resp) => {
            const cp = resp.clone();
            caches.open(CACHE).then((c) => {
              try { c.put(e.request, cp); } catch (_) {}
            });
            return resp;
          })
          .catch(() => caches.match("index.html"))
    )
  );
});
