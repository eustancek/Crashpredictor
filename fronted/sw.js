const CACHE_NAME = 'crash-predictor-cache-v2';
const DELTA_CACHE_NAME = 'delta-cache';

self.addEventListener('install', e => {
    e.waitUntil(
        caches.open(CACHE_NAME).then(cache => 
            cache.addAll(['/', '/index.html', '/manifest.json', '/js/main.js', '/css/styles.css'])
        )
    );
});

self.addEventListener('fetch', e => {
    if (e.request.url.includes('/static/')) {
        e.respondWith(
            caches.match(e.request).then(r => r || fetch(e.request))
        );
    }
    
    if (e.request.url.includes('/predict')) {
        e.respondWith(
            fetch(e.request).then(response => {
                caches.open(DELTA_CACHE_NAME).then(cache => {
                    cache.put(e.request, response.clone());
                });
                return response;
            }).catch(() => caches.match(e.request))
        );
    }
});