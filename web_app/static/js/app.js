// DeepSORVF Web C2 Dashboard Application Logic

let map;
let vesselMarkers = {};
let ws;

// Initialize Leaflet Map with OpenSeaMap & Satellite Layers
function initMap() {
    // Default center: Wuhan Yangtze River / Istanbul Strait coordinates fallback
    const defaultLat = 30.600;
    const defaultLon = 114.327;

    // Layer 1: OpenStreetMap
    const osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors'
    });

    // Layer 2: Esri World Imagery (Satellite)
    const satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles &copy; Esri'
    });

    // Layer 3: OpenSeaMap (Maritime Navigation Marks, Buoys, Sea Trails)
    const openseaMapLayer = L.tileLayer('https://tiles.openseamap.org/seamark/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenSeaMap contributors'
    });

    map = L.map('map', {
        center: [defaultLat, defaultLon],
        zoom: 13,
        layers: [satelliteLayer, openseaMapLayer]
    });

    const baseMaps = {
        "Satellite Imagery": satelliteLayer,
        "OpenStreetMap": osmLayer
    };

    const overlayMaps = {
        "OpenSeaMap Maritime Marks": openseaMapLayer
    };

    L.control.layers(baseMaps, overlayMaps).addTo(map);
}

// WebSocket Connection Management
function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/ws`;

    ws = new WebSocket(wsUrl);

    ws.onopen = () => {
        logEvent("[WEBSOCKET] Bağlantı sağlandı (Connected).", "info");
    };

    ws.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            updateDashboard(data);
        } catch (e) {
            console.error("Telemetry parse error:", e);
        }
    };

    ws.onclose = () => {
        logEvent("[WEBSOCKET] Bağlantı kesildi. Yeniden bağlanıyor...", "alert");
        setTimeout(initWebSocket, 2000);
    };
}

// Update Dashboard Telemetry and Map
function updateDashboard(data) {
    // 1. Update Badges
    document.getElementById("fps-val").innerText = data.fps || "30.0";
    document.getElementById("metric-ais").innerText = data.ais_count || 0;
    document.getElementById("metric-vis").innerText = data.vis_count || 0;
    document.getElementById("metric-fus").innerText = data.fusion_count || 0;
    document.getElementById("metric-dark").innerText = data.unmatched_count || 0;
    document.getElementById("fusion-count").innerText = data.fusion_count || 0;
    if (data.model) {
        document.getElementById("video-info").innerText = `1920x1080 @ ${data.model}`;
    }

    // 2. Dark Ship / Unidentified Target Alert Banner
    const alertBanner = document.getElementById("dark-ship-alert");
    if (data.unmatched_count > 0) {
        alertBanner.classList.remove("hidden");
    } else {
        alertBanner.classList.add("hidden");
    }

    // 3. Update Table
    const tableBody = document.getElementById("vessel-table-body");
    if (data.fusion_matches && data.fusion_matches.length > 0) {
        let html = "";
        data.fusion_matches.forEach(item => {
            html += `
                <tr>
                    <td><strong style="color:#00d2ff">#${item.vis_id}</strong></td>
                    <td><strong style="color:#00e676">${item.mmsi}</strong></td>
                    <td>${item.speed || '-'} kn</td>
                    <td>-</td>
                    <td>${item.lat ? item.lat.toFixed(5) : '-'}</td>
                    <td>${item.lon ? item.lon.toFixed(5) : '-'}</td>
                    <td><span class="badge-status-fused"><i class="fa-solid fa-link"></i> Fused (MMSI Eşleşti)</span></td>
                </tr>
            `;
        });
        tableBody.innerHTML = html;
    } else {
        tableBody.innerHTML = `<tr><td colspan="7" class="empty-msg">Aktif eşleşme bekleniyor...</td></tr>`;
    }

    // 4. Update Map Markers
    if (data.ais_vessels && data.ais_vessels.length > 0) {
        data.ais_vessels.forEach(v => {
            const mmsi = v.mmsi;
            const lat = v.lat;
            const lon = v.lon;

            if (lat && lon && !isNaN(lat) && !isNaN(lon)) {
                if (vesselMarkers[mmsi]) {
                    vesselMarkers[mmsi].setLatLng([lat, lon]);
                } else {
                    const marker = L.circleMarker([lat, lon], {
                        radius: 8,
                        fillColor: "#00d2ff",
                        color: "#ffffff",
                        weight: 2,
                        opacity: 1,
                        fillOpacity: 0.8
                    }).addTo(map);

                    marker.bindPopup(`<b>Gemi MMSI: ${mmsi}</b><br>Hız: ${v.speed} kn<br>Lat: ${lat}<br>Lon: ${lon}`);
                    vesselMarkers[mmsi] = marker;
                }
            }
        });
    }
}

// Log Event Message
function logEvent(msg, type = "info") {
    const logContainer = document.getElementById("event-log-container");
    const div = document.createElement("div");
    div.className = `log-item ${type}`;
    div.innerText = `${new Date().toLocaleTimeString()} ${msg}`;
    logContainer.appendChild(div);
    logContainer.scrollTop = logContainer.scrollHeight;
}

// Set Stream Mode (File / Live)
function setMode(mode) {
    document.getElementById("btn-mode-file").classList.toggle("active", mode === "file");
    document.getElementById("btn-mode-live").classList.toggle("active", mode === "live");

    fetch("/api/mode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode: mode })
    }).then(res => res.json()).then(d => {
        logEvent(`[MOD DEĞİŞTİ] Yeni mod: ${mode.toUpperCase()}`, "info");
    });
}

// Change YOLO Model Size
function changeModel(modelName) {
    document.getElementById("video-info").innerText = `1920x1080 @ ${modelName}`;
    fetch("/api/model", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: modelName })
    }).then(res => res.json()).then(d => {
        logEvent(`[YOLO MODEL DEĞİŞTİ] Yeni model: ${modelName}`, "info");
    });
}

// DOM Loaded Initialization
document.addEventListener("DOMContentLoaded", () => {
    initMap();
    initWebSocket();
});
