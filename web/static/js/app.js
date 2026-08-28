// DeepSORVF Web C2 Dashboard Application Logic

let map;
let vesselMarkers = {};
let ws;
let currentProfileId = "istanbul_kizkulesi";
let calibrationDebounceTimer = null;
let currentMode = "file";

const PROFILES_CONFIG = {
    "istanbul_kizkulesi": {
        name: "İstanbul Boğazı (Kız Kulesi / Tower to Tower)",
        lat: 41.0185, lon: 28.9850, zoom: 13,
        heading: 118, pitch: -3.5, height: 35, fov: 45
    },
    "istanbul_bosphorus": {
        name: "İstanbul Boğazı (Sarayburnu)",
        lat: 41.0185, lon: 28.9850, zoom: 13,
        heading: 45, pitch: -5, height: 35, fov: 45
    },
    "port_rotterdam": {
        name: "Rotterdam Limanı (Nieuwe Waterweg)",
        lat: 51.9850, lon: 4.1200, zoom: 13,
        heading: 280, pitch: -4, height: 25, fov: 40
    },
    "port_miami": {
        name: "PortMiami (Government Cut)",
        lat: 25.7650, lon: -80.1350, zoom: 13,
        heading: 95, pitch: -6, height: 30, fov: 50
    },
    "custom": {
        name: "Özel Konum",
        lat: 41.0185, lon: 28.9850, zoom: 13,
        heading: 118, pitch: -3.5, height: 35, fov: 45
    },
    "file_default": {
        name: "Wuhan Segment (Arşiv)",
        lat: 30.600, lon: 114.327, zoom: 13,
        heading: 352, pitch: -4, height: 55, fov: 30
    }
};

// Initialize Leaflet Map with OpenSeaMap & Satellite Layers
function initMap() {
    const defaultLat = 30.600;
    const defaultLon = 114.327;

    const osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; OpenStreetMap contributors'
    });

    const satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Tiles &copy; Esri'
    });

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

// Clear all markers from map
function clearMapMarkers() {
    Object.values(vesselMarkers).forEach(marker => {
        map.removeLayer(marker);
    });
    vesselMarkers = {};
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
            const shipName = item.name || `MMSI_${item.mmsi}`;
            html += `
                <tr>
                    <td><strong style="color:#00d2ff">#${item.vis_id}</strong></td>
                    <td><strong style="color:#00e676">${item.mmsi}</strong></td>
                    <td>${shipName}</td>
                    <td>${item.speed ? item.speed.toFixed(1) : '-'} kn</td>
                    <td>-</td>
                    <td>${item.lat ? item.lat.toFixed(5) : '-'}</td>
                    <td>${item.lon ? item.lon.toFixed(5) : '-'}</td>
                    <td><span class="badge-status-fused"><i class="fa-solid fa-link"></i> Fused (MMSI Eşleşti)</span></td>
                </tr>
            `;
        });
        tableBody.innerHTML = html;
    } else if (data.ais_vessels && data.ais_vessels.length > 0) {
        let html = "";
        data.ais_vessels.forEach((v, idx) => {
            html += `
                <tr>
                    <td><strong style="color:#94a3b8">#AIS_${idx+1}</strong></td>
                    <td><strong style="color:#00d2ff">${v.mmsi}</strong></td>
                    <td>${v.name || 'Gemi'}</td>
                    <td>${v.speed ? v.speed.toFixed(1) : '-'} kn</td>
                    <td>${v.course ? v.course.toFixed(0) : '-'}°</td>
                    <td>${v.lat ? v.lat.toFixed(5) : '-'}</td>
                    <td>${v.lon ? v.lon.toFixed(5) : '-'}</td>
                    <td><span style="color:#00d2ff; font-size:0.75rem;"><i class="fa-solid fa-satellite"></i> AIS Canlı</span></td>
                </tr>
            `;
        });
        tableBody.innerHTML = html;
    } else {
        tableBody.innerHTML = `<tr><td colspan="8" class="empty-msg">Aktif gemi verisi bekleniyor...</td></tr>`;
    }

    // 4. Update Map Markers
    if (data.ais_vessels && data.ais_vessels.length > 0) {
        const activeMmsis = new Set();
        data.ais_vessels.forEach(v => {
            const mmsi = v.mmsi;
            const lat = v.lat;
            const lon = v.lon;
            const name = v.name || `MMSI ${mmsi}`;
            activeMmsis.add(mmsi);

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
                        fillOpacity: 0.85
                    }).addTo(map);

                    marker.bindPopup(`<b>${name}</b><br>MMSI: ${mmsi}<br>Hız: ${v.speed} kn<br>Lat: ${lat}<br>Lon: ${lon}`);
                    vesselMarkers[mmsi] = marker;
                }
            }
        });

        // Remove stale markers
        Object.keys(vesselMarkers).forEach(mmsi => {
            if (!activeMmsis.has(parseInt(mmsi)) && !activeMmsis.has(mmsi)) {
                map.removeLayer(vesselMarkers[mmsi]);
                delete vesselMarkers[mmsi];
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
    currentMode = mode;
    const isLive = mode === "live";

    document.getElementById("btn-mode-file").classList.toggle("active", !isLive);
    document.getElementById("btn-mode-live").classList.toggle("active", isLive);

    const livePanel = document.getElementById("live-control-panel");
    const statusBadge = document.getElementById("stream-status-badge");

    clearMapMarkers();

    if (isLive) {
        livePanel.classList.remove("hidden");
        statusBadge.innerHTML = `<i class="fa-solid fa-tower-cell pulsing"></i> LIVE STREAM`;
        statusBadge.style.background = "rgba(0, 230, 118, 0.2)";
        statusBadge.style.color = "var(--accent-green)";
        statusBadge.style.borderColor = "var(--accent-green)";

        // Fly map to Istanbul Bosphorus
        const profile = PROFILES_CONFIG[currentProfileId] || PROFILES_CONFIG["istanbul_bosphorus"];
        document.getElementById("map-location-badge").innerText = `Bölge: ${profile.name}`;
        map.setView([profile.lat, profile.lon], profile.zoom);

        // Auto-connect with saved API key
        const apiKey = localStorage.getItem("deep_ais_api_key") || "";
        if (apiKey) {
            connectLiveStream();
        } else {
            logEvent(`[CANLI MOD] Canlı akış moduna geçildi. Lütfen API anahtarınızı girip başlatın.`, "info");
        }
    } else {
        livePanel.classList.add("hidden");
        statusBadge.innerHTML = `<i class="fa-solid fa-circle-play"></i> FILE REPLAYER`;
        statusBadge.style.background = "rgba(0, 210, 255, 0.15)";
        statusBadge.style.color = "var(--accent-blue)";
        statusBadge.style.borderColor = "var(--accent-blue)";

        // Fly back to dataset
        const def = PROFILES_CONFIG["file_default"];
        document.getElementById("map-location-badge").innerText = `Bölge: ${def.name}`;
        map.setView([def.lat, def.lon], def.zoom);

        fetch("/api/mode", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ mode: "file" })
        }).then(res => res.json()).then(d => {
            logEvent(`[DOSYA MODU AKTİF] Arşiv video ve AIS oynatıcıya geçildi.`, "info");
        });
    }
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

function onProfileChange(profileId) {
    currentProfileId = profileId;
    const isCustom = profileId === "custom";
    document.getElementById("custom-url-group").classList.toggle("hidden", !isCustom);

    const cfg = PROFILES_CONFIG[profileId] || PROFILES_CONFIG["istanbul_bosphorus"];
    document.getElementById("slider-heading").value = cfg.heading;
    document.getElementById("val-heading").innerText = `${cfg.heading}°`;

    document.getElementById("slider-pitch").value = cfg.pitch;
    document.getElementById("val-pitch").innerText = `${cfg.pitch}°`;

    document.getElementById("slider-height").value = cfg.height;
    document.getElementById("val-height").innerText = `${cfg.height} m`;

    document.getElementById("slider-fov").value = cfg.fov;
    document.getElementById("val-fov").innerText = `${cfg.fov}°`;

    clearMapMarkers();
    document.getElementById("map-location-badge").innerText = `Bölge: ${cfg.name}`;
    map.setView([cfg.lat, cfg.lon], cfg.zoom);
}

function onCalibrationSliderInput() {
    const heading = parseFloat(document.getElementById("slider-heading").value);
    const pitch = parseFloat(document.getElementById("slider-pitch").value);
    const height = parseFloat(document.getElementById("slider-height").value);
    const fov = parseFloat(document.getElementById("slider-fov").value);

    document.getElementById("val-heading").innerText = `${heading}°`;
    document.getElementById("val-pitch").innerText = `${pitch}°`;
    document.getElementById("val-height").innerText = `${height} m`;
    document.getElementById("val-fov").innerText = `${fov}°`;

    if (calibrationDebounceTimer) clearTimeout(calibrationDebounceTimer);
    calibrationDebounceTimer = setTimeout(() => {
        fetch("/api/live/calibrate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ heading, pitch, height, fov })
        }).then(res => res.json()).then(d => {
            logEvent(`[KALİBRASYON GÜNCELLENDİ] Yön: ${heading}°, Eğim: ${pitch}°, Yükseklik: ${height}m`, "info");
        });
    }, 150);
}

function connectLiveStream() {
    const profileId = document.getElementById("live-profile-select").value;
    const apiKey = document.getElementById("ais-api-key").value.trim();
    const customUrl = document.getElementById("custom-stream-url").value.trim();

    if (apiKey) {
        localStorage.setItem("deep_ais_api_key", apiKey);
    }

    currentProfileId = profileId;
    const profile = PROFILES_CONFIG[profileId] || PROFILES_CONFIG["istanbul_bosphorus"];

    clearMapMarkers();
    document.getElementById("map-location-badge").innerText = `Bölge: ${profile.name}`;
    map.setView([profile.lat, profile.lon], profile.zoom);

    logEvent(`[CANLI AKIŞ BAŞLATILIYOR] Profil: ${profile.name}...`, "info");

    fetch("/api/live/connect", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            profile_id: profileId,
            api_key: apiKey,
            custom_url: customUrl
        })
    }).then(res => res.json()).then(d => {
        logEvent(`[CANLI AKIŞ AKTİF] ${profile.name} yayını ve AIS verisi bağlandı.`, "info");
    }).catch(err => {
        logEvent(`[HATA] Canlı bağlantı hatası: ${err}`, "alert");
    });
}

// DOM Loaded Initialization
document.addEventListener("DOMContentLoaded", () => {
    initMap();
    initWebSocket();
    const savedKey = localStorage.getItem("deep_ais_api_key");
    if (savedKey) {
        const input = document.getElementById("ais-api-key");
        if (input) input.value = savedKey;
    }
});
