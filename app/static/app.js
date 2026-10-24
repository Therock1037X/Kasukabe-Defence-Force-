/**
 * KDF Navigation — Car Dashboard Frontend Application
 * MapLibre GL JS + Web Audio API + Real-time Dual Routing
 */

// Application State
const state = {
  map: null,
  hour: 12,
  weather: 'clear',
  month: 5,
  dayType: 'weekday',
  lambda: 1.5,
  startCoord: [21.1440, 79.0660], // Dharampeth
  endCoord: [21.0950, 79.0750],   // Wardha Road (Khapri/Ajni)
  selectedRoute: 'safest',
  routesData: null,
  simRunning: false,
  simTimer: null,
  simStep: 0,
  carMarker: null,
  livePolling: false,
  liveTimer: null,
  audioContext: null,
  lastAlertTime: 0,
  showNightLights: false,
};

// Web Audio API Chime Generator
function playAlertChime() {
  try {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) return;
    if (!state.audioContext) {
      state.audioContext = new AudioContext();
    }
    const ctx = state.audioContext;
    if (ctx.state === 'suspended') {
      ctx.resume();
    }

    // Two-tone modern vehicle advisory chime (440Hz -> 880Hz)
    const now = ctx.currentTime;
    
    // Tone 1
    const osc1 = ctx.createOscillator();
    const gain1 = ctx.createGain();
    osc1.type = 'sine';
    osc1.frequency.setValueAtTime(587.33, now); // D5
    gain1.gain.setValueAtTime(0.2, now);
    gain1.gain.exponentialRampToValueAtTime(0.001, now + 0.18);
    osc1.connect(gain1);
    gain1.connect(ctx.destination);
    osc1.start(now);
    osc1.stop(now + 0.18);

    // Tone 2
    const osc2 = ctx.createOscillator();
    const gain2 = ctx.createGain();
    osc2.type = 'sine';
    osc2.frequency.setValueAtTime(880.0, now + 0.12); // A5
    gain2.gain.setValueAtTime(0.25, now + 0.12);
    gain2.gain.exponentialRampToValueAtTime(0.001, now + 0.38);
    osc2.connect(gain2);
    gain2.connect(ctx.destination);
    osc2.start(now + 0.12);
    osc2.stop(now + 0.38);
  } catch (e) {
    console.warn('Audio chime notice:', e);
  }
}

// Initialize MapLibre GL Map
function initMap() {
  state.map = new maplibregl.Map({
    container: 'map',
    style: {
      version: 8,
      sources: {
        'carto-dark': {
          type: 'raster',
          tiles: [
            'https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
            'https://b.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
            'https://c.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png',
          ],
          tileSize: 256,
          attribution: '© OpenStreetMap, © CARTO',
        },
      },
      layers: [
        {
          id: 'carto-dark-layer',
          type: 'raster',
          source: 'carto-dark',
          minzoom: 0,
          maxzoom: 19,
        },
      ],
    },
    center: [79.0882, 21.1458], // Nagpur Center
    zoom: 12.8,
    pitch: 25,
  });

  state.map.addControl(new maplibregl.NavigationControl({ showCompass: true }), 'top-right');

  state.map.on('load', () => {
    setupMapLayers();
    fetchRiskNetwork();
    fetchRoutes();
  });

  // Map Click to Pick Custom Route Start or Destination
  let clickMode = 'start';
  state.map.on('click', (e) => {
    const features = state.map.queryRenderedFeatures(e.point, { layers: ['road-risk-layer'] });
    if (features && features.length > 0) {
      // Tap on road segment: show inspection modal
      showSegmentDetails(features[0].properties);
      return;
    }

    // Set custom coordinates
    const lat = parseFloat(e.lngLat.lat.toFixed(5));
    const lon = parseFloat(e.lngLat.lng.toFixed(5));

    if (clickMode === 'start') {
      state.startCoord = [lat, lon];
      document.getElementById('route-start').value = `Picked [${lat}, ${lon}]`;
      clickMode = 'end';
    } else {
      state.endCoord = [lat, lon];
      document.getElementById('route-dest').value = `Picked [${lat}, ${lon}]`;
      clickMode = 'start';
      fetchRoutes();
    }
  });
}

// Setup Map Layers (Road Risk Strips + Route Overlays)
function setupMapLayers() {
  // 1. Road Risk GeoJSON Source
  state.map.addSource('road-risk-source', {
    type: 'geojson',
    data: { type: 'FeatureCollection', features: [] },
  });

  // Background glow casing for color-blind clarity
  state.map.addLayer({
    id: 'road-risk-casing',
    type: 'line',
    source: 'road-risk-source',
    layout: { 'line-join': 'round', 'line-cap': 'round' },
    paint: {
      'line-color': '#000000',
      'line-width': ['interpolate', ['linear'], ['zoom'], 10, 3, 14, 6.5, 17, 11],
      'line-opacity': 0.7,
    },
  });

  // Foreground colored road strips (Green, Yellow, Orange, Red)
  state.map.addLayer({
    id: 'road-risk-layer',
    type: 'line',
    source: 'road-risk-source',
    layout: { 'line-join': 'round', 'line-cap': 'round' },
    paint: {
      'line-color': ['get', 'color'],
      'line-width': ['interpolate', ['linear'], ['zoom'], 10, 2, 14, 4.5, 17, 8],
      'line-opacity': 0.95,
    },
  });

  // 2. Route Sources & Layers
  state.map.addSource('fastest-route-source', {
    type: 'geojson',
    data: { type: 'FeatureCollection', features: [] },
  });
  state.map.addSource('safest-route-source', {
    type: 'geojson',
    data: { type: 'FeatureCollection', features: [] },
  });

  // Fastest route (dashed high-visibility blue/cyan)
  state.map.addLayer({
    id: 'fastest-route-line',
    type: 'line',
    source: 'fastest-route-source',
    layout: { 'line-join': 'round', 'line-cap': 'round' },
    paint: {
      'line-color': '#3a86ff',
      'line-width': ['interpolate', ['linear'], ['zoom'], 10, 3, 14, 6, 17, 10],
      'line-dasharray': [2, 2],
      'line-opacity': 0.85,
    },
  });

  // Safest route (solid neon-emerald green)
  state.map.addLayer({
    id: 'safest-route-line',
    type: 'line',
    source: 'safest-route-source',
    layout: { 'line-join': 'round', 'line-cap': 'round' },
    paint: {
      'line-color': '#00ff88',
      'line-width': ['interpolate', ['linear'], ['zoom'], 10, 4, 14, 7.5, 17, 12],
      'line-opacity': 0.95,
    },
  });

  // Cursor pointer on hover over segments
  state.map.on('mouseenter', 'road-risk-layer', () => {
    state.map.getCanvas().style.cursor = 'pointer';
  });
  state.map.on('mouseleave', 'road-risk-layer', () => {
    state.map.getCanvas().style.cursor = '';
  });
}

// Fetch Road Risk Strips from Backend
async function fetchRiskNetwork() {
  try {
    const url = `/risk?hour=${state.hour}&weather=${state.weather}&month=${state.month}&day_type=${state.dayType}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const geojson = await res.json();

    if (state.map.getSource('road-risk-source')) {
      state.map.getSource('road-risk-source').setData(geojson);
    }
  } catch (err) {
    console.error('Failed to fetch risk network:', err);
  }
}

// Fetch Dual Routes (Fastest vs. Safest)
async function fetchRoutes() {
  try {
    const fromStr = `${state.startCoord[0]},${state.startCoord[1]}`;
    const toStr = `${state.endCoord[0]},${state.endCoord[1]}`;
    const url = `/route?from=${fromStr}&to=${toStr}&lambda=${state.lambda}&hour=${state.hour}&weather=${state.weather}&month=${state.month}`;

    const res = await fetch(url);
    if (!res.ok) throw new Error(`Route error HTTP ${res.status}`);
    const data = await res.json();
    state.routesData = data;

    // Update One-Line Trade-Off Advisory
    document.getElementById('route-summary-text').innerText = data.summary;

    const fastest = data.routes.fastest;
    const safest = data.routes.safest;

    // Update Fastest Card
    document.getElementById('fastest-time').innerText = `${fastest.eta_minutes} min`;
    document.getElementById('fastest-dist').innerText = `${fastest.distance_km} km`;
    document.getElementById('fastest-risk').innerText = fastest.route_risk.toFixed(2);
    document.getElementById('fastest-danger').innerText = `${fastest.share_red_orange_pct}%`;
    document.getElementById('fastest-worst').innerText = `${fastest.worst_segment.road_name} (Risk ${fastest.worst_segment.risk_index})`;

    // Update Safest Card
    document.getElementById('safest-time').innerText = `${safest.eta_minutes} min`;
    document.getElementById('safest-dist').innerText = `${safest.distance_km} km`;
    document.getElementById('safest-risk').innerText = safest.route_risk.toFixed(2);
    document.getElementById('safest-danger').innerText = `${safest.share_red_orange_pct}%`;
    document.getElementById('safest-worst').innerText = `${safest.worst_segment.road_name} (Risk ${safest.worst_segment.risk_index})`;

    // Update Map Overlays
    if (state.map.getSource('fastest-route-source')) {
      state.map.getSource('fastest-route-source').setData({
        type: 'Feature',
        geometry: fastest.geometry,
      });
    }
    if (state.map.getSource('safest-route-source')) {
      state.map.getSource('safest-route-source').setData({
        type: 'Feature',
        geometry: safest.geometry,
      });
    }

    // Fit map bounds to show both routes
    if (safest.geometry && safest.geometry.coordinates && safest.geometry.coordinates.length > 0) {
      const bounds = new maplibregl.LngLatBounds();
      safest.geometry.coordinates.forEach(coord => bounds.extend(coord));
      state.map.fitBounds(bounds, { padding: 80, maxZoom: 15 });
    }
  } catch (err) {
    console.error('Route calculation error:', err);
    document.getElementById('route-summary-text').innerText = 'Routing error: check road network connection.';
  }
}

// Drive Simulation Mode along Chosen Route
function toggleDriveSimulation() {
  const btn = document.getElementById('btn-sim-drive');
  const speedo = document.getElementById('sim-speedo');

  if (state.simRunning) {
    // Stop simulation
    clearInterval(state.simTimer);
    state.simRunning = false;
    btn.classList.remove('sim-running');
    btn.innerText = '▶ Start Drive Simulation';
    speedo.classList.add('hidden');
    hideProximityAlert();
    if (state.carMarker) {
      state.carMarker.remove();
      state.carMarker = null;
    }
    return;
  }

  // Start simulation
  if (!state.routesData) return;
  const chosenRoute = state.selectedRoute === 'safest' 
    ? state.routesData.routes.safest 
    : state.routesData.routes.fastest;

  const coords = chosenRoute.geometry.coordinates;
  if (!coords || coords.length === 0) return;

  state.simRunning = true;
  state.simStep = 0;
  btn.classList.add('sim-running');
  btn.innerText = '⏹ Stop Simulation';
  speedo.classList.remove('hidden');

  // Create vehicle marker
  const el = document.createElement('div');
  el.className = 'car-marker';
  state.carMarker = new maplibregl.Marker({ element: el })
    .setLngLat(coords[0])
    .addTo(state.map);

  // Animate at ~350ms per step (~45 km/h simulated speed)
  state.simTimer = setInterval(() => {
    if (state.simStep >= coords.length - 1) {
      toggleDriveSimulation(); // Finished route
      return;
    }

    state.simStep++;
    const currentPt = coords[state.simStep];
    state.carMarker.setLngLat(currentPt);

    // Update speedometer HUD
    const speed = Math.floor(40 + Math.random() * 15);
    document.getElementById('sim-speed-val').innerText = `${speed} km/h`;
    const progressKm = (state.simStep / coords.length) * chosenRoute.distance_km;
    document.getElementById('sim-dist-val').innerText = `${progressKm.toFixed(1)} km`;

    // Query collision risk alert from backend
    checkCollisionAlert(currentPt[1], currentPt[0]);
  }, 350);
}

// Check Proximity Alert (GET /alert?lat=&lon=)
async function checkCollisionAlert(lat, lon) {
  try {
    const url = `/alert?lat=${lat}&lon=${lon}&hour=${state.hour}&weather=${state.weather}&month=${state.month}`;
    const res = await fetch(url);
    if (!res.ok) return;
    const alertData = await res.json();

    if (alertData.alert) {
      showProximityAlert(alertData.distance_m, alertData.band, alertData.message);
      // Play audio chime beep if at least 4 seconds since last chime
      const now = Date.now();
      if (now - state.lastAlertTime > 4000) {
        playAlertChime();
        state.lastAlertTime = now;
      }
    } else {
      hideProximityAlert();
    }
  } catch (e) {
    console.warn('Alert query error:', e);
  }
}

function showProximityAlert(distanceM, band, message) {
  const banner = document.getElementById('proximity-alert');
  const msgEl = document.getElementById('alert-message');
  const badgeEl = document.getElementById('alert-band-badge');

  msgEl.innerText = `${message} — ${distanceM.toFixed(0)} m ahead`;
  badgeEl.innerText = band.toUpperCase();
  badgeEl.style.color = band === 'red' ? '#ff7675' : '#f39c12';
  banner.classList.remove('hidden');
}

function hideProximityAlert() {
  document.getElementById('proximity-alert').classList.add('hidden');
}

// Show Segment Details Inspection Modal
async function showSegmentDetails(props) {
  const modal = document.getElementById('segment-modal');
  const sid = props.segment_id;

  try {
    const url = `/segment/${sid}?hour=${state.hour}&weather=${state.weather}&month=${state.month}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error();
    const data = await res.json();

    document.getElementById('seg-title').innerText = data.road_name;
    document.getElementById('seg-badge').innerText = `${data.band.toUpperCase()} RISK`;
    document.getElementById('seg-badge').style.borderColor = data.color;
    document.getElementById('seg-badge').style.color = data.color;
    document.getElementById('seg-risk-index').innerText = `Risk Index: ${data.risk_index.toFixed(3)}`;

    document.getElementById('prob-minor').innerText = `${(data.probabilities.p_minor * 100).toFixed(1)}%`;
    document.getElementById('prob-serious').innerText = `${(data.probabilities.p_serious * 100).toFixed(1)}%`;
    document.getElementById('prob-fatal').innerText = `${(data.probabilities.p_fatal * 100).toFixed(1)}%`;

    const list = document.getElementById('seg-drivers-list');
    list.innerHTML = '';
    (data.top_drivers || []).forEach(d => {
      const li = document.createElement('li');
      li.innerText = d.charAt(0).toUpperCase() + d.slice(1);
      list.appendChild(li);
    });

    document.getElementById('sm-length').innerText = `${data.length_m} m`;
    document.getElementById('sm-lanes').innerText = data.lanes;
    document.getElementById('sm-speed').innerText = `${data.maxspeed_kmh} km/h`;
    document.getElementById('sm-elevation').innerText = `${data.environmental_factors.elevation_m} m`;

    modal.classList.remove('hidden');
  } catch (e) {
    console.warn('Segment details error:', e);
  }
}

// Setup Event Listeners
function setupEventListeners() {
  // Scenario Preset Buttons
  document.querySelectorAll('.scenario-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.scenario-btn').forEach(b => b.classList.remove('active'));
      e.target.classList.add('active');

      state.hour = parseInt(e.target.dataset.hour);
      state.weather = e.target.dataset.weather;
      state.month = parseInt(e.target.dataset.month);

      // Sync form pickers
      document.getElementById('hour-picker').value = state.hour;
      document.getElementById('hour-disp').innerText = `${String(state.hour).padStart(2, '0')}:00`;
      document.getElementById('weather-select').value = state.weather;
      document.getElementById('month-select').value = state.month;

      fetchRiskNetwork();
      fetchRoutes();
    });
  });

  // Safety vs Speed Slider
  const slider = document.getElementById('safety-slider');
  slider.addEventListener('input', (e) => {
    state.lambda = parseFloat(e.target.value);
    document.getElementById('lambda-val').innerText = state.lambda.toFixed(2);
  });
  slider.addEventListener('change', () => {
    fetchRoutes();
  });

  // Location Presets
  document.querySelectorAll('.chip-loc').forEach(chip => {
    chip.addEventListener('click', (e) => {
      const s = e.target.dataset.start.split(',').map(Number);
      const end = e.target.dataset.end.split(',').map(Number);
      state.startCoord = s;
      state.endCoord = end;
      document.getElementById('route-start').value = `Start [${s[0]}, ${s[1]}]`;
      document.getElementById('route-dest').value = `Destination [${end[0]}, ${end[1]}]`;
      fetchRoutes();
    });
  });

  // Select Route Buttons
  document.getElementById('btn-choose-fastest').addEventListener('click', () => {
    state.selectedRoute = 'fastest';
    document.getElementById('card-fastest').classList.add('active-safest');
    document.getElementById('card-safest').classList.remove('active-safest');
  });
  document.getElementById('btn-choose-safest').addEventListener('click', () => {
    state.selectedRoute = 'safest';
    document.getElementById('card-safest').classList.add('active-safest');
    document.getElementById('card-fastest').classList.remove('active-safest');
  });

  // Simulation Button
  document.getElementById('btn-sim-drive').addEventListener('click', () => {
    toggleDriveSimulation();
  });

  // Modal Close
  document.getElementById('btn-close-modal').addEventListener('click', () => {
    document.getElementById('segment-modal').classList.add('hidden');
  });

  // Sidebar Tab Switcher
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

      e.target.classList.add('active');
      const tabId = `tab-${e.target.dataset.tab}`;
      const content = document.getElementById(tabId);
      if (content) content.classList.add('active');
    });
  });

  // Hour Slider
  document.getElementById('hour-picker').addEventListener('input', (e) => {
    state.hour = parseInt(e.target.value);
    document.getElementById('hour-disp').innerText = `${String(state.hour).padStart(2, '0')}:00`;
  });

  // Apply Conditions Button
  document.getElementById('btn-apply-conditions').addEventListener('click', () => {
    state.weather = document.getElementById('weather-select').value;
    state.month = parseInt(document.getElementById('month-select').value);
    fetchRiskNetwork();
    fetchRoutes();
  });

  // ESP32 Live Telemetry Feed Toggle
  document.getElementById('btn-toggle-live').addEventListener('click', () => {
    state.livePolling = !state.livePolling;
    const dot = document.getElementById('live-dot');
    const label = document.getElementById('live-label');

    if (state.livePolling) {
      dot.style.background = '#00d2ff';
      label.innerText = 'ESP32 Feed: LIVE';
      state.liveTimer = setInterval(pollLiveTelemetry, 3000);
    } else {
      dot.style.background = '#2ecc71';
      label.innerText = 'ESP32 Feed: SIM';
      clearInterval(state.liveTimer);
    }
  });

  // Inject Test Telemetry Button
  document.getElementById('btn-send-test-reading').addEventListener('click', async () => {
    try {
      const payload = {
        lat: 21.1440 + (Math.random() - 0.5) * 0.02,
        lon: 79.0750 + (Math.random() - 0.5) * 0.02,
        rain_sensor: Math.random() > 0.5 ? 750 : 120,
        ldr: 250,
        harsh_brake: Math.random() > 0.7 ? 1 : 0,
        speed: 48.0,
      };
      const res = await fetch('/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      const data = await res.json();
      alert(`ESP32 Ingest Succeeded: Inferred ${data.inferred_weather}, ${data.inferred_light} on segment #${data.nearest_segment_id}`);
      pollLiveTelemetry();
    } catch (e) {
      alert(`Ingest error: ${e}`);
    }
  });

  // Night Lights Toggle
  document.getElementById('btn-night-light').addEventListener('click', (e) => {
    state.showNightLights = !state.showNightLights;
    e.target.classList.toggle('active', state.showNightLights);
    // Adjust map opacity to highlight night lights
    if (state.map.getLayer('road-risk-layer')) {
      state.map.setPaintProperty('road-risk-layer', 'line-opacity', state.showNightLights ? 0.6 : 0.95);
    }
  });
}

// Poll Live Telemetry (GET /live)
async function pollLiveTelemetry() {
  try {
    const res = await fetch('/live');
    if (!res.ok) return;
    const data = await res.json();

    document.getElementById('tel-gps').innerText = `${data.lat.toFixed(4)}, ${data.lon.toFixed(4)}`;
    document.getElementById('tel-spd').innerText = `${data.speed.toFixed(1)} km/h`;
    document.getElementById('tel-rain').innerText = `${data.rain_sensor} (${data.inferred_weather})`;
    document.getElementById('tel-ldr').innerText = `${data.ldr} (${data.inferred_light})`;
    document.getElementById('tel-brake').innerText = data.harsh_brake ? '⚠️ HARSH BRAKE (>0.4g)' : 'Normal (0.12g)';
    document.getElementById('tel-segment').innerText = `#${data.nearest_segment_id || 'unknown'}`;
  } catch (e) {
    console.warn('Live telemetry poll error:', e);
  }
}

// Start application
window.addEventListener('DOMContentLoaded', () => {
  initMap();
  setupEventListeners();
});
