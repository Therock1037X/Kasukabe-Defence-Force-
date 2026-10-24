/**
 * KDF Navigation — Automotive Frontend Application Logic
 * Integrates MapLibre GL JS, Google Earth Engine Tiles API, Dual Routing, 
 * Google Maps Style Search & HUD, Web Audio API Advisory Chimes.
 */

// Global Application State
const appState = {
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
  activeLayer: 'dark',
  navRunning: false,
  simTimer: null,
  simIndex: 0,
  carMarker: null,
  livePolling: false,
  liveTimer: null,
  audioContext: null,
  lastAlertTime: 0,
};

// Nagpur Key Landmarks for Google Maps Style Search
const NAGPUR_LANDMARKS = [
  { name: 'Dharampeth (West Nagpur)', coords: [21.1440, 79.0660], type: 'Commercial & Residential Hub' },
  { name: 'Sitabuldi Interchange & Fort', coords: [21.1462, 79.0832], type: 'Central Transit Hub' },
  { name: 'Wardha Road (Ajni / Khapri Corridor)', coords: [21.0950, 79.0750], type: 'Major Arterial Trunk' },
  { name: 'Medical Square (GMC Nagpur)', coords: [21.1340, 79.0965], type: 'Hospital & Healthcare Corridor' },
  { name: 'Sadar (Residency Road)', coords: [21.1600, 79.0800], type: 'North Central Commercial Area' },
  { name: 'Zero Mile Stone (Geographic Center)', coords: [21.1498, 79.0806], type: 'Historical Monument' },
  { name: 'Dr. Babasaheb Ambedkar Airport (NAG)', coords: [21.0920, 79.0580], type: 'International Airport' },
  { name: 'Amravati Road (RTMNU Campus)', coords: [21.1480, 79.0500], type: 'Western Highway Corridor' },
];

// ---------------------------------------------------------------------------
// 1. WEB AUDIO API ADVISORY CHIME
// ---------------------------------------------------------------------------
function playWarningChime() {
  try {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    if (!AudioContext) return;
    if (!appState.audioContext) {
      appState.audioContext = new AudioContext();
    }
    const ctx = appState.audioContext;
    if (ctx.state === 'suspended') {
      ctx.resume();
    }

    const now = ctx.currentTime;
    
    // Tone 1: 587.33 Hz (D5)
    const osc1 = ctx.createOscillator();
    const gain1 = ctx.createGain();
    osc1.type = 'sine';
    osc1.frequency.setValueAtTime(587.33, now);
    gain1.gain.setValueAtTime(0.2, now);
    gain1.gain.exponentialRampToValueAtTime(0.001, now + 0.18);
    osc1.connect(gain1);
    gain1.connect(ctx.destination);
    osc1.start(now);
    osc1.stop(now + 0.18);

    // Tone 2: 880.0 Hz (A5)
    const osc2 = ctx.createOscillator();
    const gain2 = ctx.createGain();
    osc2.type = 'sine';
    osc2.frequency.setValueAtTime(880.0, now + 0.12);
    gain2.gain.setValueAtTime(0.25, now + 0.12);
    gain2.gain.exponentialRampToValueAtTime(0.001, now + 0.38);
    osc2.connect(gain2);
    gain2.connect(ctx.destination);
    osc2.start(now + 0.12);
    osc2.stop(now + 0.38);
  } catch (err) {
    console.warn('Audio chime notice:', err);
  }
}

// ---------------------------------------------------------------------------
// 2. INITIALIZE MAPLIBRE GL JS
// ---------------------------------------------------------------------------
function initMap() {
  appState.map = new maplibregl.Map({
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
    center: [79.0750, 21.1200], // Nagpur Center
    zoom: 12.8,
    pitch: 20,
    bearing: 0,
  });

  appState.map.on('load', () => {
    setupMapLayers();
    fetchRiskNetwork();
    calculateRoutes();
  });

  // Map Click to Set Route Start / Destination
  let clickMode = 'start';
  appState.map.on('click', (e) => {
    // If clicked on a road segment, open details modal
    const features = appState.map.queryRenderedFeatures(e.point, { layers: ['road-risk-layer'] });
    if (features && features.length > 0) {
      showSegmentDetails(features[0].properties);
      return;
    }

    const lat = parseFloat(e.lngLat.lat.toFixed(5));
    const lon = parseFloat(e.lngLat.lng.toFixed(5));

    if (clickMode === 'start') {
      appState.startCoord = [lat, lon];
      document.getElementById('input-start-loc').value = `Custom Start [${lat}, ${lon}]`;
      clickMode = 'end';
    } else {
      appState.endCoord = [lat, lon];
      document.getElementById('input-dest-loc').value = `Custom Destination [${lat}, ${lon}]`;
      clickMode = 'start';
      calculateRoutes();
    }
  });
}

// ---------------------------------------------------------------------------
// 3. MAP LAYERS SETUP (RISK STRIPS & ROUTES)
// ---------------------------------------------------------------------------
function setupMapLayers() {
  const map = appState.map;

  // 1. Road Risk GeoJSON Source
  map.addSource('road-risk-source', {
    type: 'geojson',
    data: { type: 'FeatureCollection', features: [] },
  });

  // Road risk casing (dark border for contrast)
  map.addLayer({
    id: 'road-risk-casing',
    type: 'line',
    source: 'road-risk-source',
    layout: { 'line-cap': 'round', 'line-join': 'round' },
    paint: {
      'line-color': '#051F20',
      'line-width': ['interpolate', ['linear'], ['zoom'], 11, 4, 14, 7, 17, 11],
      'line-opacity': 0.85,
    },
  });

  // Road risk main strips (exact colors: mint, amber, orange, red)
  map.addLayer({
    id: 'road-risk-layer',
    type: 'line',
    source: 'road-risk-source',
    layout: { 'line-cap': 'round', 'line-join': 'round' },
    paint: {
      'line-color': ['get', 'color'],
      'line-width': ['interpolate', ['linear'], ['zoom'], 11, 2.2, 14, 4.5, 17, 8],
      'line-opacity': 0.95,
    },
  });

  // 2. Dual Routes GeoJSON Sources
  map.addSource('route-safest-source', {
    type: 'geojson',
    data: { type: 'FeatureCollection', features: [] },
  });
  map.addSource('route-fastest-source', {
    type: 'geojson',
    data: { type: 'FeatureCollection', features: [] },
  });

  // Fastest route layer: High contrast Amber #F2C14E dashed
  map.addLayer({
    id: 'route-fastest-layer',
    type: 'line',
    source: 'route-fastest-source',
    layout: { 'line-cap': 'round', 'line-join': 'round' },
    paint: {
      'line-color': '#F2C14E',
      'line-width': ['interpolate', ['linear'], ['zoom'], 11, 4, 14, 6.5, 17, 10],
      'line-dasharray': [2, 1.5],
      'line-opacity': 0.75,
    },
  });

  // Safest route layer: Mint #8EB69B solid with outer glow
  map.addLayer({
    id: 'route-safest-casing',
    type: 'line',
    source: 'route-safest-source',
    layout: { 'line-cap': 'round', 'line-join': 'round' },
    paint: {
      'line-color': '#051F20',
      'line-width': ['interpolate', ['linear'], ['zoom'], 11, 7, 14, 10, 17, 15],
      'line-opacity': 0.9,
    },
  });

  map.addLayer({
    id: 'route-safest-layer',
    type: 'line',
    source: 'route-safest-source',
    layout: { 'line-cap': 'round', 'line-join': 'round' },
    paint: {
      'line-color': '#8EB69B',
      'line-width': ['interpolate', ['linear'], ['zoom'], 11, 5, 14, 7.5, 17, 12],
      'line-opacity': 1.0,
    },
  });

  // Change cursor to pointer on road hover
  map.on('mouseenter', 'road-risk-layer', () => { map.getCanvas().style.cursor = 'pointer'; });
  map.on('mouseleave', 'road-risk-layer', () => { map.getCanvas().style.cursor = ''; });
}

// ---------------------------------------------------------------------------
// 4. GOOGLE EARTH ENGINE TILE INTEGRATION
// ---------------------------------------------------------------------------
async function setBasemapLayer(layerKey) {
  appState.activeLayer = layerKey;
  const map = appState.map;

  // Remove existing satellite/raster layer if present
  if (map.getLayer('gee-raster-layer')) {
    map.removeLayer('gee-raster-layer');
  }
  if (map.getSource('gee-raster-source')) {
    map.removeSource('gee-raster-source');
  }

  const nameMap = {
    'dark': 'Dark Cockpit',
    'gee-satellite': 'GEE Satellite',
    'gee-viirs': 'GEE Night Lights',
    'gee-elevation': 'GEE Terrain',
  };
  document.getElementById('current-layer-name').textContent = nameMap[layerKey] || 'Map Layer';

  if (layerKey === 'dark') {
    map.setLayoutProperty('carto-dark-layer', 'visibility', 'visible');
    return;
  }

  // Request tile URL from backend FastAPI endpoint /api/gee-tiles
  const geeLayerName = layerKey === 'gee-satellite' ? 'satellite' : (layerKey === 'gee-viirs' ? 'viirs' : 'elevation');
  try {
    const resp = await fetch(`/api/gee-tiles?layer=${geeLayerName}`);
    const data = await resp.json();
    const tileUrl = data.tile_url;

    map.addSource('gee-raster-source', {
      type: 'raster',
      tiles: [tileUrl],
      tileSize: 256,
      attribution: '© Google Earth Engine / Copernicus / NOAA / USGS',
    });

    // Insert beneath road risk casing
    map.addLayer(
      {
        id: 'gee-raster-layer',
        type: 'raster',
        source: 'gee-raster-source',
        minzoom: 0,
        maxzoom: 20,
      },
      'road-risk-casing'
    );

    map.setLayoutProperty('carto-dark-layer', 'visibility', 'none');
  } catch (err) {
    console.error('Failed to load GEE layer:', err);
    map.setLayoutProperty('carto-dark-layer', 'visibility', 'visible');
  }
}

// ---------------------------------------------------------------------------
// 5. FETCH & RENDER RISK ROAD NETWORK
// ---------------------------------------------------------------------------
async function fetchRiskNetwork() {
  try {
    const url = `/risk?hour=${appState.hour}&weather=${appState.weather}&month=${appState.month}&day_type=${appState.dayType}`;
    const resp = await fetch(url);
    const geojson = await resp.json();

    const source = appState.map.getSource('road-risk-source');
    if (source) {
      source.setData(geojson);
    }
  } catch (err) {
    console.error('Error fetching risk network:', err);
  }
}

// ---------------------------------------------------------------------------
// 6. DUAL ROUTE COMPUTATION (FASTEST VS. SAFEST)
// ---------------------------------------------------------------------------
async function calculateRoutes() {
  const fromStr = `${appState.startCoord[0]},${appState.startCoord[1]}`;
  const toStr = `${appState.endCoord[0]},${appState.endCoord[1]}`;
  const url = `/route?from=${fromStr}&to=${toStr}&lambda=${appState.lambda}&weather=${appState.weather}&hour=${appState.hour}&month=${appState.month}`;

  try {
    const resp = await fetch(url);
    if (!resp.ok) {
      const errData = await resp.json();
      console.warn('Routing notice:', errData.detail);
      return;
    }
    const data = await resp.json();
    appState.routesData = data;

    // Update Tradeoff summary
    document.getElementById('tradeoff-summary-text').textContent = data.summary;

    const safest = data.routes.safest;
    const fastest = data.routes.fastest;

    // Update Safest Route Card
    document.getElementById('safest-eta').textContent = safest.eta_minutes.toFixed(1);
    document.getElementById('safest-dist').textContent = `${safest.distance_km} km`;
    document.getElementById('safest-hazard-badge').textContent = `${safest.share_red_orange_pct}% High Risk`;
    document.getElementById('safest-worst').textContent = safest.worst_segment.road_name || 'None';
    
    const avoidedKm = Math.max(0, fastest.red_orange_km - safest.red_orange_km).toFixed(1);
    document.getElementById('safest-avoided').textContent = `${avoidedKm} km avoided`;

    // Update Fastest Route Card
    document.getElementById('fastest-eta').textContent = fastest.eta_minutes.toFixed(1);
    document.getElementById('fastest-dist').textContent = `${fastest.distance_km} km`;
    document.getElementById('fastest-hazard-badge').textContent = `${fastest.share_red_orange_pct}% High Risk`;
    document.getElementById('fastest-worst').textContent = fastest.worst_segment.road_name || 'None';
    
    const timeSaved = Math.max(0, safest.eta_minutes - fastest.eta_minutes).toFixed(1);
    document.getElementById('fastest-diff').textContent = `${timeSaved} min faster`;

    // Update Map Overlays
    renderRouteGeometries(safest.geometry, fastest.geometry);
  } catch (err) {
    console.error('Failed to calculate routes:', err);
  }
}

function renderRouteGeometries(safestGeom, fastestGeom) {
  const map = appState.map;
  if (!map) return;

  const safestSource = map.getSource('route-safest-source');
  if (safestSource) {
    safestSource.setData({
      type: 'FeatureCollection',
      features: [{ type: 'Feature', geometry: safestGeom, properties: {} }],
    });
  }

  const fastestSource = map.getSource('route-fastest-source');
  if (fastestSource) {
    fastestSource.setData({
      type: 'FeatureCollection',
      features: [{ type: 'Feature', geometry: fastestGeom, properties: {} }],
    });
  }
}

function selectActiveRoute(routeType) {
  appState.selectedRoute = routeType;
  const cardSafest = document.getElementById('card-safest');
  const cardFastest = document.getElementById('card-fastest');
  const btnSafest = document.getElementById('btn-select-safest');
  const btnFastest = document.getElementById('btn-select-fastest');

  if (routeType === 'safest') {
    cardSafest.classList.add('selected');
    cardFastest.classList.remove('selected');
    btnSafest.textContent = 'Selected for Driving';
    btnFastest.textContent = 'Select Fastest';

    appState.map.setPaintProperty('route-safest-layer', 'line-opacity', 1.0);
    appState.map.setPaintProperty('route-fastest-layer', 'line-opacity', 0.4);
  } else {
    cardFastest.classList.add('selected');
    cardSafest.classList.remove('selected');
    btnFastest.textContent = 'Selected for Driving';
    btnSafest.textContent = 'Select Safest';

    appState.map.setPaintProperty('route-fastest-layer', 'line-opacity', 1.0);
    appState.map.setPaintProperty('route-safest-layer', 'line-opacity', 0.4);
  }
}

// ---------------------------------------------------------------------------
// 7. IN-CAR TURN-BY-TURN NAVIGATION & DRIVE SIMULATION
// ---------------------------------------------------------------------------
function toggleNavigationMode() {
  if (appState.navRunning) {
    stopNavigation();
  } else {
    startNavigation();
  }
}

function startNavigation() {
  if (!appState.routesData) return;
  const activeRoute = appState.selectedRoute === 'safest' ? appState.routesData.routes.safest : appState.routesData.routes.fastest;
  const coords = activeRoute.geometry.coordinates; // [[lon, lat], ...]
  if (!coords || coords.length === 0) return;

  appState.navRunning = true;
  appState.simIndex = 0;

  // Show Turn-by-turn HUD
  document.getElementById('nav-hud-banner').classList.remove('hidden');
  document.getElementById('btn-start-nav').innerHTML = `
    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
      <rect x="6" y="6" width="12" height="12"></rect>
    </svg>
    Stop Navigation
  `;
  document.getElementById('sim-btn-label').textContent = 'Pause Sim';
  document.getElementById('sim-btn-icon').textContent = '⏸';

  // 3D Perspective Tilt (60 degrees) following vehicle
  appState.map.easeTo({
    center: coords[0],
    zoom: 16.5,
    pitch: 60,
    bearing: calculateBearing(coords[0], coords[Math.min(1, coords.length - 1)]),
    duration: 1200,
  });

  // Create or place Car Marker
  if (!appState.carMarker) {
    const el = document.createElement('div');
    el.className = 'car-marker';
    el.innerHTML = `
      <div style="width:26px; height:26px; background:#DAF1DE; border:3px solid #051F20; border-radius:50%; box-shadow:0 0 12px #8EB69B; display:flex; align-items:center; justify-content:center; font-size:12px;">
        ▲
      </div>
    `;
    appState.carMarker = new maplibregl.Marker({ element: el }).setLngLat(coords[0]).addTo(appState.map);
  }

  // Animation Loop every 1.5 seconds
  appState.simTimer = setInterval(() => {
    if (appState.simIndex >= coords.length - 1) {
      stopNavigation();
      return;
    }

    appState.simIndex++;
    const currentPt = coords[appState.simIndex];
    const nextPt = coords[Math.min(appState.simIndex + 1, coords.length - 1)];

    appState.carMarker.setLngLat(currentPt);

    const bearing = calculateBearing(currentPt, nextPt);
    appState.map.easeTo({
      center: currentPt,
      bearing: bearing,
      duration: 1200,
    });

    // Update HUD maneuver
    const remainingSteps = coords.length - appState.simIndex;
    const estMeters = remainingSteps * 25;
    document.getElementById('nav-dist-callout').textContent = estMeters > 1000 ? `In ${(estMeters / 1000).toFixed(1)} km` : `In ${estMeters} m`;

    // Query collision risk alert
    checkProximityAlert(currentPt[1], currentPt[0]);
  }, 1500);
}

function stopNavigation() {
  appState.navRunning = false;
  clearInterval(appState.simTimer);
  document.getElementById('nav-hud-banner').classList.add('hidden');
  document.getElementById('proximity-warning-card').classList.add('hidden');
  document.getElementById('btn-start-nav').innerHTML = `
    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 2L4.5 20.29l.71.71L12 18l6.79 3 .71-.71z"/>
    </svg>
    Start Navigation
  `;
  document.getElementById('sim-btn-label').textContent = 'Drive Simulation';
  document.getElementById('sim-btn-icon').textContent = '▶';

  // Reset 3D pitch to overview
  appState.map.easeTo({ pitch: 20, zoom: 13, duration: 1000 });
}

function calculateBearing(p1, p2) {
  const lon1 = p1[0] * Math.PI / 180;
  const lat1 = p1[1] * Math.PI / 180;
  const lon2 = p2[0] * Math.PI / 180;
  const lat2 = p2[1] * Math.PI / 180;
  const y = Math.sin(lon2 - lon1) * Math.cos(lat2);
  const x = Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(lat2) * Math.cos(lon2 - lon1);
  return (Math.atan2(y, x) * 180 / Math.PI + 360) % 360;
}

// ---------------------------------------------------------------------------
// 8. PROXIMITY HAZARD ALERT CHECK
// ---------------------------------------------------------------------------
async function checkProximityAlert(lat, lon) {
  try {
    const url = `/alert?lat=${lat}&lon=${lon}&hour=${appState.hour}&weather=${appState.weather}&month=${appState.month}`;
    const resp = await fetch(url);
    const alertData = await resp.json();

    const banner = document.getElementById('proximity-warning-card');
    if (alertData.alert) {
      document.getElementById('warn-message').textContent = `${alertData.message} (${alertData.distance_m} m)`;
      const badge = document.getElementById('warn-band-badge');
      badge.textContent = alertData.band === 'red' ? 'VERY HIGH RISK' : 'HIGH RISK';
      badge.style.background = alertData.band === 'red' ? '#E5484D' : '#F07F3C';

      banner.classList.remove('hidden');

      // Play audio chime once every 12 seconds
      const now = Date.now();
      if (now - appState.lastAlertTime > 12000) {
        playWarningChime();
        appState.lastAlertTime = now;
      }
    } else {
      banner.classList.add('hidden');
    }
  } catch (err) {
    console.error('Proximity alert check error:', err);
  }
}

// ---------------------------------------------------------------------------
// 9. SEGMENT INSPECTION MODAL
// ---------------------------------------------------------------------------
function showSegmentDetails(props) {
  document.getElementById('modal-road-title').textContent = props.name || `Road Segment #${props.segment_id}`;
  document.getElementById('modal-risk-val').textContent = props.risk_index.toFixed(3);

  const badge = document.getElementById('modal-risk-badge');
  badge.textContent = `${(props.band || 'RISK').toUpperCase()}`;
  badge.style.background = props.color || '#F07F3C';

  document.getElementById('modal-p-minor').textContent = `${Math.round(props.p_minor * 100)}%`;
  document.getElementById('modal-p-serious').textContent = `${Math.round(props.p_serious * 100)}%`;
  document.getElementById('modal-p-fatal').textContent = `${Math.round(props.p_fatal * 100)}%`;

  const driversList = document.getElementById('modal-drivers-list');
  driversList.innerHTML = '';
  const drivers = typeof props.top_drivers === 'string' ? JSON.parse(props.top_drivers) : (props.top_drivers || []);
  drivers.forEach((d) => {
    const li = document.createElement('li');
    li.textContent = d;
    driversList.appendChild(li);
  });

  document.getElementById('segment-modal').classList.remove('hidden');
}

// ---------------------------------------------------------------------------
// 10. GOOGLE MAPS SEARCH & AUTOCOMPLETE
// ---------------------------------------------------------------------------
function setupSearchAutocomplete() {
  const input = document.getElementById('map-search-input');
  const dropdown = document.getElementById('search-dropdown');
  const clearBtn = document.getElementById('btn-clear-search');

  input.addEventListener('input', () => {
    const q = input.value.trim().toLowerCase();
    if (!q) {
      dropdown.classList.add('hidden');
      return;
    }

    const matches = NAGPUR_LANDMARKS.filter(
      (lm) => lm.name.toLowerCase().includes(q) || lm.type.toLowerCase().includes(q)
    );

    if (matches.length === 0) {
      dropdown.innerHTML = `<div class="search-item"><small>No matching Nagpur locations</small></div>`;
    } else {
      dropdown.innerHTML = matches
        .map(
          (m, idx) => `
        <div class="search-item" data-idx="${idx}">
          <strong>${m.name}</strong>
          <small>${m.type}</small>
        </div>
      `
        )
        .join('');
    }
    dropdown.classList.remove('hidden');
  });

  dropdown.addEventListener('click', (e) => {
    const item = e.target.closest('.search-item');
    if (!item) return;
    const idx = parseInt(item.getAttribute('data-idx'));
    const lm = NAGPUR_LANDMARKS[idx];
    if (lm) {
      input.value = lm.name;
      dropdown.classList.add('hidden');

      // Set destination and fly to location
      appState.endCoord = lm.coords;
      document.getElementById('input-dest-loc').value = lm.name;
      appState.map.flyTo({ center: [lm.coords[1], lm.coords[0]], zoom: 14.5, duration: 1500 });
      calculateRoutes();
    }
  });

  clearBtn.addEventListener('click', () => {
    input.value = '';
    dropdown.classList.add('hidden');
  });

  document.addEventListener('click', (e) => {
    if (!e.target.closest('.search-cluster')) {
      dropdown.classList.add('hidden');
    }
  });
}

// ---------------------------------------------------------------------------
// 11. EVENT LISTENERS & WIRING
// ---------------------------------------------------------------------------
function setupEventListeners() {
  // Sidebar Tabs
  document.querySelectorAll('.tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach((b) => b.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach((p) => p.classList.remove('active'));
      btn.classList.add('active');
      const targetId = btn.getAttribute('data-tab');
      document.getElementById(targetId).classList.add('active');
    });
  });

  // Lambda Slider
  const sliderLambda = document.getElementById('slider-lambda');
  sliderLambda.addEventListener('input', (e) => {
    appState.lambda = parseFloat(e.target.value);
    document.getElementById('lambda-val-badge').textContent = `${appState.lambda.toFixed(2)}×`;
  });
  sliderLambda.addEventListener('change', () => {
    calculateRoutes();
  });

  // Route Selection Cards
  document.getElementById('card-safest').addEventListener('click', () => selectActiveRoute('safest'));
  document.getElementById('card-fastest').addEventListener('click', () => selectActiveRoute('fastest'));
  document.getElementById('btn-select-safest').addEventListener('click', (e) => { e.stopPropagation(); selectActiveRoute('safest'); });
  document.getElementById('btn-select-fastest').addEventListener('click', (e) => { e.stopPropagation(); selectActiveRoute('fastest'); });

  // Navigation Trigger
  document.getElementById('btn-start-nav').addEventListener('click', toggleNavigationMode);
  document.getElementById('btn-sim-drive').addEventListener('click', toggleNavigationMode);
  document.getElementById('btn-exit-nav').addEventListener('click', stopNavigation);

  // Quick Trip Chips
  document.querySelectorAll('.preset-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      const startParts = chip.getAttribute('data-start').split(',');
      const endParts = chip.getAttribute('data-end').split(',');
      appState.startCoord = [parseFloat(startParts[0]), parseFloat(startParts[1])];
      appState.endCoord = [parseFloat(endParts[0]), parseFloat(endParts[1])];
      document.getElementById('input-start-loc').value = chip.textContent.split('➔')[0].trim();
      document.getElementById('input-dest-loc').value = chip.textContent.split('➔')[1].trim();
      calculateRoutes();
    });
  });

  // Scenario Chips
  document.querySelectorAll('.scen-chip').forEach((chip) => {
    chip.addEventListener('click', () => {
      document.querySelectorAll('.scen-chip').forEach((c) => c.classList.remove('active'));
      chip.classList.add('active');
      appState.hour = parseInt(chip.getAttribute('data-hour'));
      appState.weather = chip.getAttribute('data-weather');
      appState.month = parseInt(chip.getAttribute('data-month'));

      document.getElementById('slider-hour').value = appState.hour;
      document.getElementById('val-hour').textContent = `${appState.hour.toString().padStart(2, '0')}:00`;
      document.getElementById('select-month').value = appState.month;
      document.querySelectorAll('.env-btn').forEach((b) => {
        b.classList.toggle('active', b.getAttribute('data-weather') === appState.weather);
      });

      fetchRiskNetwork();
      calculateRoutes();
    });
  });

  // Weather Buttons in Tab 2
  document.querySelectorAll('.env-btn').forEach((btn) => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.env-btn').forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      appState.weather = btn.getAttribute('data-weather');
      fetchRiskNetwork();
      calculateRoutes();
    });
  });

  // Hour Slider
  document.getElementById('slider-hour').addEventListener('input', (e) => {
    appState.hour = parseInt(e.target.value);
    document.getElementById('val-hour').textContent = `${appState.hour.toString().padStart(2, '0')}:00`;
  });
  document.getElementById('slider-hour').addEventListener('change', () => {
    fetchRiskNetwork();
    calculateRoutes();
  });

  // Month Selector
  document.getElementById('select-month').addEventListener('change', (e) => {
    appState.month = parseInt(e.target.value);
    fetchRiskNetwork();
    calculateRoutes();
  });

  // Basemap Dropdown
  const basemapBtn = document.getElementById('btn-basemap');
  const basemapMenu = document.getElementById('basemap-menu');
  basemapBtn.addEventListener('click', (e) => {
    e.stopPropagation();
    basemapMenu.classList.toggle('hidden');
  });
  document.addEventListener('click', () => basemapMenu.classList.add('hidden'));

  document.querySelectorAll('.basemap-opt').forEach((opt) => {
    opt.addEventListener('click', () => {
      document.querySelectorAll('.basemap-opt').forEach((o) => o.classList.remove('active'));
      opt.classList.add('active');
      const layerKey = opt.getAttribute('data-layer');
      setBasemapLayer(layerKey);
    });
  });

  // Floating Map Controls (Google Maps Style)
  document.getElementById('btn-zoom-in').addEventListener('click', () => appState.map.zoomIn());
  document.getElementById('btn-zoom-out').addEventListener('click', () => appState.map.zoomOut());
  document.getElementById('btn-reset-north').addEventListener('click', () => appState.map.resetNorthPitch());
  
  let is3D = false;
  document.getElementById('btn-toggle-3d').addEventListener('click', () => {
    is3D = !is3D;
    appState.map.easeTo({ pitch: is3D ? 60 : 0, duration: 800 });
  });

  document.getElementById('btn-fit-route').addEventListener('click', () => {
    if (appState.routesData) {
      const coords = appState.routesData.routes.safest.geometry.coordinates;
      const bounds = coords.reduce(
        (b, coord) => b.extend(coord),
        new maplibregl.LngLatBounds(coords[0], coords[0])
      );
      appState.map.fitBounds(bounds, { padding: 80, duration: 1000 });
    }
  });

  // Modal Close
  document.getElementById('btn-close-modal').addEventListener('click', () => {
    document.getElementById('segment-modal').classList.add('hidden');
  });

  // ESP32 Live Telemetry Polling
  const liveBtn = document.getElementById('btn-toggle-live');
  liveBtn.addEventListener('click', () => {
    appState.livePolling = !appState.livePolling;
    liveBtn.classList.toggle('active', appState.livePolling);
    document.getElementById('live-status-text').textContent = appState.livePolling ? 'ESP32: Live' : 'ESP32 Feed';

    if (appState.livePolling) {
      pollLiveTelemetry();
      appState.liveTimer = setInterval(pollLiveTelemetry, 3000);
    } else {
      clearInterval(appState.liveTimer);
    }
  });

  // Inject sample telemetry button
  document.getElementById('btn-inject-sample').addEventListener('click', async () => {
    try {
      const payload = {
        lat: 21.1442,
        lon: 79.0664,
        speed: 48.0,
        rain_sensor: 780.0,
        ldr: 210.0,
        harsh_brake: 1,
      };
      await fetch('/ingest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      pollLiveTelemetry();
    } catch (e) {
      console.warn('Sample injection notice:', e);
    }
  });
}

async function pollLiveTelemetry() {
  try {
    const resp = await fetch('/live');
    const data = await resp.json();
    document.getElementById('iot-val-speed').textContent = `${data.speed.toFixed(1)} km/h`;
    document.getElementById('iot-val-brake').textContent = data.harsh_brake ? '⚠️ HARSH BRAKE (>0.4g)' : 'Normal (0.14g)';
    document.getElementById('iot-val-rain').textContent = `${Math.round(data.rain_sensor)} (${data.inferred_weather})`;
    document.getElementById('iot-val-ldr').textContent = `${Math.round(data.ldr)} (${data.inferred_light})`;
    document.getElementById('iot-val-lat').textContent = data.lat.toFixed(5);
    document.getElementById('iot-val-lon').textContent = data.lon.toFixed(5);
    document.getElementById('iot-val-segment').textContent = data.nearest_segment_id ? `#${data.nearest_segment_id}` : 'Snapping...';
  } catch (err) {
    console.error('Error polling live telemetry:', err);
  }
}

// ---------------------------------------------------------------------------
// 12. INITIALIZATION ON DOM READY
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
  initMap();
  setupSearchAutocomplete();
  setupEventListeners();
});
