/**
 * network.js — Neural cluster / synapse-network visualization for the Samantha presence surface.
 *
 * Visual concept: looking into a thinking brain.
 *   - ~400 neurons distributed in an organic, slightly asymmetric blob
 *   - Faint web of synapse edges (just-barely-visible skeleton)
 *   - Spark beads travel along edges with short trails; destination neuron flashes on arrival
 *   - Every agent state drives a distinct firing pattern, hue, and intensity
 *   - UnrealBloomPass amplifies the glow so neurons feel like bioluminescent cells
 */

import * as THREE from 'three';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass }     from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { OutputPass }     from 'three/addons/postprocessing/OutputPass.js';

// ── Constants ──────────────────────────────────────────────────────────────────

const NEURON_COUNT  = 420;
const K_NEIGHBORS   = 4;      // edges per neuron
const MAX_SPARKS    = 80;     // pooled spark slots
const SPARK_SPEED   = 0.9;    // world-units per second along edge
const FLASH_DECAY   = 3.5;    // brightness decay rate for hit-flash
const CHAIN_MAX     = 2;      // max chain-reaction depth
const CHAIN_PROB    = 0.35;   // probability of secondary spark on arrival

// State presets — lerp targets for (hue 0-1, saturation 0-1, brightness 0-1, sparkRate sparks/s)
const PRESETS = {
  idle:      { hue: 0.60, sat: 0.5,  bright: 0.55, edgeAlpha: 0.05, sparkRate: 0.8,  bloomStr: 0.7  },
  listening: { hue: 0.58, sat: 0.75, bright: 0.80, edgeAlpha: 0.08, sparkRate: 3.0,  bloomStr: 1.0  },
  thinking:  { hue: 0.72, sat: 0.70, bright: 0.90, edgeAlpha: 0.10, sparkRate: 6.5,  bloomStr: 1.2  },
  speaking:  { hue: 0.07, sat: 0.70, bright: 0.85, edgeAlpha: 0.07, sparkRate: 4.0,  bloomStr: 1.0  },
  wake:      { hue: 0.11, sat: 0.90, bright: 1.00, edgeAlpha: 0.15, sparkRate: 10.0, bloomStr: 1.6  },
  quiet:     { hue: 0.55, sat: 0.20, bright: 0.30, edgeAlpha: 0.03, sparkRate: 0.2,  bloomStr: 0.4  },
  sleep:     { hue: 0.60, sat: 0.10, bright: 0.12, edgeAlpha: 0.01, sparkRate: 0.05, bloomStr: 0.2  },
};

const LERP_RATE = 2.2;  // exponential lerp speed (per second)

// ── Neuron point shader ────────────────────────────────────────────────────────

const NEURON_VERT = /* glsl */`
  attribute float aBright;
  attribute vec3  aColor;
  attribute float aSize;
  varying   float vBrightness;
  varying   vec3  vColor;

  void main() {
    vBrightness = aBright;
    vColor      = aColor;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = aSize * (400.0 / -mv.z);
    gl_Position  = projectionMatrix * mv;
  }
`;

const NEURON_FRAG = /* glsl */`
  varying float vBrightness;
  varying vec3  vColor;

  void main() {
    float d    = length(gl_PointCoord - vec2(0.5));
    float halo = smoothstep(0.5, 0.0, d);
    float core = smoothstep(0.15, 0.0, d);
    if (halo < 0.01) discard;
    vec3 col = vColor * (halo * 0.6 + core * 1.6) * vBrightness;
    gl_FragColor = vec4(col, halo);
  }
`;

// Spark shader — same halo shape, slightly bigger
const SPARK_VERT = /* glsl */`
  attribute float aBright;
  attribute vec3  aColor;
  varying   float vBrightness;
  varying   vec3  vColor;

  void main() {
    vBrightness = aBright;
    vColor      = aColor;
    vec4 mv = modelViewMatrix * vec4(position, 1.0);
    gl_PointSize = 14.0 * (400.0 / -mv.z);
    gl_Position  = projectionMatrix * mv;
  }
`;

// ── Helpers ───────────────────────────────────────────────────────────────────

function lerp(a, b, t) { return a + (b - a) * Math.min(t, 1.0); }

/** Lerp hue on the 0..1 circle (shortest arc). */
function lerpHue(a, b, t) {
  let d = b - a;
  if (d >  0.5) d -= 1.0;
  if (d < -0.5) d += 1.0;
  return ((a + d * Math.min(t, 1.0)) + 1.0) % 1.0;
}

/** HSL (all 0-1) → RGB THREE.Color */
function hsl(h, s, l) {
  return new THREE.Color().setHSL(h, s, l);
}

/** Cheap 3-D hash → [0,1) */
function hash3(x, y, z) {
  let n = Math.sin(x * 127.1 + y * 311.7 + z * 74.7) * 43758.5453;
  return n - Math.floor(n);
}

/** Pseudo-simplex density for organic blob shaping. Returns 0-1. */
function blobDensity(x, y, z) {
  // Soft sphere with slight asymmetry baked in via skewed axes
  const r2 = x * x + y * y * 1.1 + z * z * 0.9;
  const noise = (hash3(x * 2.1, y * 1.9, z * 2.3) - 0.5) * 0.35;
  return Math.exp(-r2 * 0.8 + noise);
}

// ── NeuralNetwork class ────────────────────────────────────────────────────────

export class NeuralNetwork {
  /** @param {HTMLCanvasElement} canvas */
  constructor(canvas) {
    this._canvas = canvas;

    // ── Renderer
    this._renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
    this._renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this._renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this._renderer.toneMappingExposure = 1.0;

    // ── Scene / Camera
    this._scene  = new THREE.Scene();
    this._scene.background = new THREE.Color(0x06080d);

    this._camera = new THREE.PerspectiveCamera(50, 1, 0.1, 200);
    this._camera.position.z = 6.5;

    // ── Post-processing
    this._composer = null;  // built in _resize
    this._bloomPass = new UnrealBloomPass(
      new THREE.Vector2(window.innerWidth, window.innerHeight),
      0.8, 0.6, 0.0
    );

    // ── State
    this._state  = 'idle';
    this._cur    = { ...PRESETS.idle };   // current lerped values
    this._tgt    = { ...PRESETS.idle };   // target values
    this._spawnTimer = 0;  // seconds until next spark

    // ── Build geometry
    this._buildNeurons();
    this._buildEdges();
    this._buildSparks();

    // ── Resize
    this._resize = this._resize.bind(this);
    window.addEventListener('resize', this._resize);
    this._resize();

    // ── Slow rotation group
    this._group = new THREE.Group();
    this._group.add(this._neuronMesh);
    this._group.add(this._edgeMesh);
    this._scene.add(this._group);
    this._scene.add(this._sparkMesh);  // sparks stay in world space (simpler math)

    // ── RAF
    this._lastT = performance.now();
    this._loop  = this._loop.bind(this);
    requestAnimationFrame(this._loop);
  }

  // ── Public API ────────────────────────────────────────────────────────────────

  setState(value) {
    this._state = value;
    this._tgt   = { ...(PRESETS[value] ?? PRESETS.idle) };

    if (value === 'wake') {
      this._triggerWakeFlash();
    }
  }

  /** Manual outward burst — called by speaking response_chunk events. */
  pulse(strength = 0.4) {
    const count = Math.round(strength * 8);
    for (let i = 0; i < count; i++) {
      this._spawnSpark('outward', null, [1.0, 0.75, 0.55]);
    }
  }

  /** Visual cue for memory_recall — one bright gold spark between two random neurons. */
  recallSpark() {
    this._spawnSpark('random', null, [1.0, 0.88, 0.45]);
  }

  // ── Geometry builders ─────────────────────────────────────────────────────────

  _buildNeurons() {
    const n = NEURON_COUNT;
    const positions = new Float32Array(n * 3);
    const brights   = new Float32Array(n);
    const colors    = new Float32Array(n * 3);
    const sizes     = new Float32Array(n);

    // Rejection sampling to fill organic blob
    let filled = 0;
    const attempts = n * 30;
    for (let i = 0; i < attempts && filled < n; i++) {
      const x = (Math.random() - 0.5) * 4.2;
      const y = (Math.random() - 0.5) * 3.8;
      const z = (Math.random() - 0.5) * 3.4;
      const density = blobDensity(x, y, z);
      if (Math.random() < density) {
        const idx = filled * 3;
        positions[idx]     = x;
        positions[idx + 1] = y;
        positions[idx + 2] = z;

        // Hue variation: slightly blue-cool at extremes, peach-warm near center
        const r2 = x*x + y*y + z*z;
        const h  = 0.55 + (Math.random() - 0.5) * 0.12 - r2 * 0.015;
        const c  = hsl(h, 0.6, 0.75);
        colors[idx]     = c.r;
        colors[idx + 1] = c.g;
        colors[idx + 2] = c.b;

        brights[filled] = 0.4 + Math.random() * 0.4;
        sizes[filled]   = 0.8 + Math.random() * 1.2;
        filled++;
      }
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geo.setAttribute('aBright',  new THREE.BufferAttribute(brights, 1));
    geo.setAttribute('aColor',   new THREE.BufferAttribute(colors, 3));
    geo.setAttribute('aSize',    new THREE.BufferAttribute(sizes, 1));

    // Original ShaderMaterial — restored now that the `now` ReferenceError
    // bug in _update() (which broke every frame) is fixed.
    const mat = new THREE.ShaderMaterial({
      vertexShader:   NEURON_VERT,
      fragmentShader: NEURON_FRAG,
      transparent:    true,
      depthWrite:     false,
      blending:       THREE.AdditiveBlending,
    });

    this._neuronMesh  = new THREE.Points(geo, mat);
    this._neuronPos   = positions;   // keep reference for edge building + spark routing
    this._neuronBright = brights;
    this._neuronColor  = colors;
    this._neuronCount  = filled;
    this._neuronFlash  = new Float32Array(filled);  // per-neuron flash value 0-1
  }

  _buildEdges() {
    const n = this._neuronCount;
    const pos = this._neuronPos;

    // For each neuron, find K nearest neighbors using brute-force (one-time, ~O(n*k) with early exit)
    this._adjacency = Array.from({ length: n }, () => []);

    const edgeSet = new Set();
    const edgePositions = [];

    for (let i = 0; i < n; i++) {
      const xi = pos[i*3], yi = pos[i*3+1], zi = pos[i*3+2];
      // Collect distances to all others
      const dists = [];
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const dx = pos[j*3]-xi, dy = pos[j*3+1]-yi, dz = pos[j*3+2]-zi;
        dists.push({ j, d2: dx*dx+dy*dy+dz*dz });
      }
      dists.sort((a, b) => a.d2 - b.d2);

      for (let k = 0; k < K_NEIGHBORS && k < dists.length; k++) {
        const j = dists[k].j;
        const key = i < j ? `${i}-${j}` : `${j}-${i}`;
        if (!edgeSet.has(key)) {
          edgeSet.add(key);
          this._adjacency[i].push(j);
          this._adjacency[j].push(i);
          edgePositions.push(
            pos[i*3], pos[i*3+1], pos[i*3+2],
            pos[j*3], pos[j*3+1], pos[j*3+2]
          );
        }
      }
    }

    this._edges = [];  // array of {a, b} neuron indices for spark routing
    for (const key of edgeSet) {
      const [a, b] = key.split('-').map(Number);
      this._edges.push({ a, b });
    }

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(edgePositions), 3));
    this._edgePosAttr = geo.getAttribute('position');

    const mat = new THREE.LineBasicMaterial({
      color: 0x7ab3d4,
      transparent: true,
      opacity: 0.05,
      blending: THREE.AdditiveBlending,
      depthWrite: false,
    });

    this._edgeMesh  = new THREE.LineSegments(geo, mat);
    this._edgeMat   = mat;
  }

  _buildSparks() {
    // Object pool of MAX_SPARKS spark slots
    this._sparks = Array.from({ length: MAX_SPARKS }, () => ({
      active:    false,
      edgeIdx:   0,       // index into this._edges
      t:         0,       // 0 = neuron A, 1 = neuron B
      dir:       1,       // +1 or -1
      color:     [1, 1, 1],
      chainDepth: 0,
    }));

    const positions = new Float32Array(MAX_SPARKS * 3);
    const brights   = new Float32Array(MAX_SPARKS);
    const colors    = new Float32Array(MAX_SPARKS * 3);

    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3).setUsage(THREE.DynamicDrawUsage));
    geo.setAttribute('aBright',  new THREE.BufferAttribute(brights, 1).setUsage(THREE.DynamicDrawUsage));
    geo.setAttribute('aColor',   new THREE.BufferAttribute(colors, 3).setUsage(THREE.DynamicDrawUsage));

    const mat = new THREE.ShaderMaterial({
      vertexShader:   SPARK_VERT,
      fragmentShader: NEURON_FRAG,
      transparent:    true,
      depthWrite:     false,
      blending:       THREE.AdditiveBlending,
    });

    this._sparkMesh    = new THREE.Points(geo, mat);
    this._sparkPosAttr = geo.getAttribute('position');
    this._sparkBright  = geo.getAttribute('aBright');
    this._sparkColor   = geo.getAttribute('aColor');
  }

  // ── Spark spawning ────────────────────────────────────────────────────────────

  /** @param {'random'|'inward'|'outward'|'chaotic'} mode */
  _spawnSpark(mode, chainDepth = 0, colorOverride = null) {
    // Find a free slot
    const slot = this._sparks.find(s => !s.active);
    if (!slot) return;

    const n   = this._neuronCount;
    const pos = this._neuronPos;

    let edgeIdx, dir;

    if (mode === 'inward') {
      // Pick a peripheral neuron (high |pos|) as origin
      const outer = this._peripheryNeurons();
      const fromN = outer[Math.floor(Math.random() * outer.length)];
      // Pick an edge from fromN toward center
      const neigh = this._adjacency[fromN];
      if (!neigh.length) return;
      const toN = neigh[Math.floor(Math.random() * neigh.length)];
      edgeIdx = this._findEdge(fromN, toN);
      dir = (this._edges[edgeIdx].a === fromN) ? 1 : -1;

    } else if (mode === 'outward') {
      // Origin near cluster center
      const inner = this._centerNeurons();
      const fromN = inner[Math.floor(Math.random() * inner.length)];
      const neigh = this._adjacency[fromN];
      if (!neigh.length) return;
      const toN = neigh[Math.floor(Math.random() * neigh.length)];
      edgeIdx = this._findEdge(fromN, toN);
      dir = (this._edges[edgeIdx].a === fromN) ? 1 : -1;

    } else {
      // Random edge, random direction
      edgeIdx = Math.floor(Math.random() * this._edges.length);
      dir     = Math.random() < 0.5 ? 1 : -1;
    }

    // Pick spark hue from current state
    let sparkColor;
    if (colorOverride) {
      sparkColor = colorOverride;
    } else {
      const h = this._cur.hue;
      const c = hsl(h, 0.8, 0.85);
      sparkColor = [c.r, c.g, c.b];
    }

    slot.active     = true;
    slot.edgeIdx    = edgeIdx;
    slot.t          = dir === 1 ? 0 : 1;
    slot.dir        = dir;
    slot.color      = sparkColor;
    slot.chainDepth = chainDepth ?? 0;
  }

  _triggerWakeFlash() {
    // Central flash: briefly max out all neuron brights
    this._wakeFlash = 1.0;
    // Radiate ~10 outward sparks
    for (let i = 0; i < 10; i++) {
      this._spawnSpark('outward', 0, [1.0, 0.85, 0.4]);
    }
  }

  /** Cached periphery neuron indices (top 20% by distance from center). */
  _peripheryNeurons() {
    if (this._periCache) return this._periCache;
    const pos = this._neuronPos;
    const n   = this._neuronCount;
    const r2s = Array.from({ length: n }, (_, i) => ({
      i, r2: pos[i*3]*pos[i*3] + pos[i*3+1]*pos[i*3+1] + pos[i*3+2]*pos[i*3+2]
    }));
    r2s.sort((a, b) => b.r2 - a.r2);
    this._periCache = r2s.slice(0, Math.floor(n * 0.2)).map(x => x.i);
    return this._periCache;
  }

  /** Cached center neuron indices (inner 15%). */
  _centerNeurons() {
    if (this._centerCache) return this._centerCache;
    const pos = this._neuronPos;
    const n   = this._neuronCount;
    const r2s = Array.from({ length: n }, (_, i) => ({
      i, r2: pos[i*3]*pos[i*3] + pos[i*3+1]*pos[i*3+1] + pos[i*3+2]*pos[i*3+2]
    }));
    r2s.sort((a, b) => a.r2 - b.r2);
    this._centerCache = r2s.slice(0, Math.floor(n * 0.15)).map(x => x.i);
    return this._centerCache;
  }

  _findEdge(a, b) {
    // Find the edge index for the pair (a,b) — O(adjacency) lookup is fine
    for (let i = 0; i < this._edges.length; i++) {
      const e = this._edges[i];
      if ((e.a === a && e.b === b) || (e.a === b && e.b === a)) return i;
    }
    return Math.floor(Math.random() * this._edges.length);
  }

  // ── Resize + composer ─────────────────────────────────────────────────────────

  _resize() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    this._renderer.setSize(w, h, false);
    this._camera.aspect = w / h;
    this._camera.updateProjectionMatrix();

    // Rebuild composer on resize so bloom resolution matches.
    // Order: RenderPass → BloomPass → OutputPass (sRGB conversion + tone mapping).
    // Without OutputPass on Three.js r163+, the linear-color buffer goes to
    // screen unconverted and everything reads as near-black.
    this._composer = new EffectComposer(this._renderer);
    this._composer.addPass(new RenderPass(this._scene, this._camera));
    this._bloomPass.resolution.set(w, h);
    this._composer.addPass(this._bloomPass);
    this._composer.addPass(new OutputPass());
    this._composer.setSize(w, h);
  }

  // ── Main loop ─────────────────────────────────────────────────────────────────

  _loop() {
    requestAnimationFrame(this._loop);

    const now = performance.now();
    const dt  = Math.min((now - this._lastT) / 1000, 0.05);
    this._lastT = now;

    this._update(dt);
    if (this._composer) this._composer.render();
    else                this._renderer.render(this._scene, this._camera);
  }

  _update(dt) {
    // 1. Lerp current toward target
    const c = this._cur, t = this._tgt, k = 1 - Math.exp(-LERP_RATE * dt);
    c.hue       = lerpHue(c.hue, t.hue, k);
    c.sat       = lerp(c.sat,       t.sat,       k);
    c.bright    = lerp(c.bright,    t.bright,    k);
    c.edgeAlpha = lerp(c.edgeAlpha, t.edgeAlpha, k);
    c.sparkRate = lerp(c.sparkRate, t.sparkRate, k);
    c.bloomStr  = lerp(c.bloomStr,  t.bloomStr,  k);

    // 2. Update bloom strength
    this._bloomPass.strength = c.bloomStr;

    const now = performance.now();

    // 3. Slow cluster rotation (30-second full cycle ≈ 0.21 rad/s)
    this._group.rotation.y += dt * 0.021;
    this._group.rotation.x  = Math.sin(now * 0.0001) * 0.08;

    // 4. Neuron ambient flicker + flash decay
    const brights  = this._neuronBright;
    const colors   = this._neuronColor;
    const flashes  = this._neuronFlash;
    const baseH    = c.hue;
    const n        = this._neuronCount;

    for (let i = 0; i < n; i++) {
      // Slow independent flicker using deterministic hash as phase offset
      const phase   = hash3(i * 0.37, i * 0.71, i * 0.13) * Math.PI * 2;
      const flicker = 0.4 + 0.3 * (0.5 + 0.5 * Math.sin(now * 0.001 * (0.8 + hash3(i, 0, 0) * 0.4) + phase));

      // Flash contribution
      const flash = flashes[i];
      flashes[i]  = Math.max(0, flash - dt * FLASH_DECAY);

      brights[i] = (flicker + flash) * c.bright;

      // Hue slight variation per-neuron (baked offset preserved)
      const hOff = (hash3(i * 1.1, i * 0.9, 0) - 0.5) * 0.12;
      const hh   = ((baseH + hOff) + 1) % 1;
      const col  = hsl(hh, c.sat, 0.7);
      colors[i*3]   = col.r;
      colors[i*3+1] = col.g;
      colors[i*3+2] = col.b;
    }
    this._neuronMesh.geometry.getAttribute('aBright').needsUpdate = true;
    this._neuronMesh.geometry.getAttribute('aColor').needsUpdate  = true;

    // 5. Wake flash global brightness overlay
    if (this._wakeFlash > 0) {
      this._wakeFlash = Math.max(0, this._wakeFlash - dt * 2.5);
      for (let i = 0; i < n; i++) {
        brights[i] = Math.min(1.0, brights[i] + this._wakeFlash * 0.8);
      }
      this._neuronMesh.geometry.getAttribute('aBright').needsUpdate = true;
    }

    // 6. Edge alpha
    this._edgeMat.opacity = c.edgeAlpha;

    // 7. Spark spawning
    this._spawnTimer -= dt;
    if (this._spawnTimer <= 0 && c.sparkRate > 0.01) {
      this._spawnTimer = 1.0 / c.sparkRate + (Math.random() - 0.5) * (0.5 / c.sparkRate);
      const mode = this._spawnModeForState();
      this._spawnSpark(mode);

      // Thinking: occasional burst
      if (this._state === 'thinking' && Math.random() < 0.15) {
        const burst = 3 + Math.floor(Math.random() * 5);
        for (let b = 0; b < burst; b++) this._spawnSpark(mode);
      }
    }

    // 8. Advance sparks
    this._updateSparks(dt);
  }

  _spawnModeForState() {
    switch (this._state) {
      case 'listening': return 'inward';
      case 'speaking':  return 'outward';
      default:          return 'random';
    }
  }

  _updateSparks(dt) {
    const pos    = this._neuronPos;
    const sparks = this._sparks;
    const sPos   = this._sparkPosAttr;
    const sBri   = this._sparkBright;
    const sCol   = this._sparkColor;

    for (let i = 0; i < MAX_SPARKS; i++) {
      const s = sparks[i];
      if (!s.active) {
        // Park offscreen
        sPos.setXYZ(i, 0, -999, 0);
        sBri.setX(i, 0);
        continue;
      }

      const edge = this._edges[s.edgeIdx];
      const aIdx = edge.a * 3;
      const bIdx = edge.b * 3;

      // Advance t along edge
      const ax = pos[aIdx],   ay = pos[aIdx+1], az = pos[aIdx+2];
      const bx = pos[bIdx],   by = pos[bIdx+1], bz = pos[bIdx+2];
      const edgeLen = Math.sqrt((bx-ax)**2 + (by-ay)**2 + (bz-az)**2);
      const dtT = (SPARK_SPEED * dt) / Math.max(edgeLen, 0.01);
      s.t += s.dir * dtT;

      // Interpolate position along edge
      const tc  = Math.max(0, Math.min(1, s.t));
      const wx  = ax + (bx - ax) * tc;
      const wy  = ay + (by - ay) * tc;
      const wz  = az + (bz - az) * tc;

      // Transform to world space (accounting for group rotation)
      const local = new THREE.Vector3(wx, wy, wz);
      const world = local.applyMatrix4(this._group.matrixWorld);
      sPos.setXYZ(i, world.x, world.y, world.z);
      sBri.setX(i, 1.0);
      sCol.setXYZ(i, s.color[0], s.color[1], s.color[2]);

      // Check arrival
      const arrived = (s.dir === 1 && s.t >= 1.0) || (s.dir === -1 && s.t <= 0.0);
      if (arrived) {
        const destNeuron = s.dir === 1 ? edge.b : edge.a;
        // Flash destination neuron
        this._neuronFlash[destNeuron] = Math.min(1.0, this._neuronFlash[destNeuron] + 0.9);

        // Chain reaction
        if (s.chainDepth < CHAIN_MAX && Math.random() < CHAIN_PROB) {
          const neigh = this._adjacency[destNeuron];
          if (neigh.length) {
            const nextN    = neigh[Math.floor(Math.random() * neigh.length)];
            const nextEdge = this._findEdge(destNeuron, nextN);
            const freeSlot = sparks.find(ss => !ss.active);
            if (freeSlot) {
              freeSlot.active     = true;
              freeSlot.edgeIdx    = nextEdge;
              freeSlot.dir        = this._edges[nextEdge].a === destNeuron ? 1 : -1;
              freeSlot.t          = this._edges[nextEdge].a === destNeuron ? 0 : 1;
              freeSlot.color      = s.color;
              freeSlot.chainDepth = s.chainDepth + 1;
            }
          }
        }
        s.active = false;
      }
    }

    sPos.needsUpdate = true;
    sBri.needsUpdate = true;
    sCol.needsUpdate = true;
  }
}
