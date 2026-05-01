/**
 * orb.js — Three.js shader orb for the Samantha presence surface.
 *
 * Aesthetic: fluid, ambient, warm.  No hard edges.  Organic breathing on idle.
 * Each agent state drives a preset (hue / intensity) that the uniforms lerp toward.
 *
 * Noise implementation: 3-D simplex noise by Ashima Arts
 *   Ian McEwan, Ashima Arts — https://github.com/ashima/webgl-noise
 *   MIT License (see VERTEX shader comment).
 */

import * as THREE from 'three';

// ── State presets ─────────────────────────────────────────────────────────────
// hue is 0..1 (HSL).  intensity 0..1 drives displacement + brightness.
const PRESETS = {
  idle:      { intensity: 0.20, hue: 0.08, hueShift: 0.00 },  // warm peach / amber
  listening: { intensity: 0.55, hue: 0.55, hueShift: 0.00 },  // cool teal-blue
  thinking:  { intensity: 0.35, hue: 0.78, hueShift: 0.05 },  // soft lavender, slow swirl
  speaking:  { intensity: 0.70, hue: 0.06, hueShift: 0.00 },  // bright coral / warm
  quiet:     { intensity: 0.10, hue: 0.00, hueShift: 0.00 },  // near-gray, dim
  wake:      { intensity: 1.00, hue: 0.10, hueShift: 0.00 },  // bright amber flash
  sleep:     { intensity: 0.05, hue: 0.00, hueShift: 0.00 },  // barely visible
};

// Lerp speed constants (per-second rate for exponential lerp)
const INTENSITY_SPEED = 3.0;
const HUE_SPEED       = 2.0;
const PULSE_DECAY     = 1.5;  // units per second

// ── Orb class ─────────────────────────────────────────────────────────────────
export class Orb {
  /**
   * @param {HTMLCanvasElement} canvas
   */
  constructor(canvas) {
    this._canvas  = canvas;
    this._target  = { ...PRESETS.idle };

    // ── Renderer
    this._renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: true,
    });
    this._renderer.setPixelRatio(Math.min(devicePixelRatio, 2));

    // ── Scene / Camera
    this._scene  = new THREE.Scene();
    this._camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
    this._camera.position.z = 3;

    // ── Geometry — high-poly icosahedron so displacement is smooth
    const geometry = new THREE.IcosahedronGeometry(1, 64);

    // ── Uniforms
    this.uniforms = {
      uTime:      { value: 0.0 },
      uIntensity: { value: PRESETS.idle.intensity },
      uHue:       { value: PRESETS.idle.hue },
      uHueShift:  { value: 0.0 },
      uPulse:     { value: 0.0 },
    };

    // ── Material (custom shader)
    const material = new THREE.ShaderMaterial({
      uniforms:       this.uniforms,
      vertexShader:   VERTEX_SHADER,
      fragmentShader: FRAGMENT_SHADER,
      transparent:    true,
      side:           THREE.FrontSide,
    });

    this._mesh = new THREE.Mesh(geometry, material);
    this._scene.add(this._mesh);

    // ── Resize
    this._resize = this._resize.bind(this);
    window.addEventListener('resize', this._resize);
    this._resize();

    // ── RAF loop
    this._lastT  = performance.now();
    this._raf    = null;
    this._loop   = this._loop.bind(this);
    this._loop();
  }

  // ── Public API ───────────────────────────────────────────────────────────────

  /**
   * Transition the orb to the given state.
   * @param {'idle'|'listening'|'thinking'|'speaking'|'quiet'|'wake'|'sleep'} value
   */
  setState(value) {
    this._target = { ...(PRESETS[value] ?? PRESETS.idle) };
  }

  /**
   * Add a short amplitude burst.
   * Call on response_chunk, memory_recall, or wake events.
   * @param {number} strength  0..1
   */
  pulse(strength = 0.6) {
    this.uniforms.uPulse.value = Math.min(
      this.uniforms.uPulse.value + strength,
      1.0,
    );
  }

  /** Release RAF and event listener (call if you ever unmount). */
  dispose() {
    if (this._raf !== null) cancelAnimationFrame(this._raf);
    window.removeEventListener('resize', this._resize);
    this._renderer.dispose();
  }

  // ── Internal ─────────────────────────────────────────────────────────────────

  _resize() {
    const w = window.innerWidth;
    const h = window.innerHeight;
    this._renderer.setSize(w, h, false);
    this._camera.aspect = w / h;
    this._camera.updateProjectionMatrix();
  }

  _loop() {
    this._raf = requestAnimationFrame(this._loop);

    const now = performance.now();
    const dt  = Math.min((now - this._lastT) / 1000, 0.05);  // cap to avoid spiral on tab hide
    this._lastT = now;

    // Advance time
    this.uniforms.uTime.value += dt;

    // Exponential lerp uniforms toward target
    const u = this.uniforms;
    const t = this._target;
    u.uIntensity.value = _lerp(u.uIntensity.value, t.intensity, dt * INTENSITY_SPEED);
    u.uHue.value       = _lerpHue(u.uHue.value,    t.hue,       dt * HUE_SPEED);
    u.uHueShift.value  = t.hueShift;

    // Pulse decay
    u.uPulse.value = Math.max(0.0, u.uPulse.value - dt * PULSE_DECAY);

    // Very slow Y rotation — gives the orb a gentle life even at idle
    this._mesh.rotation.y += dt * 0.04;

    this._renderer.render(this._scene, this._camera);
  }
}

// ── Helpers ──────────────────────────────────────────────────────────────────

function _lerp(a, b, t) {
  return a + (b - a) * Math.min(t, 1.0);
}

/** Lerp hue on the 0..1 circle (takes shortest arc). */
function _lerpHue(a, b, t) {
  let diff = b - a;
  if (diff > 0.5)  diff -= 1.0;
  if (diff < -0.5) diff += 1.0;
  return (a + diff * Math.min(t, 1.0) + 1.0) % 1.0;
}

// ── GLSL shaders ─────────────────────────────────────────────────────────────

// Language: GLSL 300 es (Three.js ShaderMaterial uses WebGL2 by default when available,
// but ShaderMaterial raw GLSL works on WebGL1 too as long as we stay compatible).

const VERTEX_SHADER = /* glsl */`
/*
 * 3-D Simplex noise — Ashima Arts (Ian McEwan, Stefan Gustavson)
 * https://github.com/ashima/webgl-noise
 * MIT License
 */
vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289(((x * 34.0) + 1.0) * x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float snoise(vec3 v) {
  const vec2 C = vec2(1.0/6.0, 1.0/3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

  // First corner
  vec3 i  = floor(v + dot(v, C.yyy));
  vec3 x0 = v   - i + dot(i, C.xxx);

  // Other corners
  vec3 g  = step(x0.yzx, x0.xyz);
  vec3 l  = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);

  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - D.yyy;

  // Permutations
  i = mod289(i);
  vec4 p = permute(permute(permute(
    i.z + vec4(0.0, i1.z, i2.z, 1.0)) +
    i.y + vec4(0.0, i1.y, i2.y, 1.0)) +
    i.x + vec4(0.0, i1.x, i2.x, 1.0));

  // Gradients (7x7 points over a square, mapped to octahedron)
  float n_ = 0.142857142857;
  vec3  ns  = n_ * D.wyz - D.xzx;
  vec4 j    = p - 49.0 * floor(p * ns.z * ns.z);
  vec4 x_   = floor(j * ns.z);
  vec4 y_   = floor(j - 7.0 * x_);
  vec4 x    = x_ * ns.x + ns.yyyy;
  vec4 y    = y_ * ns.x + ns.yyyy;
  vec4 h    = 1.0 - abs(x) - abs(y);
  vec4 b0   = vec4(x.xy, y.xy);
  vec4 b1   = vec4(x.zw, y.zw);
  vec4 s0   = floor(b0) * 2.0 + 1.0;
  vec4 s1   = floor(b1) * 2.0 + 1.0;
  vec4 sh   = -step(h, vec4(0.0));
  vec4 a0   = b0.xzyw + s0.xzyw * sh.xxyy;
  vec4 a1   = b1.xzyw + s1.xzyw * sh.zzww;
  vec3 p0   = vec3(a0.xy, h.x);
  vec3 p1   = vec3(a0.zw, h.y);
  vec3 p2   = vec3(a1.xy, h.z);
  vec3 p3   = vec3(a1.zw, h.w);

  // Normalise gradients
  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;

  // Mix final noise value
  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}
/* end Ashima snoise */

uniform float uTime;
uniform float uIntensity;
uniform float uPulse;

varying vec3  vNormal;
varying float vDisp;
varying vec3  vPosition;

void main() {
  vec3 pos = position;

  // Two octaves of noise for a more organic surface.
  // Slow time multiplier keeps motion ambient, not twitchy.
  float n1 = snoise(pos * 1.6  + uTime * 0.25);
  float n2 = snoise(pos * 3.2  + uTime * 0.15) * 0.4;
  float n  = n1 + n2;

  // Breathing: a gentle sinusoidal envelope (~4 s period) layered on top.
  float breath = sin(uTime * 1.5707963) * 0.04;  // π/2 ≈ 1.5708, so period ≈ 4 s

  float disp = (n + breath) * (0.08 + uIntensity * 0.28 + uPulse * 0.18);
  pos += normal * disp;

  vNormal   = normal;
  vDisp     = disp;
  vPosition = pos;

  gl_Position = projectionMatrix * modelViewMatrix * vec4(pos, 1.0);
}
`;

const FRAGMENT_SHADER = /* glsl */`
uniform float uTime;
uniform float uHue;
uniform float uHueShift;
uniform float uIntensity;
uniform float uPulse;

varying vec3  vNormal;
varying float vDisp;
varying vec3  vPosition;

// Standard HSL → RGB (continuous, no conditionals needed in GLSL with fract/abs)
vec3 hsl2rgb(vec3 c) {
  vec3 rgb = clamp(abs(mod(c.x * 6.0 + vec3(0.0, 4.0, 2.0), 6.0) - 3.0) - 1.0, 0.0, 1.0);
  return c.z + c.y * (rgb - 0.5) * (1.0 - abs(2.0 * c.z - 1.0));
}

void main() {
  // Hue: base + shift animated by time (driven by thinking state) + displacement tint.
  float h = uHue
    + uHueShift * sin(uTime * 0.40)
    + vDisp * 0.08;

  // Saturation is moderate; lightness slightly above mid so it glows.
  float s = 0.55 + uIntensity * 0.15;
  float l = 0.55 + vDisp * 0.10;
  vec3 col = hsl2rgb(vec3(fract(h), clamp(s, 0.0, 1.0), clamp(l, 0.0, 1.0)));

  // Fresnel rim — warm lavender/peach glow at silhouette edges.
  vec3 viewDir = vec3(0.0, 0.0, 1.0);  // camera is along +Z
  float fres = pow(1.0 - max(dot(normalize(vNormal), viewDir), 0.0), 2.0);
  vec3 rimColor = vec3(0.55, 0.42, 0.60);  // lavender
  col += rimColor * fres * (0.30 + uPulse * 0.55);

  // Alpha: base transparency + intensity + pulse + fresnel glow.
  float a = 0.32 + uIntensity * 0.48 + uPulse * 0.35 + fres * 0.20;

  gl_FragColor = vec4(col, clamp(a, 0.0, 1.0));
}
`;
