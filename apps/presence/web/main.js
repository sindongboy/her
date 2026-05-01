/**
 * main.js — WebSocket client, subtitle controller, and orb state machine.
 *
 * Responsibilities:
 *   1. Connect to /ws and auto-reconnect on disconnect.
 *   2. Drive Orb state and pulse from server events.
 *   3. Show user transcript and agent response chunks as film-style subtitles.
 *   4. Update status label and connection dot.
 */

import { NeuralNetwork } from '/static/network.js';

// ── DOM refs ──────────────────────────────────────────────────────────────────
const canvas   = /** @type {HTMLCanvasElement} */ (document.getElementById('orb'));
const statusEl = /** @type {HTMLDivElement}    */ (document.getElementById('status'));
const connEl   = /** @type {HTMLDivElement}    */ (document.getElementById('conn'));
const subsEl   = /** @type {HTMLDivElement}    */ (document.getElementById('subtitles'));

// ── NeuralNetwork instance ────────────────────────────────────────────────────
const orb = new NeuralNetwork(canvas);

// ── Korean status strings ──────────────────────────────────────────────────────
// state values: idle | listening | thinking | speaking | quiet | wake | sleep
const STATUS_KO = {
  idle:      '',
  listening: '[듣는 중]',
  thinking:  '[생각 중]',
  speaking:  '[말하는 중]',
  quiet:     '[조용 모드]',
  wake:      '',
  sleep:     '[잠자는 중]',
};

// ── Subtitle state ────────────────────────────────────────────────────────────
/** @type {ReturnType<typeof setTimeout>|null} */
let _subTimer     = null;
let _subBuffer    = '';
let _lastChunkAt  = 0;

const SUB_FADE_MS     = 5000;  // auto-fade after last update
const CHUNK_GAP_MS    = 1500;  // gap that forces a new subtitle line

/**
 * Replace subtitles with `text` and restart the fade timer.
 * @param {string} text
 * @param {boolean} [isUser=false]  true → dim styling (transcript)
 */
function setSubtitle(text, isUser = false) {
  _clearSubTimer();
  _subBuffer = text;
  _renderSubs(text, isUser);
  _startSubTimer();
}

/**
 * Append `text` to the current subtitle buffer, or start fresh if the gap
 * since the last chunk is too large.
 * @param {string} text
 */
function appendChunk(text) {
  const now = performance.now();
  if (now - _lastChunkAt > CHUNK_GAP_MS || _subBuffer.length === 0) {
    _subBuffer = text;
  } else {
    _subBuffer += text;
  }
  _lastChunkAt = now;

  _clearSubTimer();
  _renderSubs(_subBuffer, false);
  _startSubTimer();
}

/** Clear subtitle text and hide the band. */
function clearSubtitle() {
  _clearSubTimer();
  _subBuffer   = '';
  _lastChunkAt = 0;
  subsEl.classList.remove('visible');
  subsEl.textContent = '';
}

// ── Subtitle helpers ──────────────────────────────────────────────────────────

function _renderSubs(text, isUser) {
  subsEl.textContent = '';
  if (isUser) {
    // Wrap in a span for the dimmer CSS class.
    const span = document.createElement('span');
    span.className   = 'transcript';
    span.textContent = text;
    subsEl.appendChild(span);
  } else {
    subsEl.textContent = text;
  }
  subsEl.classList.add('visible');
}

function _clearSubTimer() {
  if (_subTimer !== null) {
    clearTimeout(_subTimer);
    _subTimer = null;
  }
}

function _startSubTimer() {
  _subTimer = setTimeout(() => {
    subsEl.classList.remove('visible');
    _subBuffer   = '';
    _lastChunkAt = 0;
  }, SUB_FADE_MS);
}

// ── Event handler ─────────────────────────────────────────────────────────────

/**
 * Dispatch a single server event to the orb and subtitle layer.
 * @param {{ type: string, payload: Record<string, unknown>, ts: number }} ev
 */
function handle(ev) {
  switch (ev.type) {

    case 'hello': {
      // Version guard — log and move on.
      const version = ev.payload?.schema_version ?? 'unknown';
      if (version !== 1) {
        console.warn('[her] unexpected schema_version:', version);
      }
      break;
    }

    case 'state': {
      const value = /** @type {string} */ (ev.payload.value ?? 'idle');
      orb.setState(value);
      statusEl.textContent = STATUS_KO[value] ?? '';

      // Wake event gets a strong pulse and then the orb moves to listening.
      if (value === 'wake') {
        orb.pulse(1.0);
      }
      break;
    }

    case 'transcript': {
      // User speech — show in subtitles with '› ' prefix, dimmed styling.
      if (ev.payload.final) {
        setSubtitle('› ' + ev.payload.text, true);
      } else {
        // Partial (interim) transcript — show but don't restart timer aggressively.
        // Use the same appendChunk path so it feels live.
        _subBuffer = '› ' + ev.payload.text;
        _renderSubs(_subBuffer, true);
        // Don't restart timer for partial updates — final will do that.
      }
      break;
    }

    case 'response_chunk': {
      appendChunk(/** @type {string} */ (ev.payload.text ?? ''));
      orb.pulse(0.25);
      break;
    }

    case 'response_end': {
      // Let the subtitle linger then fade naturally via the existing timer.
      // Mild pulse to mark completion.
      orb.pulse(0.10);
      break;
    }

    case 'memory_recall': {
      // A bright gold spark travels between two random neurons — "I remember something".
      orb.recallSpark();
      break;
    }

    case 'error': {
      console.warn('[her] server error:', ev.payload?.message, 'in', ev.payload?.where);
      break;
    }

    default:
      // Unknown events — ignore silently in production.
      break;
  }
}

// ── WebSocket connection ──────────────────────────────────────────────────────

/** @type {WebSocket|null} */
let _ws = null;

function connect() {
  if (_ws && (_ws.readyState === WebSocket.OPEN || _ws.readyState === WebSocket.CONNECTING)) {
    return;  // already connected / connecting
  }

  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  const url   = `${proto}//${location.host}/ws`;
  _ws = new WebSocket(url);

  _ws.addEventListener('open', () => {
    connEl.classList.add('ok');
  });

  _ws.addEventListener('close', () => {
    connEl.classList.remove('ok');
    _ws = null;
    // Auto-reconnect after 2 s — avoids hammering the server.
    setTimeout(connect, 2000);
  });

  _ws.addEventListener('error', () => {
    // The close event will fire afterward and trigger reconnect.
    connEl.classList.remove('ok');
  });

  _ws.addEventListener('message', (/** @type {MessageEvent} */ e) => {
    try {
      const ev = JSON.parse(e.data);
      handle(ev);
    } catch (err) {
      console.warn('[her] bad message:', e.data, err);
    }
  });
}

// ── Bootstrap ─────────────────────────────────────────────────────────────────
connect();
