// Tiny shared state with key-scoped subscribers.
export const state = {
  sessionId: null,
};

const listeners = new Map();

export function on(key, fn) {
  if (!listeners.has(key)) listeners.set(key, new Set());
  listeners.get(key).add(fn);
  return () => listeners.get(key)?.delete(fn);
}

export function set(key, value) {
  if (state[key] === value) return;
  state[key] = value;
  for (const fn of listeners.get(key) ?? []) fn(value);
}
