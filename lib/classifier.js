// Stage 1 classifier: KNN over single-frame hand landmarks.
// Swap point: in Stage 2 replace `predict` with the trained sequence model.
const KEY = "signcaption.samples.v1";

// 2 hands x 21 landmarks x (x,y,z) = 126 numbers. Missing hand = zeros.
export function toFeatures(result) {
  const out = new Array(126).fill(0);
  const hands = result.landmarks || [];
  const handed = result.handedness || [];
  hands.slice(0, 2).forEach((lm, i) => {
    // Keep left/right in a stable slot
    const label = handed[i]?.[0]?.categoryName === "Left" ? 0 : 1;
    const wrist = lm[0];
    const mid = lm[9];
    const scale =
      Math.hypot(mid.x - wrist.x, mid.y - wrist.y, mid.z - wrist.z) || 1;
    lm.forEach((p, j) => {
      const k = label * 63 + j * 3;
      out[k] = (p.x - wrist.x) / scale;
      out[k + 1] = (p.y - wrist.y) / scale;
      out[k + 2] = (p.z - wrist.z) / scale;
    });
  });
  return out;
}

export function loadSamples() {
  try {
    return JSON.parse(localStorage.getItem(KEY) || "[]");
  } catch {
    return [];
  }
}

export function saveSamples(samples) {
  try {
    localStorage.setItem(KEY, JSON.stringify(samples));
  } catch {}
}

function dist(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s);
}

export function predict(samples, features, k = 5) {
  if (samples.length < k) return null;
  const nearest = samples
    .map((s) => ({ label: s.label, d: dist(s.f, features) }))
    .sort((a, b) => a.d - b.d)
    .slice(0, k);
  const votes = {};
  let total = 0;
  for (const n of nearest) {
    const w = 1 / (n.d + 0.001);
    votes[n.label] = (votes[n.label] || 0) + w;
    total += w;
  }
  const [label, score] = Object.entries(votes).sort((a, b) => b[1] - a[1])[0];
  return { label, confidence: score / total, nearest: nearest[0].d };
}
