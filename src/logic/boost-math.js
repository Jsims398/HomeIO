// Pure math for Topic 16 (boosting: forward stagewise additive modelling, least squares boosting,
// AdaBoost, LogitBoost, §18.5.1 – 18.5.4). Loaded only by boosting.html, after kt-math.js
// (it reuses KTMath.rng). boost-view.js calls it.
window.BoostMath = (function () {
  const T = () => window.KTMath;
  const LIM = 10, EPS = 1e-12, ERR_FLOOR = 1e-4, Z_MAX = 4, W_FLOOR = 1e-6;
  const REG_N = 24, CLS_PER = 12, CLS_SPREAD = 1.5;
  const CLS_CENTERS = [[3.5, 3.5], [6.5, 6.5]];
  const OUTLIER = [1.6, 2.0];

  const gauss = rand => Math.sqrt(-2 * Math.log(rand() || 1e-12)) * Math.cos(2 * Math.PI * rand());
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const round1 = v => Math.round(v * 10) / 10;
  const sum = arr => arr.reduce((s, v) => s + v, 0);
  const mean = arr => (arr.length ? sum(arr) / arr.length : 0);

  // 1-D regression data: y = 3.2 + 2 sin(0.8x) + noise, sorted by x. Each point is { x: [x], y }
  function regData(seed) {
    const rand = T().rng(seed);
    return Array.from({ length: REG_N }, (_, i) => {
      const x = round1(clamp(0.2 + (9.6 * (i + 0.5)) / REG_N + 0.3 * gauss(rand), 0.1, LIM - 0.1));
      const y = round1(clamp(3.2 + 2 * Math.sin(0.8 * x) + 0.45 * gauss(rand), 0.2, 6.8));
      return { x: [x], y };
    }).sort((a, b) => a.x[0] - b.x[0]);
  }

  // 2-D binary data, labels ỹ ∈ {−1, +1}: two overlapping blobs, plus an optional mislabelled outlier
  function clsData(seed, withOutlier) {
    const rand = T().rng(seed), pts = [];
    CLS_CENTERS.forEach(([cx, cy], k) => {
      for (let n = 0; n < CLS_PER; n++) {
        const x = [round1(clamp(cx + CLS_SPREAD * gauss(rand), 0.3, LIM - 0.3)), round1(clamp(cy + CLS_SPREAD * gauss(rand), 0.3, LIM - 0.3))];
        pts.push({ x, y: k ? 1 : -1, outlier: false });
      }
    });
    return withOutlier ? [...pts, { x: OUTLIER, y: 1, outlier: true }] : pts;
  }

  // candidate thresholds: midpoints between consecutive sorted unique values
  function thresholds(vals) {
    const u = [...new Set(vals)].sort((a, b) => a - b);
    return u.slice(1).map((v, i) => (u[i] + v) / 2);
  }

  // a stump is { j, t, a, b }: F(x) = a if x_j ≤ t, else b
  const evalStump = (s, x) => (x[s.j] <= s.t ? s.a : s.b);

  // f(x) = f0 + Σ coef · F(x)  for terms [{ stump, coef }]
  const ensemble = (terms, x, f0 = 0) => terms.reduce((f, tm) => f + tm.coef * evalStump(tm.stump, x), f0);

  // weighted least-squares stump: argmin Σ w_i (z_i − F(x_i))². Each leaf value is the weighted mean of z in that leaf
  function lsStump(X, z, w) {
    let best = null;
    X[0].forEach((_, j) => thresholds(X.map(x => x[j])).forEach(t => {
      const side = X.map(x => x[j] <= t);
      const wl = sum(w.filter((_, i) => side[i])), wr = sum(w.filter((_, i) => !side[i]));
      const a = wl > EPS ? sum(z.map((v, i) => (side[i] ? w[i] * v : 0))) / wl : 0;
      const b = wr > EPS ? sum(z.map((v, i) => (side[i] ? 0 : w[i] * v))) / wr : 0;
      const sse = sum(z.map((v, i) => w[i] * (v - (side[i] ? a : b)) ** 2));
      if (!best || sse < best.sse - 1e-9) best = { j, t, a, b, sse };
    }));
    return best;
  }

  // classification stump with outputs ±1: argmin weighted error Σ ω_i I(ỹ_i ≠ F(x_i)) / Σ ω_i
  function classStump(X, y, w) {
    const total = sum(w);
    let best = null;
    X[0].forEach((_, j) => thresholds(X.map(x => x[j])).forEach(t => [1, -1].forEach(a => {
      const s = { j, t, a, b: -a };
      const err = sum(w.filter((_, i) => evalStump(s, X[i]) !== y[i])) / total;
      if (!best || err < best.err - 1e-12) best = { ...s, err };
    })));
    return best;
  }

  // §18.5.2 least squares boosting. f0 = ȳ, then each round fits a stump to the residuals r = y − f
  function lsBoost(data, rounds, nu) {
    const X = data.map(d => d.x), y = data.map(d => d.y), ones = y.map(() => 1), f0 = mean(y);
    const steps = [];
    let f = y.map(() => f0);
    for (let m = 0; m < rounds; m++) {
      const r = y.map((v, i) => v - f[i]);
      const stump = lsStump(X, r, ones);
      const F = X.map(x => evalStump(stump, x));
      const fAfter = f.map((v, i) => v + nu * F[i]);
      steps.push({ fBefore: f, r, stump, F, fAfter, rAfter: y.map((v, i) => v - fAfter[i]) });
      f = fAfter;
    }
    return { f0, steps };
  }

  // §18.5.3 AdaBoost (exponential loss). ω starts uniform; each round:
  //   F_m = weighted-error stump, β_m = ½ log((1 − err)/err), ω_i ← ω_i exp(−β_m ỹ_i F_m(x_i)) / Z, f ← f + β_m F_m
  function adaBoost(data, rounds) {
    const X = data.map(d => d.x), y = data.map(d => d.y), N = y.length, steps = [];
    let w = y.map(() => 1 / N), f = y.map(() => 0);
    for (let m = 0; m < rounds; m++) {
      const stump = classStump(X, y, w);
      const F = X.map(x => evalStump(stump, x));
      const err = clamp(stump.err, ERR_FLOOR, 1 - ERR_FLOOR);
      const beta = 0.5 * Math.log((1 - err) / err);
      const raw = w.map((v, i) => v * Math.exp(-beta * y[i] * F[i]));
      const Z = sum(raw);
      const wNext = raw.map(v => v / Z);
      const fAfter = f.map((v, i) => v + beta * F[i]);
      steps.push({ w, stump, F, miss: F.map((v, i) => v !== y[i]), err, beta, raw, Z, wNext, fBefore: f, fAfter });
      w = wNext;
      f = fAfter;
    }
    return { steps };
  }

  // p(ỹ = +1 | x) = 1 / (1 + e^{−2f(x)})
  const prob = f => 1 / (1 + Math.exp(-2 * f));

  // §18.5.4 LogitBoost (log loss, one Newton step per round). With y* = (ỹ + 1)/2 ∈ {0, 1}:
  //   π_i = prob(f(x_i)), w_i = π_i(1 − π_i), z_i = (y*_i − π_i)/w_i (clipped to ±Z_MAX),
  //   F_m = weighted least-squares stump on z, f ← f + ½ F_m
  function logitBoost(data, rounds) {
    const X = data.map(d => d.x), ys = data.map(d => (d.y > 0 ? 1 : 0)), steps = [];
    let f = ys.map(() => 0);
    for (let m = 0; m < rounds; m++) {
      const pi = f.map(prob);
      const w = pi.map(p => Math.max(p * (1 - p), W_FLOOR));
      const zRaw = ys.map((v, i) => (v - pi[i]) / w[i]);
      const z = zRaw.map(v => clamp(v, -Z_MAX, Z_MAX));
      const stump = lsStump(X, z, w);
      const F = X.map(x => evalStump(stump, x));
      const fAfter = f.map((v, i) => v + 0.5 * F[i]);
      steps.push({ fBefore: f, pi, w, zRaw, z, stump, F, fAfter, piAfter: fAfter.map(prob) });
      f = fAfter;
    }
    return { steps };
  }

  // the rectangles over which an additive model of stumps is constant (all thresholds on each axis), with f at each
  function ensembleCells(terms, f0 = 0) {
    const breaks = j => [0, ...[...new Set(terms.filter(tm => tm.stump.j === j).map(tm => tm.stump.t))].sort((a, b) => a - b), LIM];
    const bx = breaks(0), by = breaks(1), cells = [];
    bx.slice(1).forEach((x1, i) => by.slice(1).forEach((y1, k) => {
      const x0 = bx[i], y0 = by[k];
      cells.push({ x0, x1, y0, y1, f: ensemble(terms, [(x0 + x1) / 2, (y0 + y1) / 2], f0) });
    }));
    return cells;
  }

  // classify with sign(f), counting f = 0 as +1
  const classError = (f, y) => (y.length ? y.filter((v, i) => (f[i] >= 0 ? 1 : -1) !== v).length / y.length : 0);
  const mse = (f, y) => mean(y.map((v, i) => (v - f[i]) ** 2));

  // losses as functions of the margin ỹ f(x), scaled so all three equal 1 at margin 0 (log loss in bits)
  const LOSSES = {
    zeroOne: m => (m <= 0 ? 1 : 0),
    exp: m => Math.exp(-m),
    log: m => Math.log2(1 + Math.exp(-2 * m))
  };
  // how hard each loss pulls on a point: |dℓ/d(margin)|
  const PULL = {
    exp: m => Math.exp(-m),
    log: m => 2 / (Math.LN2 * (1 + Math.exp(2 * m)))
  };

  return {
    LIM, Z_MAX, regData, clsData, thresholds, evalStump, ensemble, lsStump, classStump,
    lsBoost, adaBoost, logitBoost, prob, ensembleCells, classError, mse, mean, LOSSES, PULL
  };
})();
