// Pure math for Topic 11 (kernel density estimation, §16.3).
// Loaded only by kde.html; kde-view.js calls it.
window.KDEMath = (function () {
  const SQRT2PI = Math.sqrt(2 * Math.PI);
  const TRICUBE_C = 70 / 81;
  const MAD_TO_SD = 1.4826;       // robust spread: σ̂ ≈ 1.4826 · median |x − median|
  const LOG_FLOOR = -50;          // keeps a leave-one-out score finite when a point gets ~0 density

  function rng(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const gauss = rand => Math.sqrt(-2 * Math.log(rand() || 1e-12)) * Math.cos(2 * Math.PI * rand());
  const uniform = rand => 2 * rand() - 1;

  // ---------- §16.3.1: density kernels (area 1, symmetric, ≥ 0) ----------

  const SHAPES = {
    gaussian: u => Math.exp(-u * u / 2) / SQRT2PI,
    boxcar: u => (Math.abs(u) <= 1 ? 0.5 : 0),
    epanechnikov: u => (Math.abs(u) <= 1 ? 0.75 * (1 - u * u) : 0),
    tricube: u => { const a = Math.abs(u); return a <= 1 ? TRICUBE_C * (1 - a ** 3) ** 3 : 0; }
  };

  // one random draw from the unit-width kernel
  const DRAWS = {
    gaussian: gauss,
    boxcar: uniform,
    // the median of three uniforms on [−1, 1] has exactly the Epanechnikov density
    epanechnikov: rand => {
      const u1 = uniform(rand), u2 = uniform(rand), u3 = uniform(rand);
      return Math.abs(u3) >= Math.abs(u2) && Math.abs(u3) >= Math.abs(u1) ? u2 : u3;
    },
    // rejection sampling: propose uniformly, keep with probability K(u) / K(0)
    tricube: rand => {
      for (;;) {
        const u = uniform(rand);
        if (rand() * TRICUBE_C <= SHAPES.tricube(u)) return u;
      }
    }
  };

  // K_h(x) = K(x / h) / h
  const kernel = (kind, x, h) => SHAPES[kind](x / h) / h;

  // midpoint rule (never samples exactly on the boxcar's jumps), wide enough for h up to 2.5
  function area(kind, h, lo = -10, hi = 10, steps = 4000) {
    const dx = (hi - lo) / steps;
    let sum = 0;
    for (let i = 0; i < steps; i++) sum += kernel(kind, lo + (i + 0.5) * dx, h);
    return sum * dx;
  }

  // ---------- §16.3.2: Parzen window estimate and sampling from it ----------

  function kde(xs, x, h, kind = 'gaussian') {
    if (!xs.length) return 0;
    return xs.reduce((a, xn) => a + kernel(kind, x - xn, h), 0) / xs.length;
  }

  // p̂ is a mixture of N equal bumps: pick a bump, then draw from it
  function sample(xs, h, kind, rand) {
    const i = Math.min(xs.length - 1, Math.floor(rand() * xs.length));
    const nudge = h * DRAWS[kind](rand);
    return { i, nudge, x: xs[i] + nudge };
  }

  // ---------- §16.3.3: choosing the bandwidth ----------

  // two humps of different widths, the "true" density students try to recover
  const TRUTH = [{ w: 0.6, m: -1.1, s: 0.5 }, { w: 0.4, m: 1.6, s: 0.9 }];
  const normal = (x, m, s) => Math.exp(-((x - m) ** 2) / (2 * s * s)) / (s * SQRT2PI);

  const truePdf = x => TRUTH.reduce((a, c) => a + c.w * normal(x, c.m, c.s), 0);

  function sampleTruth(n, rand) {
    return Array.from({ length: n }, () => {
      const c = rand() < TRUTH[0].w ? TRUTH[0] : TRUTH[1];
      return c.m + c.s * gauss(rand);
    });
  }

  function median(v) {
    const s = [...v].sort((a, b) => a - b), m = s.length >> 1;
    return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
  }

  // rule of thumb: h = σ̂ (4 / 3N)^(1/5), with σ̂ from the median absolute deviation
  function silverman(xs) {
    const med = median(xs), sigma = MAD_TO_SD * median(xs.map(x => Math.abs(x - med)));
    return sigma * Math.pow(4 / (3 * xs.length), 0.2);
  }

  // average log p̂(x_i) where p̂ is built from every point except x_i
  function looScore(xs, h, kind = 'gaussian') {
    const n = xs.length;
    let total = 0;
    for (let i = 0; i < n; i++) {
      let d = 0;
      for (let j = 0; j < n; j++) if (j !== i) d += kernel(kind, xs[i] - xs[j], h);
      total += Math.max(LOG_FLOOR, Math.log(d / (n - 1)));
    }
    return total / n;
  }

  // ---------- §16.3.4: from KDE to KNN ----------

  const dist = (u, v) => Math.hypot(u[0] - v[0], u[1] - v[1]);

  // two overlapping clouds plus a far-off pocket of class 1
  function blobs(rand) {
    const X = [], y = [];
    const add = (n, cx, cy, s, c) => {
      for (let i = 0; i < n; i++) { X.push([cx + s * gauss(rand), cy + s * gauss(rand)]); y.push(c); }
    };
    add(28, -0.8, 0.4, 0.6, 0);
    add(24, 0.8, -0.3, 0.55, 1);
    add(5, -1.4, -1.6, 0.25, 1);
    return { X, y };
  }

  function counts(y, idx, k) {
    const c = Array(k).fill(0);
    idx.forEach(i => { c[y[i]] += 1; });
    return c;
  }

  // balloon: grow a circle until it holds K points
  function balloon(X, y, q, K, classes = 2) {
    const order = X.map((p, i) => [i, dist(p, q)]).sort((a, b) => a[1] - b[1]).slice(0, K);
    const r = order[order.length - 1][1], inside = order.map(o => o[0]), c = counts(y, inside, classes);
    return { r, inside, counts: c, total: K, probs: c.map(v => v / K), density: K / (X.length * Math.PI * r * r) };
  }

  // fixed window: a circle of radius r, however many points land in it
  function window2d(X, y, q, r, classes = 2) {
    const inside = X.map((p, i) => (dist(p, q) <= r ? i : -1)).filter(i => i >= 0);
    const c = counts(y, inside, classes), total = inside.length;
    return { r, inside, counts: c, total, probs: total ? c.map(v => v / total) : null, density: total / (X.length * Math.PI * r * r) };
  }

  // ---------- §16.3.5: kernel regression ----------

  const trend = x => 0.4 * x + Math.sin(1.6 * x);

  function regData(rand, n = 34, noise = 0.28) {
    const xs = Array.from({ length: n }, (_, i) => (6 * (i + rand())) / n);   // one x per slot, so both edges have data
    return { xs, ys: xs.map(x => trend(x) + noise * gauss(rand)) };
  }

  // Gaussian weights that add to 1; shifting by the nearest distance stops underflow at tiny h
  function weights(xs, x, h) {
    const d2 = xs.map(xn => ((x - xn) / h) ** 2), lo = Math.min(...d2);
    const raw = d2.map(v => Math.exp(-(v - lo) / 2)), sum = raw.reduce((a, b) => a + b, 0);
    return raw.map(v => v / sum);
  }

  // Nadaraya–Watson: weighted average of y, and the spread of the weighted y's plus the y-bump width
  function nadarayaWatson(xs, ys, x, h, sy) {
    const w = weights(xs, x, h);
    const mean = w.reduce((a, wn, n) => a + wn * ys[n], 0);
    const second = w.reduce((a, wn, n) => a + wn * ys[n] * ys[n], 0);
    return { w, mean, sd: Math.sqrt(Math.max(0, sy * sy + second - mean * mean)) };
  }

  // locally weighted regression: weighted least-squares line around x, read off at x
  function localLine(xs, ys, x, h) {
    const w = weights(xs, x, h);
    let sw = 0, sx = 0, sy = 0, sxx = 0, sxy = 0;
    w.forEach((wn, n) => {
      const u = xs[n] - x;
      sw += wn; sx += wn * u; sy += wn * ys[n]; sxx += wn * u * u; sxy += wn * u * ys[n];
    });
    const den = sw * sxx - sx * sx;
    const slope = Math.abs(den) < 1e-12 ? 0 : (sw * sxy - sx * sy) / den;
    return { w, mean: (sy - slope * sx) / sw, slope };
  }

  return {
    rng, kernel, area, kde, sample, truePdf, sampleTruth, silverman, looScore,
    blobs, balloon, window2d, trend, regData, nadarayaWatson, localLine, SHAPES
  };
})();
