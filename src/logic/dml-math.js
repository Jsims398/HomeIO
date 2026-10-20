// Pure math for Topic 10 (deep metric learning, §16.2.2 – 16.2.4).
// Loaded only by metric-learning.html; dml-view.js calls it.
window.DMLMath = (function () {
  const K = 3;                    // classes in the toy data
  const SIZES = [2, 24, 24, 2];   // MLP: input → two tanh layers → 2-d embedding
  const BATCH = 60;
  const LR = 0.03, B1 = 0.9, B2 = 0.999, EPS = 1e-8;
  const MARGIN = { triplet: 0.4, contrastive: 1.0 };
  const PROXY_SCALE = 8;
  const INPUT_SCALE = 3;          // spreads the [-1, 1] inputs across tanh's bendy range

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
  const sq = (u, v) => (u[0] - v[0]) ** 2 + (u[1] - v[1]) ** 2;
  const dist = (u, v) => Math.sqrt(sq(u, v));

  // three interleaved spiral arms: neighbours are same-class, but the classes
  // wrap around each other, so raw distance says little about class overall
  function spiral(nPer, rand, turns = 0.8, noise = 0.045) {
    const X = [], y = [];
    for (let k = 0; k < K; k++) {
      for (let i = 0; i < nPer; i++) {
        const t = i / (nPer - 1), r = 0.15 + 0.85 * t;
        const ang = (2 * Math.PI * k) / K + t * turns * 2 * Math.PI;
        X.push([r * Math.cos(ang) + noise * gauss(rand), r * Math.sin(ang) + noise * gauss(rand)]);
        y.push(k);
      }
    }
    return { X, y, byClass: Array.from({ length: K }, (_, k) => y.map((c, i) => (c === k ? i : -1)).filter(i => i >= 0)) };
  }

  // ---- network: params live in one flat vector so Adam is a single loop ----

  const LAYOUT = (() => {
    let o = 0;
    const layers = SIZES.slice(1).map((out, l) => {
      const inp = SIZES[l], L = { inp, out, w: o, b: o + inp * out };
      o += inp * out + out;
      return L;
    });
    return { layers, proxy: o, size: o + K * 2 };
  })();

  function initModel(rand) {
    const P = new Float64Array(LAYOUT.size);
    LAYOUT.layers.forEach(L => {
      const lim = Math.sqrt(6 / (L.inp + L.out));
      for (let j = 0; j < L.inp * L.out; j++) P[L.w + j] = (2 * rand() - 1) * lim;
    });
    for (let c = 0; c < K; c++) {
      const a = 2 * Math.PI * rand();
      P[LAYOUT.proxy + 2 * c] = Math.cos(a);
      P[LAYOUT.proxy + 2 * c + 1] = Math.sin(a);
    }
    return { P, m: new Float64Array(LAYOUT.size), v: new Float64Array(LAYOUT.size), t: 0 };
  }

  // e = f(x) / ||f(x)||: the embedding always sits on the unit circle
  function forward(P, raw) {
    const x = [raw[0] * INPUT_SCALE, raw[1] * INPUT_SCALE];
    const acts = [x], last = LAYOUT.layers.length - 1;
    let a = x;
    LAYOUT.layers.forEach((L, l) => {
      const z = new Array(L.out);
      for (let o = 0; o < L.out; o++) {
        let s = P[L.b + o];
        for (let i = 0; i < L.inp; i++) s += P[L.w + o * L.inp + i] * a[i];
        z[o] = l < last ? Math.tanh(s) : s;
      }
      a = z;
      acts.push(z);
    });
    const norm = Math.hypot(a[0], a[1]) || 1e-9;
    return { acts, norm, e: [a[0] / norm, a[1] / norm] };
  }

  // adds ∂loss/∂params into G, given ∂loss/∂e for one point
  function backward(P, c, dE, G) {
    const [e0, e1] = c.e, dot = e0 * dE[0] + e1 * dE[1];
    let delta = [(dE[0] - e0 * dot) / c.norm, (dE[1] - e1 * dot) / c.norm];
    for (let l = LAYOUT.layers.length - 1; l >= 0; l--) {
      const L = LAYOUT.layers[l], a = c.acts[l];
      for (let o = 0; o < L.out; o++) {
        G[L.b + o] += delta[o];
        for (let i = 0; i < L.inp; i++) G[L.w + o * L.inp + i] += delta[o] * a[i];
      }
      if (l === 0) break;
      const prev = new Array(L.inp);
      for (let i = 0; i < L.inp; i++) {
        let s = 0;
        for (let o = 0; o < L.out; o++) s += P[L.w + o * L.inp + i] * delta[o];
        prev[i] = s * (1 - a[i] * a[i]);
      }
      delta = prev;
    }
  }

  const embed = (P, X) => X.map(x => forward(P, x).e);

  function proxies(P) {
    return Array.from({ length: K }, (_, c) => {
      const v = [P[LAYOUT.proxy + 2 * c], P[LAYOUT.proxy + 2 * c + 1]], n = Math.hypot(v[0], v[1]) || 1e-9;
      return { v, n, w: [v[0] / n, v[1] / n] };
    });
  }

  // ---- batch losses. Each returns the mean loss and ∂loss/∂e per point ----

  const pick = (rand, arr) => arr[Math.floor(rand() * arr.length)];
  const otherClass = (rand, k) => (k + 1 + Math.floor(rand() * (K - 1))) % K;

  function tripletBatch(E, data, rand, m = MARGIN.triplet) {
    const dE = E.map(() => [0, 0]);
    let loss = 0;
    for (let b = 0; b < BATCH; b++) {
      const a = Math.floor(rand() * E.length), k = data.y[a];
      let p = pick(rand, data.byClass[k]);
      while (p === a) p = pick(rand, data.byClass[k]);
      const n = pick(rand, data.byClass[otherClass(rand, k)]);
      const l = sq(E[a], E[p]) - sq(E[a], E[n]) + m;
      if (l <= 0) continue;
      loss += l;
      for (let d = 0; d < 2; d++) {
        dE[a][d] += (2 * (E[n][d] - E[p][d])) / BATCH;
        dE[p][d] += (2 * (E[p][d] - E[a][d])) / BATCH;
        dE[n][d] += (2 * (E[a][d] - E[n][d])) / BATCH;
      }
    }
    return { loss: loss / BATCH, dE, dW: null };
  }

  function contrastiveBatch(E, data, rand, m = MARGIN.contrastive) {
    const dE = E.map(() => [0, 0]);
    let loss = 0;
    for (let b = 0; b < BATCH; b++) {
      const i = Math.floor(rand() * E.length), k = data.y[i], same = b % 2 === 0;
      let j = pick(rand, data.byClass[same ? k : otherClass(rand, k)]);
      while (j === i) j = pick(rand, data.byClass[k]);
      const d = dist(E[i], E[j]);
      // same: pull with d². different: push with [m − d]₊² only while inside the margin
      const coef = same ? 2 : d < m ? (-2 * (m - d)) / Math.max(d, 1e-9) : 0;
      loss += same ? d * d : Math.max(0, m - d) ** 2;
      for (let q = 0; q < 2; q++) {
        const g = (coef * (E[i][q] - E[j][q])) / BATCH;
        dE[i][q] += g;
        dE[j][q] -= g;
      }
    }
    return { loss: loss / BATCH, dE, dW: null };
  }

  // §16.2.3: plain softmax classifier on the normalized embedding. Logit for
  // class c is s · eᵀw_c with ||w_c|| = 1, so each w_c is a class "proxy"
  function proxyBatch(E, data, P, s = PROXY_SCALE) {
    const W = proxies(P), N = E.length;
    const dE = E.map(() => [0, 0]), dW = W.map(() => [0, 0]);
    let loss = 0;
    E.forEach((e, i) => {
      const pr = softmax(W.map(p => s * (e[0] * p.w[0] + e[1] * p.w[1])));
      loss -= Math.log(pr[data.y[i]]);
      pr.forEach((pc, c) => {
        const g = (s * (pc - (c === data.y[i] ? 1 : 0))) / N;
        for (let d = 0; d < 2; d++) {
          dE[i][d] += g * W[c].w[d];
          dW[c][d] += g * e[d];
        }
      });
    });
    return { loss: loss / N, dE, dW };
  }

  function lossAndGrad(model, data, kind, rand) {
    const P = model.P, caches = data.X.map(x => forward(P, x)), E = caches.map(c => c.e);
    const r = kind === 'proxy' ? proxyBatch(E, data, P)
      : kind === 'contrastive' ? contrastiveBatch(E, data, rand) : tripletBatch(E, data, rand);
    const G = new Float64Array(LAYOUT.size);
    caches.forEach((c, i) => { if (r.dE[i][0] || r.dE[i][1]) backward(P, c, r.dE[i], G); });
    if (r.dW) {
      proxies(P).forEach((p, c) => {
        const dot = p.w[0] * r.dW[c][0] + p.w[1] * r.dW[c][1];
        for (let d = 0; d < 2; d++) G[LAYOUT.proxy + 2 * c + d] += (r.dW[c][d] - p.w[d] * dot) / p.n;
      });
    }
    return { loss: r.loss, G };
  }

  // one Adam step; returns a new model rather than editing the old one
  function step(model, data, kind, rand) {
    const { loss, G } = lossAndGrad(model, data, kind, rand);
    const t = model.t + 1, P = new Float64Array(model.P), m = new Float64Array(model.m), v = new Float64Array(model.v);
    const c1 = 1 - B1 ** t, c2 = 1 - B2 ** t;
    for (let j = 0; j < P.length; j++) {
      m[j] = B1 * m[j] + (1 - B1) * G[j];
      v[j] = B2 * v[j] + (1 - B2) * G[j] * G[j];
      P[j] -= (LR * (m[j] / c1)) / (Math.sqrt(v[j] / c2) + EPS);
    }
    return { model: { P, m, v, t }, loss };
  }

  // ---- how well does distance track class? ----

  // AUC = P(random same-class pair is closer than random different-class pair),
  // plus normalized histograms of the two groups of pair distances
  function pairStats(pts, y, bins = 20, maxD = null) {
    const same = [], diff = [];
    for (let i = 0; i < pts.length; i++) {
      for (let j = i + 1; j < pts.length; j++) (y[i] === y[j] ? same : diff).push(dist(pts[i], pts[j]));
    }
    const all = same.map(d => [d, 1]).concat(diff.map(d => [d, 0])).sort((a, b) => a[0] - b[0]);
    let rankSum = 0;
    all.forEach(([, s], r) => { if (s) rankSum += r + 1; });
    // Mann–Whitney U counts (same, diff) pairs where the same-class pair is farther
    const auc = 1 - (rankSum - (same.length * (same.length + 1)) / 2) / (same.length * diff.length);
    const top = maxD || all[all.length - 1][0] || 1;
    const hist = arr => {
      const h = new Array(bins).fill(0);
      arr.forEach(d => { h[Math.min(bins - 1, Math.floor((d / top) * bins))]++; });
      return h.map(c => c / arr.length);
    };
    return { auc, hSame: hist(same), hDiff: hist(diff), maxD: top };
  }

  function nearest(pts, i, k) {
    return pts.map((p, j) => [j, sq(p, pts[i])]).filter(([j]) => j !== i)
      .sort((a, b) => a[1] - b[1]).slice(0, k).map(([j]) => j);
  }

  // ---- single-example losses for the hands-on widgets ----

  function softmax(z) {
    const mx = Math.max(...z), ex = z.map(v => Math.exp(v - mx)), tot = ex.reduce((a, v) => a + v, 0);
    return ex.map(v => v / tot);
  }

  // §16.2.4 contrastive: I(same)·d² + I(diff)·[m − d]₊². Gradient is w.r.t. x_i (x_j gets −g)
  function contrastive(xi, xj, same, m) {
    const d = dist(xi, xj);
    const loss = same ? d * d : Math.max(0, m - d) ** 2;
    const coef = same ? 2 : d < m ? (-2 * (m - d)) / Math.max(d, 1e-9) : 0;
    return { d, loss, gi: [coef * (xi[0] - xj[0]), coef * (xi[1] - xj[1])] };
  }

  // §16.2.4 triplet: [d(a,p)² − d(a,n)² + m]₊
  function triplet(a, p, n, m) {
    const dap2 = sq(a, p), dan2 = sq(a, n), raw = dap2 - dan2 + m, on = raw > 0;
    const kind = dan2 < dap2 ? 'hard' : dan2 < dap2 + m ? 'semi-hard' : 'easy';
    const g = f => (on ? f : [0, 0]);
    return {
      dap: Math.sqrt(dap2), dan: Math.sqrt(dan2), loss: Math.max(0, raw), kind,
      ga: g([2 * (n[0] - p[0]), 2 * (n[1] - p[1])]),
      gp: g([2 * (p[0] - a[0]), 2 * (p[1] - a[1])]),
      gn: g([2 * (a[0] - n[0]), 2 * (a[1] - n[1])])
    };
  }

  // §16.2.3: embedding at angle φ, unit proxies at angles θ_c. Logit = s(cos − margin for the true class)
  function cosSoftmax(phi, thetas, cls, s, margin) {
    const cos = thetas.map(t => Math.cos(phi - t));
    const logits = cos.map((c, k) => s * (c - (k === cls ? margin : 0)));
    const probs = softmax(logits);
    return { cos, logits, probs, loss: -Math.log(probs[cls]) };
  }

  // §16.2.4 N-pairs: log(1 + Σ_k exp(s·eᵀe_k − s·eᵀe⁺)), i.e. softmax cross-entropy
  // where the positive is the "right answer" and every negative competes with it
  function nPairs(anchor, pos, negs, s) {
    const sim = ang => s * Math.cos(ang - anchor);
    const sp = sim(pos), sn = negs.map(sim);
    const probs = softmax([sp, ...sn]);
    const loss = Math.log(1 + sn.reduce((a, v) => a + Math.exp(v - sp), 0));
    const single = Math.log(1 + Math.exp(sn[0] - sp));
    return { sp, sn, probs, loss, single };
  }

  return { K, MARGIN, PROXY_SCALE, LAYOUT, rng, spiral, initModel, forward, embed, proxies, step, lossAndGrad,
           pairStats, nearest, contrastive, triplet, cosSoftmax, nPairs };
})();
