// Pure math for Topic 13 (the kernel trick, §17.3.4).
// Loaded only by kernel-trick.html; kt-view.js calls it.
window.KTMath = (function () {
  // simplified SMO (Platt 1998, CS229 version): small data, so a few passes are plenty
  const SMO_TOL = 1e-3, SMO_EPS = 1e-5, SMO_PASSES = 8, SMO_MAX_ITER = 300, SV_MIN = 1e-6;
  const DATA_LIM = 2.9;

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
  const dot = (u, v) => u.reduce((a, ui, i) => a + ui * v[i], 0);
  const sqDist = (u, v) => u.reduce((a, ui, i) => a + (ui - v[i]) ** 2, 0);
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

  // ---------- kernels: K(a, b) = φ(a)·φ(b) without ever building φ ----------

  // p = { type: 'linear' | 'poly' | 'rbf', gamma }
  function kernel(p, a, b) {
    if (p.type === 'poly') return dot(a, b) ** 2;
    if (p.type === 'rbf') return Math.exp(-p.gamma * sqDist(a, b));
    return dot(a, b);
  }

  // the explicit features behind the degree-2 kernel: (a·b)² = φ(a)·φ(b)
  const lift2 = ([x1, x2]) => [x1 * x1, Math.SQRT2 * x1 * x2, x2 * x2];

  // number of degree-d monomials in D variables = C(D + d − 1, d): the length of φ for (a·b)^d
  function featureCount(D, d) {
    let c = 1;
    for (let k = 1; k <= d; k++) c = (c * (D + k - 1)) / k;
    return Math.round(c);
  }

  // ---------- §17.3.4: the dual SVM only ever needs K(xᵢ, xⱼ) ----------

  // pts = [[x1, x2, label ±1]]. Returns α and w₀ (as b) for f(x) = Σₙ αₙ yₙ K(xₙ, x) + w₀
  function trainSvm(p, pts, C, seed) {
    const N = pts.length, X = pts.map(q => [q[0], q[1]]), y = pts.map(q => q[2]);
    const K = X.map(a => X.map(b => kernel(p, a, b)));
    const alpha = new Array(N).fill(0), rand = rng(seed);
    let b = 0, passes = 0, iter = 0;
    const f = i => alpha.reduce((s, a, k) => (a ? s + a * y[k] * K[k][i] : s), b);

    while (N > 1 && passes < SMO_PASSES && iter < SMO_MAX_ITER) {
      let changed = 0;
      for (let i = 0; i < N; i++) {
        const Ei = f(i) - y[i];
        if (!((y[i] * Ei < -SMO_TOL && alpha[i] < C) || (y[i] * Ei > SMO_TOL && alpha[i] > 0))) continue;
        let j = Math.floor(rand() * (N - 1));
        if (j >= i) j++;
        const Ej = f(j) - y[j], ai = alpha[i], aj = alpha[j];
        const L = y[i] !== y[j] ? Math.max(0, aj - ai) : Math.max(0, ai + aj - C);
        const H = y[i] !== y[j] ? Math.min(C, C + aj - ai) : Math.min(C, ai + aj);
        const eta = 2 * K[i][j] - K[i][i] - K[j][j];
        if (L === H || eta >= 0) continue;
        const ajNew = clamp(aj - (y[j] * (Ei - Ej)) / eta, L, H);
        if (Math.abs(ajNew - aj) < SMO_EPS) continue;
        const aiNew = ai + y[i] * y[j] * (aj - ajNew);
        const b1 = b - Ei - y[i] * (aiNew - ai) * K[i][i] - y[j] * (ajNew - aj) * K[i][j];
        const b2 = b - Ej - y[i] * (aiNew - ai) * K[i][j] - y[j] * (ajNew - aj) * K[j][j];
        b = aiNew > 0 && aiNew < C ? b1 : ajNew > 0 && ajNew < C ? b2 : (b1 + b2) / 2;
        alpha[i] = aiNew;
        alpha[j] = ajNew;
        changed++;
      }
      passes = changed ? 0 : passes + 1;
      iter++;
    }
    const sv = alpha.map((a, n) => n).filter(n => alpha[n] > SV_MIN);
    return { alpha, b, sv, X, y };
  }

  // only the support vectors (αₙ > 0) contribute
  function decision(p, model, x) {
    return model.sv.reduce((s, n) => s + model.alpha[n] * model.y[n] * kernel(p, model.X[n], x), model.b);
  }

  // ---------- example data on [−3, 3]² ----------

  function dataset(name, seed) {
    const rand = rng(seed), pts = [];
    const ring = (count, r0, dr, label) => {
      for (let n = 0; n < count; n++) {
        const r = r0 + dr * Math.sqrt(rand()), t = 2 * Math.PI * rand();
        pts.push([r * Math.cos(t), r * Math.sin(t), label]);
      }
    };
    const blob = (count, mx, my, sd, label) => {
      for (let n = 0; n < count; n++) pts.push([mx + sd * gauss(rand), my + sd * gauss(rand), label]);
    };
    if (name === 'rings') { ring(14, 0, 1.05, 1); ring(20, 1.9, 0.8, -1); }
    else if (name === 'xor') [[1, 1], [-1, -1], [1, -1], [-1, 1]].forEach(([sx, sy]) => blob(8, 1.4 * sx, 1.4 * sy, 0.45, sx * sy));
    else { blob(14, 1.2, 0.9, 0.7, 1); blob(14, -1.2, -0.8, 0.7, -1); }
    return pts.map(([a, b, l]) => [clamp(a, -DATA_LIM, DATA_LIM), clamp(b, -DATA_LIM, DATA_LIM), l]);
  }

  return { kernel, lift2, featureCount, trainSvm, decision, dataset, dot };
})();
