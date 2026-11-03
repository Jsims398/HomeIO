// Pure math for Topic 14 (choosing C, kernel ridge regression and SVM regression, §17.3.6 – 17.3.8).
// Loaded only by kernel-machines.html, after kt-math.js (it reuses KTMath's kernel and SMO). km-view.js calls it.
window.KMMath = (function () {
  const T = () => window.KTMath;
  const FOLDS = 5, SVR_SWEEPS = 400, SVR_TOL = 1e-7, ZERO = 1e-6;
  const CLS_N = 60, CLS_LIM = 2.8, CLS_FLIP = 0.08;
  const REG_N = 24, REG_LIM = 3, REG_NOISE = 0.25;

  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const gauss = rand => Math.sqrt(-2 * Math.log(rand() || 1e-12)) * Math.cos(2 * Math.PI * rand());

  // ---------- §17.3.6: pick C by cross-validation ----------

  // the true boundary is a wave, x₂ = 0.8 sin(1.5 x₁); some labels are flipped so a perfect fit is overfitting
  const trueSide = (x1, x2) => (x2 > 0.8 * Math.sin(1.5 * x1) ? 1 : -1);

  function classData(seed) {
    const rand = T().rng(seed), pts = [];
    for (let n = 0; n < CLS_N; n++) {
      const x1 = (2 * rand() - 1) * CLS_LIM, x2 = (2 * rand() - 1) * CLS_LIM;
      pts.push([x1, x2, rand() < CLS_FLIP ? -trueSide(x1, x2) : trueSide(x1, x2)]);
    }
    return pts;
  }

  const errorRate = (p, model, pts) =>
    pts.length ? pts.filter(q => (T().decision(p, model, [q[0], q[1]]) >= 0 ? 1 : -1) !== q[2]).length / pts.length : 0;

  // shuffle once, then fold k holds every point whose shuffled position is k mod FOLDS
  function folds(N, seed) {
    const rand = T().rng(seed), order = Array.from({ length: N }, (_, i) => i);
    for (let i = N - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }
    const fold = new Array(N);
    order.forEach((n, pos) => { fold[n] = pos % FOLDS; });
    return fold;
  }

  // for each C: train error on all points, and the K-fold estimate of error on unseen points
  function cvCurve(p, pts, Cs, seed) {
    const fold = folds(pts.length, seed);
    return Cs.map(C => {
      let wrong = 0;
      for (let k = 0; k < FOLDS; k++) {
        const train = pts.filter((_, n) => fold[n] !== k), held = pts.filter((_, n) => fold[n] === k);
        wrong += errorRate(p, T().trainSvm(p, train, C, seed), held) * held.length;
      }
      return { C, train: errorRate(p, T().trainSvm(p, pts, C, seed), pts), cv: wrong / pts.length };
    });
  }

  // ---------- §17.3.7 – 17.3.8: kernel ridge vs ε-insensitive SVR, 1-D ----------

  const truth = x => Math.sin(1.4 * x);

  function regData(seed) {
    const rand = T().rng(seed);
    return Array.from({ length: REG_N }, (_, n) => {
      const x = -REG_LIM + (2 * REG_LIM * (n + 0.2 + 0.6 * rand())) / REG_N;
      return [x, truth(x) + REG_NOISE * gauss(rand)];
    });
  }

  // RBF plus a constant 1: the constant plays the role of the bias w₀, so neither solver needs a separate one
  const k1 = (gamma, a, b) => Math.exp(-gamma * (a - b) ** 2) + 1;
  const gram = (gamma, xs) => xs.map(a => xs.map(b => k1(gamma, a, b)));

  // solves A z = b for symmetric positive definite A
  function cholSolve(A, b) {
    const N = A.length, L = A.map(() => new Array(N).fill(0));
    for (let i = 0; i < N; i++) {
      for (let j = 0; j <= i; j++) {
        let s = A[i][j];
        for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
        L[i][j] = i === j ? Math.sqrt(Math.max(s, 1e-12)) : s / L[j][j];
      }
    }
    const u = new Array(N).fill(0), z = new Array(N).fill(0);
    for (let i = 0; i < N; i++) {
      let s = b[i];
      for (let k = 0; k < i; k++) s -= L[i][k] * u[k];
      u[i] = s / L[i][i];
    }
    for (let i = N - 1; i >= 0; i--) {
      let s = u[i];
      for (let k = i + 1; k < N; k++) s -= L[k][i] * z[k];
      z[i] = s / L[i][i];
    }
    return z;
  }

  // kernel ridge: α = (K + λI)⁻¹ y. Squared loss, so every point gets a weight
  function ridge(gamma, pts, lambda) {
    const xs = pts.map(q => q[0]), K = gram(gamma, xs);
    const alpha = pts.length ? cholSolve(K.map((row, i) => row.map((v, j) => v + (i === j ? lambda : 0))), pts.map(q => q[1])) : [];
    return { xs, alpha };
  }

  // SVR dual without the bias constraint (the +1 in the kernel covers it):
  //   min ½ βᵀKβ − yᵀβ + ε Σ|βₙ|,  −C ≤ βₙ ≤ C
  // Exact coordinate descent: each βₙ is a soft-threshold, so points inside the tube land on exactly 0
  function svr(gamma, pts, eps, C) {
    const xs = pts.map(q => q[0]), y = pts.map(q => q[1]), K = gram(gamma, xs), N = xs.length;
    const beta = new Array(N).fill(0), f = new Array(N).fill(0);
    for (let sweep = 0; sweep < SVR_SWEEPS; sweep++) {
      let moved = 0;
      for (let i = 0; i < N; i++) {
        const g = y[i] - (f[i] - K[i][i] * beta[i]);
        const next = clamp((Math.sign(g) * Math.max(Math.abs(g) - eps, 0)) / K[i][i], -C, C);
        const d = next - beta[i];
        if (!d) continue;
        for (let j = 0; j < N; j++) f[j] += d * K[i][j];
        beta[i] = next;
        moved = Math.max(moved, Math.abs(d));
      }
      if (moved < SVR_TOL) break;
    }
    return { xs, alpha: beta };
  }

  const predict = (gamma, model, x) => model.alpha.reduce((s, a, n) => s + a * k1(gamma, model.xs[n], x), 0);
  const active = model => model.alpha.filter(a => Math.abs(a) > ZERO).length;

  // mean squared distance from the noise-free curve: how well the fit recovers the truth
  function truthMse(gamma, model, grid) {
    return grid.reduce((s, x) => s + (predict(gamma, model, x) - truth(x)) ** 2, 0) / grid.length;
  }

  return { trueSide, classData, cvCurve, errorRate, truth, regData, ridge, svr, predict, active, truthMse, ZERO, REG_LIM };
})();
