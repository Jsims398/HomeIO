// Pure math for Topic 19 (PCA, §20.1.1, 20.1.3, 20.1.4). Loaded only by pca.html, after kt-math.js
// (it reuses KTMath.rng). The pca-*-view.js files call it.
//
// Conventions follow Murphy: rows of X are examples, μ is the mean row, Σ = (1/N) X̃ᵀX̃ with X̃ = X − μ,
// and the principal directions w_j are the eigenvectors of Σ sorted by eigenvalue λ_j, biggest first.
window.PCAMath = (function () {
  const T = () => window.KTMath;
  const JACOBI_SWEEPS = 60, TINY = 1e-12, VAR_FLOOR = 1e-9;

  // ---- small dense linear algebra (matrices are arrays of rows) ----
  const transpose = A => A[0].map((_, j) => A.map(r => r[j]));
  const matmul = (A, B) => A.map(r => B[0].map((_, j) => r.reduce((s, v, k) => s + v * B[k][j], 0)));
  const dot = (a, b) => a.reduce((s, v, i) => s + v * b[i], 0);
  const scale = (A, k) => A.map(r => r.map(v => v * k));
  const identity = n => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
  const sum = arr => arr.reduce((s, v) => s + v, 0);

  // flip each eigenvector so its largest-magnitude entry is positive: same direction, stable display
  function canonical(v) {
    const big = v.reduce((b, x) => (Math.abs(x) > Math.abs(b) + TINY ? x : b), 0);
    return big < 0 ? v.map(x => -x) : v;
  }

  // one Jacobi rotation J that zeroes A[p][q]; returns the new A = JᵀAJ and V = VJ
  function rotate(A, V, p, q) {
    const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
    const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
    const c = 1 / Math.sqrt(t * t + 1), s = t * c;
    const J = identity(A.length).map((r, i) => r.map((v, j) => {
      if (i === p && j === p) return c;
      if (i === q && j === q) return c;
      if (i === p && j === q) return s;
      if (i === q && j === p) return -s;
      return v;
    }));
    return { A: matmul(matmul(transpose(J), A), J), V: matmul(V, J) };
  }

  // eigen-decomposition of a symmetric matrix by cyclic Jacobi; values sorted biggest first
  function eigSym(S) {
    const n = S.length;
    let A = S.map(r => [...r]), V = identity(n);
    for (let sweep = 0; sweep < JACOBI_SWEEPS; sweep++) {
      let off = 0;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
      if (off < TINY * TINY) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) > TINY) ({ A, V } = rotate(A, V, p, q));
      }
    }
    const cols = transpose(V);
    return A.map((r, i) => ({ value: Math.max(0, r[i]), vector: canonical(cols[i]) }))
      .sort((a, b) => b.value - a.value);
  }

  // ---- PCA itself ----
  const meanRow = X => X[0].map((_, j) => sum(X.map(r => r[j])) / X.length);
  const center = (X, mu) => X.map(r => r.map((v, j) => v - mu[j]));
  const covariance = Xc => scale(matmul(transpose(Xc), Xc), 1 / Xc.length);

  function pca(X) {
    const mu = meanRow(X), Xc = center(X, mu), S = covariance(Xc);
    const eig = eigSym(S);
    return { mu, Xc, S, values: eig.map(e => e.value), W: eig.map(e => e.vector) };
  }

  // z = Wᵀ(x − μ) using the first L directions, and x̂ = μ + Σ_j z_j w_j
  const encode = (x, mu, W, L) => W.slice(0, L).map(w => dot(w, x.map((v, j) => v - mu[j])));
  const decode = (z, mu, W) => mu.map((m, j) => m + sum(z.map((zj, k) => zj * W[k][j])));
  const sqDist = (a, b) => sum(a.map((v, i) => (v - b[i]) ** 2));

  // mean squared reconstruction error of the rows of X using the first L directions of a fitted model
  const reconError = (X, model, L) =>
    sum(X.map(x => sqDist(x, decode(encode(x, model.mu, model.W, L), model.mu, model.W)))) / X.length;

  // ---- §20.1.1: the hand-sized example ----
  const TOY = [[1, 1.5], [2, 3], [3, 2.5], [5, 5.5], [6, 5], [7, 6.5]];

  // variance of the projections onto a unit direction u, and the mean squared distance to that line
  function lineStats(model, angle) {
    const u = [Math.cos(angle), Math.sin(angle)];
    const z = model.Xc.map(r => dot(u, r));
    const variance = sum(z.map(v => v * v)) / z.length;
    const error = sum(model.Xc.map((r, n) => sqDist(r, u.map(ui => ui * z[n])))) / z.length;
    return { u, z, variance, error };
  }

  // ---- §20.1.3: scale, the N < D trick, SVD ----
  const stdRow = (Xc) => Xc[0].map((_, j) => Math.sqrt(sum(Xc.map(r => r[j] * r[j])) / Xc.length));
  const standardize = X => {
    const mu = meanRow(X), Xc = center(X, mu), sd = stdRow(Xc);
    return { sd, Z: Xc.map(r => r.map((v, j) => v / sd[j])) };
  };
  const rescaleColumn = (X, j, k) => X.map(r => r.map((v, i) => (i === j ? v * k : v)));

  // N = 3 examples with D = 5 features: more features than examples
  const WIDE = [[2, 0, 1, 3, 1], [4, 1, 0, 1, 2], [0, 2, 2, 2, 3]];

  // eigenvectors of the N×N Gram matrix K = X̃X̃ᵀ, mapped back to D dimensions by v = X̃ᵀu / √λ
  function gramTrick(X) {
    const mu = meanRow(X), Xc = center(X, mu);
    const big = matmul(transpose(Xc), Xc), K = matmul(Xc, transpose(Xc));
    const eigBig = eigSym(big), eigK = eigSym(K);
    const mapped = eigK.filter(e => e.value > VAR_FLOOR).map(e => {
      const v = transpose(Xc).map(col => dot(col, e.vector) / Math.sqrt(e.value));
      return { value: e.value, u: e.vector, v: canonical(v) };
    });
    return { Xc, big, K, eigBig, eigK, mapped };
  }

  // thin SVD X̃ = U S Vᵀ, read off the eigen-decomposition of X̃ᵀX̃: s_j = √(N λ_j), u_j = X̃ v_j / s_j
  function svd(Xc) {
    const eig = eigSym(matmul(transpose(Xc), Xc));
    const s = eig.map(e => Math.sqrt(e.value));
    const V = eig.map(e => e.vector);
    const U = Xc.map(r => V.map((v, j) => (s[j] > TINY ? dot(r, v) / s[j] : 0)));
    return { U, s, V };
  }

  // ---- §20.1.4: how many dimensions? ----
  // three latent directions of similar size; a single dominant one would fool the profile likelihood
  const SYN = { D: 8, L: 3, N: 50, SCALES: [2, 1.7, 1.4], NOISE: 0.35, SEED: 3 };
  const gauss = rand => Math.sqrt(-2 * Math.log(rand() || TINY)) * Math.cos(2 * Math.PI * rand());

  // Gram-Schmidt: random directions made orthonormal, so the true eigenvalues are ≈ scale² + noise²
  function orthonormal(cols) {
    return cols.reduce((done, c) => {
      const r = done.reduce((v, q) => { const k = dot(v, q); return v.map((x, i) => x - k * q[i]); }, c);
      const n = Math.sqrt(dot(r, r));
      return [...done, r.map(x => x / n)];
    }, []);
  }

  // x = W z + noise with 3 real latent directions in 8 dimensions; train and test from the same W
  function synthetic(seed) {
    const rand = T().rng(seed);
    const dirs = orthonormal(SYN.SCALES.map(() => Array.from({ length: SYN.D }, () => gauss(rand))));
    const Wt = transpose(dirs);
    const draw = () => Array.from({ length: SYN.N }, () => {
      const z = SYN.SCALES.map(s => s * gauss(rand));
      return Wt.map(row => dot(row, z) + SYN.NOISE * gauss(rand));
    });
    return { train: draw(), test: draw() };
  }

  const logNormal = (x, m, v) => -0.5 * Math.log(2 * Math.PI * v) - ((x - m) ** 2) / (2 * v);

  // Zhu & Ghodsi profile likelihood: λ_1..λ_L ~ N(μ₁, σ²), λ_{L+1}..λ_D ~ N(μ₂, σ²), shared σ²
  function profileLogLik(values, L) {
    const head = values.slice(0, L), tail = values.slice(L);
    const m1 = sum(head) / head.length, m2 = sum(tail) / tail.length;
    const ss = sum(head.map(v => (v - m1) ** 2)) + sum(tail.map(v => (v - m2) ** 2));
    const v = Math.max(ss / values.length, VAR_FLOOR);
    return sum(head.map(x => logNormal(x, m1, v))) + sum(tail.map(x => logNormal(x, m2, v)));
  }

  // one row per L = 1..D: variance explained, train and test reconstruction error, profile log-likelihood
  function chooseL(train, test) {
    const model = pca(train), total = sum(model.values);
    const rows = model.values.map((_, i) => {
      const L = i + 1;
      return {
        L,
        frac: sum(model.values.slice(0, L)) / total,
        trainErr: reconError(train, model, L),
        testErr: reconError(test, model, L),
        profile: L < model.values.length ? profileLogLik(model.values, L) : null
      };
    });
    return { values: model.values, rows };
  }

  return {
    transpose, matmul, dot, sum, eigSym, meanRow, center, covariance, pca, encode, decode, reconError,
    TOY, lineStats, standardize, rescaleColumn, WIDE, gramTrick, svd, SYN, synthetic, profileLogLik, chooseL
  };
})();
