// Pure math for Topic 12 (Gaussian process regression, §17.2.1 – 17.2.2).
// Loaded only by gp.html; gp-view.js calls it.
window.GPMath = (function () {
  const JITTERS = [1e-10, 1e-8, 1e-6];    // tiny diagonal boosts (× σ_f²) that absorb round-off, never a real negative eigenvalue

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
  const normals = (n, rand) => Array.from({ length: n }, () => gauss(rand));

  // RBF kernel, p = { l, sf }: K(x, x′) = σ_f² · exp(−(x − x′)² ⁄ 2ℓ²)
  const kernel = (p, x, x2) => p.sf * p.sf * Math.exp(-((x - x2) ** 2) / (2 * p.l * p.l));

  function gram(p, xs, noise = 0) {
    return xs.map((a, i) => xs.map((b, j) => kernel(p, a, b) + (i === j ? noise * noise : 0)));
  }

  const linspace = (lo, hi, n) => Array.from({ length: n }, (_, i) => lo + ((hi - lo) * i) / (n - 1));
  const dot = (u, v) => u.reduce((a, ui, i) => a + ui * v[i], 0);

  // ---------- linear algebra ----------

  // A = L Lᵀ; null when a pivot is not positive (A is not positive definite)
  function cholesky(A) {
    const n = A.length, L = A.map(() => new Array(n).fill(0));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = A[i][j];
        for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
        if (i === j) {
          if (!(s > 0)) return null;
          L[i][i] = Math.sqrt(s);
        } else {
          L[i][j] = s / L[j][j];
        }
      }
    }
    return L;
  }

  // retry with a tiny jitter on the diagonal before giving up
  function choleskySafe(A, scale = 1) {
    for (const j of [0, ...JITTERS]) {
      const L = cholesky(j ? A.map((row, i) => row.map((v, k) => (i === k ? v + j * scale : v))) : A);
      if (L) return L;
    }
    return null;
  }

  function solveLower(L, b) {
    const x = new Array(b.length);
    for (let i = 0; i < b.length; i++) {
      let s = b[i];
      for (let k = 0; k < i; k++) s -= L[i][k] * x[k];
      x[i] = s / L[i][i];
    }
    return x;
  }

  function solveUpperT(L, b) {
    const n = b.length, x = new Array(n);
    for (let i = n - 1; i >= 0; i--) {
      let s = b[i];
      for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k];
      x[i] = s / L[i][i];
    }
    return x;
  }

  // A⁻¹ b through the Cholesky factor of A
  const cholSolve = (L, b) => solveUpperT(L, solveLower(L, b));

  // ---------- draws from a Gaussian ----------

  // mean + L z for each fixed z, so samples morph smoothly as the settings change; null if cov is not PSD
  function drawsFrom(mean, cov, zs, scale) {
    const L = choleskySafe(cov, scale);
    if (!L) return null;
    return zs.map(z => mean.map((m, i) => { let s = m; for (let k = 0; k <= i; k++) s += L[i][k] * z[k]; return s; }));
  }

  // ---------- §17.2.1 – 17.2.2: GP posterior ----------

  // posterior over f at the grid points given (xs, ys) with noise sd `noise` (0 = noise-free, up to jitter)
  //   μ* = K*ᵀ K_σ⁻¹ y          Σ* = K** − K*ᵀ K_σ⁻¹ K*          K_σ = K_XX + σ_y² I
  function posterior(p, xs, ys, noise, grid, withCov = false) {
    const sf2 = p.sf * p.sf, Kgg = withCov ? gram(p, grid) : null;
    if (!xs.length) {
      return { mean: grid.map(() => 0), sd: grid.map(() => p.sf), cov: Kgg };
    }
    const L = choleskySafe(gram(p, xs, noise), sf2);
    if (!L) return null;
    const alpha = cholSolve(L, ys);
    const kStar = grid.map(g => xs.map(xn => kernel(p, g, xn)));
    const V = kStar.map(k => solveLower(L, k));
    const mean = kStar.map(k => dot(k, alpha));
    const sd = V.map(v => Math.sqrt(Math.max(0, sf2 - dot(v, v))));
    const cov = withCov ? Kgg.map((row, i) => row.map((kij, j) => kij - dot(V[i], V[j]))) : null;
    return { mean, sd, cov };
  }

  // μ(x*) = Σ_n w_n y_n with w = K_σ⁻¹ k*: the GP mean is a weighted sum of the observed y's
  function weights(p, xs, noise, x) {
    if (!xs.length) return [];
    const L = choleskySafe(gram(p, xs, noise), p.sf * p.sf);
    return L ? cholSolve(L, xs.map(xn => kernel(p, x, xn))) : xs.map(() => 0);
  }

  // one observation y₁ at x₁ and one query f(x₂): slice the 2-D Gaussian at y₁ = a
  //   mean = k₁₂ ⁄ (k₁₁ + σ_y²) · a        var = k₂₂ − k₁₂² ⁄ (k₁₁ + σ_y²)
  function condition2(p, x1, x2, noise, a) {
    const k11 = kernel(p, x1, x1) + noise * noise, k12 = kernel(p, x1, x2), k22 = kernel(p, x2, x2);
    return { k11, k12, k22, mean: (k12 / k11) * a, sd: Math.sqrt(Math.max(0, k22 - (k12 * k12) / k11)) };
  }

  return {
    rng, normals, kernel, gram, linspace, choleskySafe,
    drawsFrom, posterior, weights, condition2
  };
})();
