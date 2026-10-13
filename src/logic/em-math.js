// Pure math for Topic 8 (EM, §8.7): a 1-D Gaussian mixture.
// Loaded only by em.html; reading-log.js calls it from t8().
window.EMMath = (function () {
  const HALF_LN_2PI = 0.5 * Math.log(2 * Math.PI);
  // keeps a component from collapsing onto a single point (the MAP fix is §8.7.3.4, skipped here)
  const SD_FLOOR = 0.08;

  const lnN = (x, m, sd) => -0.5 * ((x - m) / sd) ** 2 - Math.log(sd) - HALF_LN_2PI;
  const normal = (x, m, sd) => Math.exp(lnN(x, m, sd));

  function logSumExp(a) {
    const m = Math.max(...a);
    return m + Math.log(a.reduce((s, v) => s + Math.exp(v - m), 0));
  }

  // log π_k + log N(x | μ_k, σ_k²) for each k
  const joint = (x, th) => th.pi.map((p, k) => Math.log(p) + lnN(x, th.mu[k], th.sd[k]));

  // ℓ(θ) = Σ_n log Σ_k π_k N(x_n | μ_k, σ_k²)
  const logLik = (xs, th) => xs.reduce((s, x) => s + logSumExp(joint(x, th)), 0);

  // E step: r_nk = p(z_n = k | x_n, θ)
  function eStep(xs, th) {
    return xs.map(x => {
      const j = joint(x, th), z = logSumExp(j);
      return j.map(v => Math.exp(v - z));
    });
  }

  // M step: weighted MLEs. A component that owns no points keeps its old μ, σ.
  function mStep(xs, R, prev) {
    const N = xs.length, K = prev.pi.length;
    const Nk = [], pi = [], mu = [], sd = [];
    for (let k = 0; k < K; k++) {
      const n = R.reduce((s, r) => s + r[k], 0);
      Nk.push(n);
      pi.push(Math.max(n / N, 1e-6));
      if (n < 1e-6) { mu.push(prev.mu[k]); sd.push(prev.sd[k]); continue; }
      const m = xs.reduce((s, x, i) => s + R[i][k] * x, 0) / n;
      const v = xs.reduce((s, x, i) => s + R[i][k] * (x - m) ** 2, 0) / n;
      mu.push(m);
      sd.push(Math.max(SD_FLOOR, Math.sqrt(v)));
    }
    return { Nk, theta: { pi, mu, sd } };
  }

  // EM lower bound Ł(θ; q) = Σ_n Σ_k q_nk [log π_k + log N(x_n | μ_k, σ_k²) − log q_nk]
  function bound(xs, th, Q) {
    let s = 0;
    xs.forEach((x, i) => {
      joint(x, th).forEach((lj, k) => { const q = Q[i][k]; if (q > 0) s += q * (lj - Math.log(q)); });
    });
    return s;
  }

  // small seeded generator so the bound lab shows the same data on every visit
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

  function gauss(rand) {
    const u = Math.max(rand(), 1e-12), v = rand();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  const TRUTH = {
    2: { n: 40, comps: [{ w: 0.45, m: -1.8, s: 0.7 }, { w: 0.55, m: 1.6, s: 0.9 }] },
    3: { n: 60, comps: [{ w: 0.3, m: -3, s: 0.6 }, { w: 0.3, m: 0, s: 0.7 }, { w: 0.4, m: 3, s: 0.8 }] }
  };

  // draw from the true mixture, sorted so "point n" reads left to right
  function sample(K, rand) {
    const { n, comps } = TRUTH[K], xs = [];
    for (let i = 0; i < n; i++) {
      let u = rand(), c = comps[comps.length - 1];
      for (const comp of comps) { if (u < comp.w) { c = comp; break; } u -= comp.w; }
      xs.push(c.m + c.s * gauss(rand));
    }
    return xs.sort((a, b) => a - b);
  }

  // deliberately naive start: K distinct random data points as means, pooled σ, equal weights
  function init(xs, K, rand) {
    const mean = xs.reduce((s, x) => s + x, 0) / xs.length;
    const sd = Math.sqrt(xs.reduce((s, x) => s + (x - mean) ** 2, 0) / xs.length);
    const picks = [];
    while (picks.length < K) {
      const i = Math.floor(rand() * xs.length);
      if (!picks.includes(i)) picks.push(i);
    }
    return { pi: picks.map(() => 1 / K), mu: picks.map(i => xs[i]).sort((a, b) => a - b), sd: picks.map(() => sd) };
  }

  return { lnN, normal, logLik, eStep, mStep, bound, rng, sample, init, TRUTH };
})();
