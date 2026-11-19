// View logic for Topic 19 ③ (choosing L, §20.1.4): scree plot, fraction of variance explained,
// train/test reconstruction error, and profile likelihood on 8-D data with 3 real directions.
// Loaded only by pca.html; reading-log.js merges render(c) into t19(). Math is in pca-math.js.
window.PCAChooseView = (function () {
  const M = () => window.PCAMath, W = () => window.Walkthrough;
  const KEY = 'pl';
  const CW = 420, CH = 240, PL = 46, PR = 12, PT = 14, PB = 34;
  const THRESHOLD = 0.9, Y_TICKS = 4, BAR_FILL = 0.6;
  const KIND = ['scree', 'frac', 'recon', 'profile'];
  const COL = { main: 'var(--color-accent)', test: '#4f6d8a', dim: 'var(--color-neutral-400)' };

  const f1 = v => v.toFixed(1), f2 = v => v.toFixed(2);
  const pct = v => Math.round(v * 100) + '%';
  const sum = arr => arr.reduce((s, v) => s + v, 0);

  let cached = null;
  function model() {
    if (cached) return cached;
    const syn = M().synthetic(M().SYN.SEED), res = M().chooseL(syn.train, syn.test);
    const withProfile = res.rows.filter(r => r.profile !== null);
    const best = withProfile.reduce((a, b) => (b.profile > a.profile ? b : a)).L;
    const byThreshold = res.rows.find(r => r.frac >= THRESHOLD).L;
    cached = { ...res, best, byThreshold, D: res.values.length };
    return cached;
  }

  function phases(d) {
    const v = d.values, rb = d.rows[d.best - 1];
    return [
      { title: 'Sort the eigenvalues: scree plot', formula: 'λ₁ ≥ λ₂ ≥ … ≥ λ_' + d.D,
        body: '8-D data built from ' + M().SYN.L + ' real directions plus a little noise. λ = ' + v.map(f2).join(', ') +
          '. A few tall bars, then a flat "scree" of noise. You look for the elbow where the cliff meets the rubble.' },
      { title: 'Fraction of variance explained', formula: 'F_L = (λ₁ + … + λ_L) / (λ₁ + … + λ_D)',
        body: 'How much of the total variance the first L directions keep. A common rule: the smallest L with F_L ≥ ' + pct(THRESHOLD) +
          '. Here that is L = ' + d.byThreshold + '. The threshold is your choice, so the answer is too.' },
      { title: 'Reconstruction error', formula: 'train: (1/N) Σ ‖x_n − x̂_n‖² = λ_{L+1} + … + λ_D',
        body: 'Train error is exactly the eigenvalues you left out (at L = ' + d.best + ': ' + f2(rb.trainErr) + ' = ' + f2(sum(v.slice(d.best))) +
          '). Test error on 50 new points also keeps falling as L grows: PCA is not a density model, so adding directions never hurts reconstruction. The bend at L = ' +
          d.best + ' is visible, but there is no minimum to pick.' },
      { title: 'Profile likelihood', formula: 'ℓ(L) = Σ_{j≤L} log N(λ_j | μ₁, σ²) + Σ_{j>L} log N(λ_j | μ₂, σ²)',
        body: 'Split the eigenvalues at L into "signal" and "noise", fit one Gaussian to each group with a shared σ², and score how well the split fits. ℓ(L) peaks at L = ' +
          d.best + ': an automatic way to find the elbow. Caveat: if λ₁ dwarfs everything, {λ₁} vs the rest wins and it says L = 1.' }
    ];
  }

  const plotW = CW - PL - PR;
  const xs = (i, n) => PL + ((i + 0.5) / n) * plotW;
  const ys = (v, lo, hi) => CH - PB - ((v - lo) / (hi - lo || 1)) * (CH - PT - PB);
  const points = (vals, lo, hi, n) => vals.map((v, i) => f1(xs(i, n)) + ',' + f1(ys(v, lo, hi))).join(' ');
  const dots = (vals, lo, hi, n, color, L) =>
    vals.map((v, i) => ({ cx: f1(xs(i, n)), cy: f1(ys(v, lo, hi)), r: i + 1 === L ? 6 : 3.5, color }));
  const yTicks = (lo, hi, fmt) => Array.from({ length: Y_TICKS + 1 }, (_, k) => {
    const v = lo + ((hi - lo) * k) / Y_TICKS;
    return { y: f1(ys(v, lo, hi)), ty: f1(ys(v, lo, hi) + 4), x1: PL, x2: CW - PR, tx: PL - 6, label: fmt(v) };
  });

  function scree(d, L, n) {
    const hi = d.values[0], bw = (plotW / n) * BAR_FILL;
    const bars = d.values.map((v, i) => ({
      x: f1(xs(i, n) - bw / 2), y: f1(ys(v, 0, hi)), w: f1(bw), h: f1(ys(0, 0, hi) - ys(v, 0, hi)), color: i < L ? COL.main : COL.dim
    }));
    return { bars, yTicks: yTicks(0, hi, f1), yTitle: 'eigenvalue λ_j',
      legend: [{ color: COL.main, label: 'kept (j ≤ L)' }, { color: COL.dim, label: 'dropped' }] };
  }

  function frac(d, L, n) {
    const vals = d.rows.map(r => r.frac);
    return { lines: [{ points: points(vals, 0, 1, n), color: COL.main }], dots: dots(vals, 0, 1, n, COL.main, L),
      ref: [{ x1: PL, x2: CW - PR, y: f1(ys(THRESHOLD, 0, 1)) }], yTicks: yTicks(0, 1, pct), yTitle: 'F_L',
      legend: [{ color: COL.main, label: 'variance explained' }, { color: COL.dim, label: pct(THRESHOLD) + ' line' }] };
  }

  function recon(d, L, n) {
    const tr = d.rows.map(r => r.trainErr), te = d.rows.map(r => r.testErr), hi = Math.max(...te, ...tr);
    return { lines: [{ points: points(tr, 0, hi, n), color: COL.main }, { points: points(te, 0, hi, n), color: COL.test }],
      dots: [...dots(tr, 0, hi, n, COL.main, L), ...dots(te, 0, hi, n, COL.test, L)],
      yTicks: yTicks(0, hi, f1), yTitle: 'mean ‖x − x̂‖²', legend: [{ color: COL.main, label: 'train' }, { color: COL.test, label: 'test' }] };
  }

  function profile(d, L, n) {
    const pr = d.rows.filter(r => r.profile !== null).map(r => r.profile);
    const lo = Math.min(...pr), hi = Math.max(...pr);
    return { lines: [{ points: points(pr, lo, hi, n), color: COL.main }],
      dots: dots(pr, lo, hi, n, COL.main, L).map((dt, i) => (i + 1 === d.best ? { ...dt, r: 7, color: COL.test } : dt)),
      yTicks: yTicks(lo, hi, f1), yTitle: 'profile log-lik ℓ(L)',
      legend: [{ color: COL.main, label: 'ℓ(L)' }, { color: COL.test, label: 'best L = ' + d.best }] };
  }

  const CHARTS = { scree, frac, recon, profile };

  // every chart shares the x axis L = 1..D and a band on the chosen L; the series change per step
  function chart(d, kind, L) {
    const n = d.D;
    const base = {
      band: { x: f1(xs(L - 1, n) - plotW / n / 2), w: f1(plotW / n), y: PT, h: CH - PT - PB },
      bars: [], lines: [], dots: [], ref: []
    };
    return { ...base, ...CHARTS[kind](d, L, n) };
  }

  function stats(d, L) {
    const r = d.rows[L - 1];
    return [
      { k: 'λ_L', v: f2(d.values[L - 1]) },
      { k: 'F_L', v: pct(r.frac) },
      { k: 'train error', v: f2(r.trainErr) },
      { k: 'test error', v: f2(r.testErr) },
      { k: 'ℓ(L)', v: r.profile === null ? '—' : f1(r.profile) }
    ];
  }

  function render(c) {
    const d = model(), s = c.state[KEY], ch = chart(d, KIND[s.p], s.L);
    return {
      ...W().steps(c, KEY, 'pl', phases(d)),
      plL: s.L, plD: d.D,
      plOnL: e => W().patch(c, KEY, { L: parseInt(e.target.value, 10) }),
      plBand: ch.band, plYTicks: ch.yTicks, plYTitle: ch.yTitle,
      plYLo: ch.yTicks[0].label, plYHi: ch.yTicks[ch.yTicks.length - 1].label,
      plBars: ch.bars, plLines: ch.lines, plDots: ch.dots, plRef: ch.ref, plLegend: ch.legend,
      plStats: stats(d, s.L)
    };
  }

  function initState() {
    return { [KEY]: { p: 0, L: 1 } };
  }

  return { initState, render };
})();
