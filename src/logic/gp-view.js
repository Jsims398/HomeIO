// View logic for Topic 12 (Gaussian process regression, §17.2.1 – 17.2.2).
// Loaded only by gp.html. reading-log.js calls GPView.initState()
// for the starting state and GPView.render(component) from t12(); the math is in gp-math.js.
window.GPView = (function () {
  const M = () => window.GPMath;
  const SERIES = ['var(--color-accent)', 'var(--color-accent-2-600)', '#4f6d8a', 'var(--color-accent-700)', 'var(--color-accent-2-800)', 'var(--color-neutral-700)'];
  const POS = 'var(--color-accent-2-600)', NEG = 'var(--color-accent)';

  // shared plot box over x ∈ [0, 10], f ∈ [−3, 3]
  const W = 560, H = 220, PL = 34, PR = 14, PT = 14, PB = 34;
  const X_LO = 0, X_HI = 10, Y_LO = -3, Y_HI = 3;
  // scatter box for the 2-D Gaussian picture (y₁, f(x₂))
  const SC = 240, SC_C = 120, SC_S = 34, SC_LIM = 3.3;
  // §17.2.1 – 17.2.2 labs
  const LAB_G = 81, LAB_DRAWS = 3, LAB_GRAB = 12, LAB_MAX = 25, WEIGHT_MID = 40, WEIGHT_H = 34, SHOOT_OFF = 3.2;
  const NF_START = [[1.2, -0.6], [3, 0.9], [4.5, 1.4], [6.8, -0.8], [8.6, 0.3]];
  // two dots at x ≈ 3.4 that disagree by 1.2: noise-free has to pass through both
  const NY_START = [[0.6, -0.3], [1.3, 0.8], [2.1, 1.2], [2.7, 0.6], [3.3, 1.3], [3.45, 0.1], [4.4, -0.5], [5.2, -1.1],
    [6.1, -0.6], [7.0, 0.4], [7.9, 1.0], [8.8, 0.5], [9.5, -0.2]];

  const f1 = v => v.toFixed(1);
  const f2 = v => v.toFixed(2);
  const pill = active => ({ bg: active ? 'var(--color-accent)' : 'transparent', fg: active ? 'var(--color-bg)' : 'var(--color-text)' });
  const capture = e => { try { e.currentTarget.setPointerCapture(e.pointerId); } catch (err) { /* capture is optional */ } };
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const svgX = (e, width) => { const r = e.currentTarget.getBoundingClientRect(); return ((e.clientX - r.left) / r.width) * width; };
  const svgY = (e, height) => { const r = e.currentTarget.getBoundingClientRect(); return ((e.clientY - r.top) / r.height) * height; };
  const patch = (c, key, change) => c.setState({ [key]: { ...c.state[key], ...change } });
  const num = e => parseFloat(e.target.value);

  // ---------- plot helpers ----------

  function frame(x0, x1, y0, y1) {
    const X = x => PL + ((x - x0) / (x1 - x0)) * (W - PL - PR);
    const Y = y => H - PB - ((clamp(y, y0, y1) - y0) / (y1 - y0)) * (H - PT - PB);
    const inv = px => x0 + ((px - PL) / (W - PL - PR)) * (x1 - x0);
    const invY = py => y0 + ((H - PB - py) / (H - PT - PB)) * (y1 - y0);
    return { X, Y, inv, invY, x0, x1, y0, y1 };
  }

  const pathFrom = (fr, xs, ys) => xs.map((x, i) => (i ? 'L ' : 'M ') + f1(fr.X(x)) + ' ' + f1(fr.Y(ys[i]))).join(' ');

  function bandPath(fr, xs, mean, half) {
    const up = xs.map((x, i) => f1(fr.X(x)) + ' ' + f1(fr.Y(mean[i] + half[i])));
    const down = xs.map((x, i) => f1(fr.X(x)) + ' ' + f1(fr.Y(mean[i] - half[i]))).reverse();
    return 'M ' + up.join(' L ') + ' L ' + down.join(' L ') + ' Z';
  }

  const xTicks = fr => [0, 2, 4, 6, 8, 10].map(v => ({ x: f1(fr.X(v)), label: String(v) }));
  const yTicks = fr => [-2, 0, 2].map(v => ({ y: f1(fr.Y(v) + 4), label: String(v) }));

  const scX = v => SC_C + clamp(v, -SC_LIM, SC_LIM) * SC_S;
  const scY = v => SC_C - clamp(v, -SC_LIM, SC_LIM) * SC_S;

  // the k-standard-deviation ellipse of a 2-D Gaussian with covariance [[a, b], [b, d]]
  function ellipse(a, b, d, k) {
    const mid = (a + d) / 2, rad = Math.sqrt(((a - d) / 2) ** 2 + b * b);
    const deg = (Math.atan2(2 * b, a - d) / 2) * (180 / Math.PI);
    return {
      rx: f1(k * SC_S * Math.sqrt(mid + rad)), ry: f1(k * SC_S * Math.sqrt(Math.max(0, mid - rad))),
      t: 'rotate(' + f1(-deg) + ' ' + SC_C + ' ' + SC_C + ')'
    };
  }

  const zsFor = (count, n, seed) => { const rand = M().rng(seed); return Array.from({ length: count }, () => M().normals(n, rand)); };

  // ---------- state ----------

  function initState() {
    return {
      gpSlice: { l: 1.5, x1: 3, x2: 4.5, a: 1.5, noise: 0 },
      gpNf: { pts: NF_START, l: 1, sf: 1 },
      gpNy: { pts: NY_START, l: 1, sf: 1, noise: 0.3, q: 3.4, showNf: false, bandY: false },
      gpZ: zsFor(LAB_DRAWS, LAB_G, 12)
    };
  }

  // ---------- §17.2.1: conditioning = slicing the Gaussian ----------

  function sliceNote(s, cond) {
    const shrink = cond.k12 / cond.k11;
    if (s.noise > 0.05) return 'With noise, the observed y₁ might be off by about σ_y, so the oval is stretched sideways and the slice is wider. The guess for f(x₂) is pulled toward 0 (shrink factor ' + f2(shrink) + ') and is never certain, even at x₂ = x₁.';
    if (Math.abs(s.x1 - s.x2) < 0.05) return 'x₂ is on top of the observation, so the slice has zero width: noise-free means we know f(x₁) exactly. This is why the band pinches to nothing at every data point.';
    return 'The slice through the oval at f(x₁) = ' + f2(s.a) + ' is itself a bell curve. Its centre is the posterior mean of f(x₂) and its width is the posterior uncertainty. Do this at every x₂ and you get the whole band on the left.';
  }

  function slice(c) {
    const s = c.state.gpSlice, D = M(), p = { l: s.l, sf: 1 };
    const grid = D.linspace(X_LO, X_HI, LAB_G), fr = frame(X_LO, X_HI, Y_LO, Y_HI);
    const k11 = 1 + s.noise * s.noise, ks = grid.map(x => D.kernel(p, x, s.x1));
    const mean = ks.map(k => (k / k11) * s.a), sd = ks.map(k => Math.sqrt(Math.max(0, 1 - (k * k) / k11)));
    const cond = D.condition2(p, s.x1, s.x2, s.noise, s.a), drawSd = Math.max(cond.sd, 0.03);
    const fs = D.linspace(-SC_LIM, SC_LIM, 161);
    const bell = 'M ' + f1(scX(s.a)) + ' ' + f1(scY(-SC_LIM)) + ' ' + fs.map(f => 'L ' + f1(scX(s.a) + 46 * Math.exp(-((f - cond.mean) ** 2) / (2 * drawSd * drawSd))) + ' ' + f1(scY(f))).join(' ') + ' L ' + f1(scX(s.a)) + ' ' + f1(scY(SC_LIM)) + ' Z';
    const drag = e => {
      if (e.type === 'pointerdown') capture(e);
      else if (!e.buttons) return;
      patch(c, 'gpSlice', { a: clamp((svgX(e, SC) - SC_C) / SC_S, -2.5, 2.5) });
    };
    return {
      slBand: bandPath(fr, grid, mean, sd.map(v => 2 * v)), slMean: pathFrom(fr, grid, mean),
      slObsX: f1(fr.X(s.x1)), slObsY: f1(fr.Y(s.a)),
      slX2px: f1(fr.X(s.x2)), slTop: PT, slBottom: H - PB,
      slSegY1: f1(fr.Y(cond.mean + 2 * cond.sd)), slSegY2: f1(fr.Y(cond.mean - 2 * cond.sd)), slMeanY: f1(fr.Y(cond.mean)),
      slTicks: xTicks(fr), slYTicks: yTicks(fr), slZero: f1(fr.Y(0)),
      slEll: [1, 2].map(k => ellipse(cond.k11, cond.k12, cond.k22, k)),
      slLineX: f1(scX(s.a)), slBell: bell, slDotY: f1(scY(cond.mean)),
      slScSeg1: f1(scY(cond.mean + 2 * cond.sd)), slScSeg2: f1(scY(cond.mean - 2 * cond.sd)),
      slOnDown: drag, slOnMove: drag,
      slA: s.a, slALabel: f2(s.a), slOnA: e => patch(c, 'gpSlice', { a: num(e) }),
      slX1: s.x1, slX1Label: f2(s.x1), slOnX1: e => patch(c, 'gpSlice', { x1: num(e) }),
      slX2: s.x2, slX2Label: f2(s.x2), slOnX2: e => patch(c, 'gpSlice', { x2: num(e) }),
      slL: s.l, slLLabel: f2(s.l), slOnL: e => patch(c, 'gpSlice', { l: num(e) }),
      slNoise: s.noise, slNoiseLabel: f2(s.noise), slOnNoise: e => patch(c, 'gpSlice', { noise: num(e) }),
      slK12: f2(cond.k12), slK11: f2(cond.k11), slShrink: f2(cond.k12 / cond.k11),
      slPostMean: f2(cond.mean), slPostSd: f2(cond.sd), slNote: sliceNote(s, cond)
    };
  }

  // ---------- §17.2.1 – 17.2.2: the GP regression labs ----------

  function labClick(c, key, fr, e) {
    const s = c.state[key], px = svgX(e, W), py = svgY(e, H);
    const near = s.pts.findIndex(([x, y]) => Math.hypot(fr.X(x) - px, fr.Y(y) - py) < LAB_GRAB);
    if (near >= 0) { patch(c, key, { pts: s.pts.filter((_, i) => i !== near) }); return; }
    const x = fr.inv(px), y = fr.invY(py);
    if (x < X_LO || x > X_HI || y < Y_LO || y > Y_HI || s.pts.length >= LAB_MAX) return;
    patch(c, key, { pts: [...s.pts, [x, y]] });
  }

  // everything both labs draw; `noise` is 0 for the noise-free lab
  function labCore(c, key, noise, bandExtra, start) {
    const s = c.state[key], D = M(), p = { l: s.l, sf: s.sf };
    const xs = s.pts.map(q => q[0]), ys = s.pts.map(q => q[1]);
    const grid = D.linspace(X_LO, X_HI, LAB_G), fr = frame(X_LO, X_HI, Y_LO, Y_HI);
    const post = D.posterior(p, xs, ys, noise, grid, true);
    const draws = post ? D.drawsFrom(post.mean, post.cov, c.state.gpZ, s.sf * s.sf) || [] : [];
    const half = post ? post.sd.map(v => 2 * Math.sqrt(v * v + bandExtra)) : [];
    return {
      p, xs, ys, fr, grid,
      out: {
        Band: post ? bandPath(fr, grid, post.mean, half) : '', Mean: post ? pathFrom(fr, grid, post.mean) : '',
        Samples: draws.map((f, k) => ({ d: pathFrom(fr, grid, f), color: SERIES[k + 1] })),
        Dots: s.pts.map(([x, y]) => ({ x: f1(fr.X(x)), y: f1(fr.Y(y)) })),
        L: s.l, LLabel: f2(s.l), OnL: e => patch(c, key, { l: num(e) }),
        Sf: s.sf, SfLabel: f2(s.sf), OnSf: e => patch(c, key, { sf: num(e) }),
        OnDown: e => labClick(c, key, fr, e),
        OnClear: () => patch(c, key, { pts: [] }), OnReset: () => patch(c, key, { pts: start }),
        Count: s.pts.length, Ticks: xTicks(fr), YTicks: yTicks(fr), Zero: f1(fr.Y(0)),
        ShootsOff: post ? post.mean.some(m => Math.abs(m) > SHOOT_OFF) : false
      }
    };
  }

  const prefixed = (prefix, obj) => Object.fromEntries(Object.entries(obj).map(([k, v]) => [prefix + k, v]));

  function noiseFreeNote(n, shootsOff) {
    if (!n) return 'No data yet, so this is the prior: three random functions the kernel thinks are plausible, and a band of ± 2σ_f everywhere. Click to add a point.';
    if (shootsOff) return 'The curve shoots off the chart. Two points close together with different heights force a noise-free GP into a huge swing to pass through both. Real data is rarely this exact, which is the reason for 17.2.2.';
    return 'Every sampled curve passes exactly through every point, and the band pinches to zero there: noise-free means f(xₙ) is known for certain. Away from the data the band relaxes back to the prior width ± 2σ_f.';
  }

  function noiseFree(c) {
    const { p, xs, ys, out } = labCore(c, 'gpNf', 0, 0, NF_START);
    const atData = xs.length ? M().posterior(p, xs, ys, 0, xs) : null;
    return prefixed('nf', {
      ...out,
      AtData: atData ? f2(Math.max(...atData.sd)) : '—',
      Note: noiseFreeNote(xs.length, out.ShootsOff)
    });
  }

  function noisyNote(s) {
    if (s.noise < 0.08) return 'Almost noise-free: the curve must pass (nearly) through every dot, so the two disagreeing dots near x = 3.4 make it swing wildly. Turn σ_y up.';
    if (s.noise > 0.8) return 'σ_y is so large the GP thinks the dots are mostly noise. It barely trusts them, so the curve flattens toward 0 (the prior mean) and the band stays wide.';
    return 'The curve no longer has to touch the dots: each y is treated as f(x) plus a random error of size ≈ σ_y. It runs between dots that disagree, and the band never pinches to zero.';
  }

  function weightBars(fr, xs, w) {
    const top = Math.max(1e-9, ...w.map(Math.abs));
    return xs.map((x, n) => {
      const h = (Math.abs(w[n]) / top) * WEIGHT_H;
      return { x: f1(fr.X(x) - 3), y: f1(w[n] >= 0 ? WEIGHT_MID - h : WEIGHT_MID), h: f1(Math.max(h, 0.5)), fill: w[n] >= 0 ? POS : NEG };
    });
  }

  function noisy(c) {
    const s = c.state.gpNy, D = M(), extra = s.bandY ? s.noise * s.noise : 0;
    const { p, xs, ys, fr, grid, out } = labCore(c, 'gpNy', s.noise, extra, NY_START);
    const w = D.weights(p, xs, s.noise, s.q), pred = w.reduce((a, wn, n) => a + wn * ys[n], 0);
    const here = D.posterior(p, xs, ys, s.noise, [s.q]), sdF = here ? here.sd[0] : 0;
    const nf = s.showNf ? D.posterior(p, xs, ys, 0, grid) : null;
    return prefixed('ny', {
      ...out,
      NfCurve: nf ? pathFrom(fr, grid, nf.mean) : '',
      Noise: s.noise, NoiseLabel: f2(s.noise), OnNoise: e => patch(c, 'gpNy', { noise: num(e) }),
      NfBtn: s.showNf ? 'Hide noise-free fit' : 'Show noise-free fit (dashed)', OnNf: () => patch(c, 'gpNy', { showNf: !s.showNf }),
      BandBtns: [[false, 'Band for f (the true curve)'], [true, 'Band for a new y (adds σ_y²)']].map(([v, label]) => ({ label, ...pill(v === s.bandY), onClick: () => patch(c, 'gpNy', { bandY: v }) })),
      Q: s.q, QLabel: f2(s.q), OnQ: e => patch(c, 'gpNy', { q: num(e) }),
      Qx: f1(fr.X(s.q)), Qy: f1(fr.Y(pred)), Top: PT, Bottom: H - PB,
      Weights: weightBars(fr, xs, w),
      Pred: f2(pred), PredSd: '± ' + f2(2 * Math.sqrt(sdF * sdF + extra)),
      WSum: xs.length ? f2(w.reduce((a, b) => a + b, 0)) : '—',
      NegCount: String(w.filter(v => v < -1e-3).length),
      Note: noisyNote(s)
    });
  }

  function render(c) {
    return { ...slice(c), ...noiseFree(c), ...noisy(c) };
  }

  return { initState, render };
})();
