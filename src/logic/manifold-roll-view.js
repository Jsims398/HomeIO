// View logic for Topic 20 ① (what is a manifold, §20.4.1): points on a rolled-up curve in ℝ², which is
// a 1-D manifold. Loaded only by manifold.html; reading-log.js merges render(c) into t20().
// Math is in manifold-math.js; Next/Back controls are Walkthrough.steps in walkthrough.js.
// Also exports geom (plot scale, colour ramp, 1-D strip) for manifold-graph-view.js.
window.ManifoldRollView = (function () {
  const M = () => window.ManifoldMath, W = () => window.Walkthrough;
  const KEY = 'mr';
  const S = 400, PAD = 14, LO = -6.5, HI = 7.5;
  const STRIP = { X0: 22, X1: 378, Y: 34 };
  const EPS = 1.2, HALF_TURN = Math.PI, PCA_HALF = 9, TAN_HALF = 2, DEFAULT_I = 15;
  const STEP = { LOCAL: 1, CHART: 2, PAIR: 3, PCA: 4 };
  const RAMP = [[198, 113, 57], [122, 138, 94], [79, 109, 138]];

  const f1 = v => v.toFixed(1), f2 = v => v.toFixed(2);

  // ---- geometry shared with ③ ----
  const sx = v => PAD + ((v - LO) / (HI - LO)) * (S - 2 * PAD);
  const sy = v => S - PAD - ((v - LO) / (HI - LO)) * (S - 2 * PAD);
  const pt = x => ({ x: f1(sx(x[0])), y: f1(sy(x[1])) });
  const seg = (a, b) => ({ x1: f1(sx(a[0])), y1: f1(sy(a[1])), x2: f1(sx(b[0])), y2: f1(sy(b[1])) });
  const polyline = xs => xs.map(x => f1(sx(x[0])) + ',' + f1(sy(x[1]))).join(' ');

  // orange → olive → blue as u goes 0 → 1 (position along the curve)
  function color(u) {
    const s = Math.min(1, Math.max(0, u)) * (RAMP.length - 1), i = Math.min(RAMP.length - 2, Math.floor(s)), f = s - i;
    const c = RAMP[i].map((v, k) => Math.round(v + f * (RAMP[i + 1][k] - v)));
    return 'rgb(' + c.join(',') + ')';
  }

  // a 1-D number line: each value becomes a dot, coloured by its true position u along the curve
  function strip(values, us, marks) {
    const lo = Math.min(...values), hi = Math.max(...values), span = hi - lo || 1;
    const xs = v => STRIP.X0 + ((v - lo) / span) * (STRIP.X1 - STRIP.X0);
    return {
      dots: values.map((v, n) => ({ x: f1(xs(v)), y: STRIP.Y, color: color(us[n]) })),
      marks: marks.map(n => ({ x: f1(xs(values[n])), y: STRIP.Y }))
    };
  }

  let cached = null;
  function model() {
    if (cached) return cached;
    const sp = M().spiral(), pca = M().pca(sp.X);
    const zPca = sp.X.map(x => M().sum(pca.W[0].map((w, j) => w * (x[j] - pca.mu[j]))));
    cached = { ...sp, pca, zPca, u: sp.t.map(t => t / sp.total), rhoPca: Math.abs(M().spearman(zPca, sp.t)) };
    return cached;
  }

  // the nearest point that is at least half a turn away round the spiral, i.e. across the gap
  function partner(d, i) {
    const across = d.X.map((_, j) => j).filter(j => Math.abs(d.th[j] - d.th[i]) >= HALF_TURN);
    return across.reduce((best, j) => (M().dist(d.X[j], d.X[i]) < M().dist(d.X[best], d.X[i]) ? j : best));
  }

  // the patch inside a circle of radius EPS around point i, and its worst gap from the tangent line
  function patch(d, i) {
    const tan = M().tangent(d.th[i]);
    const inside = d.X.map((_, j) => j).filter(j => M().dist(d.X[j], d.X[i]) <= EPS);
    const gaps = inside.map(j => Math.abs((d.X[j][0] - d.X[i][0]) * tan[1] - (d.X[j][1] - d.X[i][1]) * tan[0]));
    return { tan, inside, gap: Math.max(...gaps) };
  }

  function phases(d, i, j, pa) {
    const e = M().dist(d.X[i], d.X[j]), g = Math.abs(d.t[i] - d.t[j]);
    return [
      { title: 'Start with the data', formula: 'x_n ∈ ℝ²,  D = 2,  N = ' + d.X.length,
        body: 'Points on a curve rolled up like a swiss roll seen end-on. Colour shows how far along the curve each point sits; the algorithm never gets to see that colour.' },
      { title: 'Zoom in: locally it is flat', formula: 'small patch ≈ ℝᵈ,   here d = 1',
        body: 'Inside the circle (radius ' + EPS + ') the ' + pa.inside.length + ' points sit almost on the straight tangent line: the worst gap is ' + f2(pa.gap) +
          '. That is what a manifold is: a space that looks like flat ℝᵈ in every small patch. The tangent line is the tangent space at P, and d = 1 is the intrinsic dimension, even though D = 2.' },
      { title: 'One number per point: a chart', formula: 'z_n = t_n  (distance along the curve)',
        body: 'Unroll the curve and every point gets one coordinate, its arc length t. The strip below is that 1-D picture: the colours come out in order. Finding coordinates like these from x alone is the goal of manifold learning.' },
      { title: 'Two different distances', formula: '‖P − Q‖ = ' + f2(e) + '   vs   along the curve = ' + f2(g),
        body: 'Q is the nearest point on another turn. In ℝ² it is only ' + f2(e) + ' away, but you would travel ' + f2(g) + ' along the curve to reach it. ' +
          'Distance in the plane only matches distance on the manifold for close neighbours, which is why manifold methods trust only small, local distances.' },
      { title: 'A straight line cannot unroll it: PCA', formula: 'z_n = w₁ᵀ(x_n − μ)',
        body: 'PCA projects onto the best straight line w₁. The strip now shows those z: the colours are scrambled, and P and Q land ' + f2(Math.abs(d.zPca[i] - d.zPca[j])) +
          ' apart. Rank correlation with the true order is ' + f2(d.rhoPca) + ' (1 would be perfect). A linear map can\'t follow a curve; that is what the rest of §20.4 is for.' }
    ];
  }

  function stats(d, p, i, j, pa) {
    const base = [{ k: 'P sits at t =', v: f2(d.t[i]) }];
    if (p === STEP.LOCAL) return [...base, { k: 'points in circle', v: String(pa.inside.length) }, { k: 'max gap to tangent', v: f2(pa.gap) }];
    if (p === STEP.CHART) return [...base, { k: 'curve length', v: f2(d.total) }];
    if (p === STEP.PAIR) return [...base, { k: '‖P − Q‖', v: f2(M().dist(d.X[i], d.X[j])) }, { k: 'along the curve', v: f2(Math.abs(d.t[i] - d.t[j])) }];
    if (p === STEP.PCA) return [...base, { k: '|z_P − z_Q| (PCA)', v: f2(Math.abs(d.zPca[i] - d.zPca[j])) }, { k: 'rank corr. (PCA)', v: f2(d.rhoPca) }];
    return base;
  }

  function stripFor(d, p, i, j) {
    if (p < STEP.CHART) return { show: false, dots: [], marks: [], title: '' };
    const usePca = p >= STEP.PCA, st = strip(usePca ? d.zPca : d.t, d.u, [i, j]);
    return { show: true, ...st, title: usePca ? 'PCA score z = w₁ᵀ(x − μ): colours scrambled' : 'arc length t: colours in order' };
  }

  function render(c) {
    const d = model(), s = c.state[KEY], p = s.p, i = s.i, j = partner(d, i), pa = patch(d, i);
    const P = d.X[i], Q = d.X[j], st = stripFor(d, p, i, j);
    const tanEnd = k => [P[0] + k * TAN_HALF * pa.tan[0], P[1] + k * TAN_HALF * pa.tan[1]];
    const w = d.pca.W[0], mu = d.pca.mu;
    const showPair = p >= STEP.PAIR;
    return {
      ...W().steps(c, KEY, 'mr', phases(d, i, j, pa)),
      mrI: i, mrMaxI: d.X.length - 1,
      mrOnI: e => W().patch(c, KEY, { i: parseInt(e.target.value, 10) }),
      mrCurve: polyline(d.X),
      mrDots: d.X.map((x, n) => ({ ...pt(x), color: color(d.u[n]), r: p === STEP.LOCAL && pa.inside.includes(n) ? 5.5 : 4 })),
      mrP: pt(P), mrPLbl: { x: f1(sx(P[0]) + 9), y: f1(sy(P[1]) - 9) },
      mrShowLocal: p === STEP.LOCAL,
      mrCircle: { cx: f1(sx(P[0])), cy: f1(sy(P[1])), r: f1((EPS / (HI - LO)) * (S - 2 * PAD)) },
      mrTangent: seg(tanEnd(-1), tanEnd(1)),
      mrShowPair: showPair,
      mrQ: pt(Q), mrQLbl: { x: f1(sx(Q[0]) + 9), y: f1(sy(Q[1]) - 9) },
      mrPairSeg: seg(P, Q), mrArc: polyline(d.X.slice(Math.min(i, j), Math.max(i, j) + 1)),
      mrShowPca: p >= STEP.PCA, mrPcaLine: seg(mu.map((m, k) => m - PCA_HALF * w[k]), mu.map((m, k) => m + PCA_HALF * w[k])),
      mrShowStrip: st.show, mrStrip: st.dots, mrStripTitle: st.title,
      mrStripP: st.marks[0] || { x: 0, y: 0 }, mrStripQ: st.marks[1] || { x: 0, y: 0 },
      mrStats: stats(d, p, i, j, pa)
    };
  }

  function initState() {
    return { [KEY]: { p: 0, i: DEFAULT_I } };
  }

  const geom = { sx, sy, pt, seg, polyline, color, strip };
  return { initState, render, geom, model };
})();
