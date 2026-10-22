// View logic for Topic 11 (kernel density estimation, §16.3).
// Loaded only by kde.html. reading-log.js calls KDEView.initState()
// for the starting state and KDEView.render(component) from t11(); the math is in kde-math.js.
window.KDEView = (function () {
  const M = () => window.KDEMath;
  const COLORS = ['#c67139', '#7a8a5e', '#4f6d8a'];
  const CLASS_NAMES = ['orange', 'green'];
  const KINDS = {
    gaussian: { label: 'Gaussian', note: 'The bell curve. Smooth everywhere, and its tails never quite reach zero, so every data point has at least a tiny say everywhere.' },
    boxcar: { label: 'Boxcar', note: 'A flat block. Anything within h of the center counts fully; anything farther counts zero. Simple, but its sharp edges make the final curve look like stairs.' },
    epanechnikov: { label: 'Epanechnikov', note: 'An upside-down parabola that touches zero at ±h. Mathematically it is the most efficient shape, but in practice the width h matters far more than the shape.' },
    tricube: { label: 'Tri-cube', note: 'Flat on top, then curves smoothly down to exactly zero at ±h. Popular for kernel regression: smooth, and far-away points are ignored completely.' }
  };

  // shared 1-D plot box
  const W = 560, H = 220, PL = 34, PR = 14, PT = 14, PB = 34;
  // §16.3.2 Parzen lab
  const PZ_LO = 0, PZ_HI = 10, PZ_GRAB = 0.2, PZ_MAX_PTS = 40, PZ_MAX_NEW = 5000, PZ_SHOW_NEW = 250, PZ_BIN = 0.25;
  const PZ_START = [1.4, 2.1, 2.5, 2.9, 5.2, 7.3, 7.8];
  // §16.3.3 bandwidth lab
  const BW_LO = -4, BW_HI = 5, BW_YMAX = 0.9, BW_HMIN = 0.03, BW_HMAX = 2.5, BW_GRID = 60;
  // §16.3.4 KNN lab
  const NN_C = 150, NN_S = 56, NN_GRID = 30;
  // §16.3.5 regression lab
  const RG_LO = 0, RG_HI = 6, RG_YLO = -1, RG_YHI = 4, RG_SY = 0.2, RG_EDGE = 0.6;

  const f1 = v => v.toFixed(1);
  const f2 = v => v.toFixed(2);
  const pill = active => ({ bg: active ? 'var(--color-accent)' : 'transparent', fg: active ? 'var(--color-bg)' : 'var(--color-text)' });
  const capture = e => { try { e.currentTarget.setPointerCapture(e.pointerId); } catch (err) { /* capture is optional */ } };
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const svgX = (e, width) => { const r = e.currentTarget.getBoundingClientRect(); return ((e.clientX - r.left) / r.width) * width; };
  const svgY = (e, height) => { const r = e.currentTarget.getBoundingClientRect(); return ((e.clientY - r.top) / r.height) * height; };

  // ---------- plot helpers ----------

  function frame(x0, x1, y0, y1) {
    const X = x => PL + ((x - x0) / (x1 - x0)) * (W - PL - PR);
    const Y = y => H - PB - ((clamp(y, y0, y1) - y0) / (y1 - y0)) * (H - PT - PB);
    const inv = px => x0 + ((px - PL) / (W - PL - PR)) * (x1 - x0);
    return { X, Y, inv, x0, x1, base: Y(0) };
  }

  function curvePath(fr, fn, steps = 300) {
    let d = '';
    for (let i = 0; i <= steps; i++) {
      const x = fr.x0 + ((fr.x1 - fr.x0) * i) / steps;
      d += (i ? 'L ' : 'M ') + f1(fr.X(x)) + ' ' + f1(fr.Y(fn(x))) + ' ';
    }
    return d;
  }

  const areaPath = (fr, fn, steps) => curvePath(fr, fn, steps) + 'L ' + f1(fr.X(fr.x1)) + ' ' + f1(fr.base) + ' L ' + f1(fr.X(fr.x0)) + ' ' + f1(fr.base) + ' Z';
  const ticks = (fr, vals) => vals.map(v => ({ x: f1(fr.X(v)), label: String(v) }));
  const range = (lo, hi, step) => Array.from({ length: Math.round((hi - lo) / step) + 1 }, (_, i) => lo + i * step);

  function shapeBtns(c, key) {
    return Object.keys(KINDS).map(k => ({ label: KINDS[k].label, ...pill(k === c.state[key]), onClick: () => c.setState(onShape(key, k)) }));
  }

  // switching the Parzen kernel makes old generated points stale
  const onShape = (key, k) => (key === 'pzSh' ? { pzSh: k, pzNew: [], pzLast: null } : { [key]: k });

  // ---------- state ----------

  function hToSlider(h) { return (100 * Math.log(h / BW_HMIN)) / Math.log(BW_HMAX / BW_HMIN); }
  function sliderToH(v) { return BW_HMIN * Math.pow(BW_HMAX / BW_HMIN, v / 100); }

  function freshBw(n, seed) {
    const D = M(), xs = D.sampleTruth(n, D.rng(seed));
    const grid = Array.from({ length: BW_GRID }, (_, i) => sliderToH((100 * i) / (BW_GRID - 1)));
    const scores = grid.map(h => D.looScore(xs, h));
    const bestI = scores.indexOf(Math.max(...scores));
    return { bwN: n, bwSeed: seed, bwXs: xs, bwLoo: { grid, scores, best: grid[bestI], bestScore: scores[bestI] }, bwSil: D.silverman(xs) };
  }

  function initState() {
    const D = M();
    return {
      kSh: 'gaussian', kH: 0.6,
      pzPts: PZ_START, pzSh: 'gaussian', pzH: 0.6, pzNew: [], pzLast: null, pzRand: D.rng(2026),
      ...freshBw(60, 3), bwH: 0.08,
      nnData: D.blobs(D.rng(5)), nnQ: [0.1, 0.2], nnMode: 'balloon', nnK: 5, nnR: 0.5,
      rgData: D.regData(D.rng(9)), rgQ: 3, rgH: 0.5, rgMode: 'nw'
    };
  }

  // ---------- §16.3.1: kernel shapes ----------

  function kernels(c) {
    const s = c.state, D = M(), fr = frame(-3, 3, 0, 1.9);
    const peak = D.kernel(s.kSh, 0, s.kH);
    return {
      kShapeBtns: shapeBtns(c, 'kSh'),
      kH: s.kH, kHLabel: f2(s.kH), onKH: e => c.setState({ kH: parseFloat(e.target.value) }),
      kFill: areaPath(fr, x => D.kernel(s.kSh, x, s.kH), 600),
      kCurve: curvePath(fr, x => D.kernel(s.kSh, x, s.kH), 600),
      kGhosts: Object.keys(KINDS).filter(k => k !== s.kSh).map(k => ({ d: curvePath(fr, x => D.kernel(k, x, s.kH), 600) })),
      kEdges: [-s.kH, s.kH].map(v => ({ x: f1(fr.X(v)), label: v < 0 ? '−h' : '+h' })),
      kTicks: ticks(fr, [-3, -2, -1, 0, 1, 2, 3]), kBase: f1(fr.base),
      kArea: D.area(s.kSh, s.kH).toFixed(2), kPeak: f2(peak),
      kReach: s.kSh === 'gaussian' ? 'never (tails go on forever)' : 'at ± ' + f2(s.kH),
      kNote: KINDS[s.kSh].note
    };
  }

  // ---------- §16.3.2: Parzen window + generating new data ----------

  function pzGenerate(c, count) {
    const s = c.state, D = M();
    if (!s.pzPts.length) return;
    const draws = Array.from({ length: count }, () => D.sample(s.pzPts, s.pzH, s.pzSh, s.pzRand));
    const last = draws[draws.length - 1];
    c.setState({ pzNew: [...s.pzNew, ...draws.map(d => d.x)].slice(-PZ_MAX_NEW), pzLast: { ...last, from: s.pzPts[last.i] } });
  }

  function pzClick(c, e) {
    const s = c.state, fr = frame(PZ_LO, PZ_HI, 0, 1), x = fr.inv(svgX(e, W));
    if (x < PZ_LO || x > PZ_HI) return;
    const near = s.pzPts.findIndex(p => Math.abs(p - x) < PZ_GRAB);
    if (near < 0 && s.pzPts.length >= PZ_MAX_PTS) return;
    const pts = near >= 0 ? s.pzPts.filter((_, i) => i !== near) : [...s.pzPts, x];
    c.setState({ pzPts: pts, pzNew: [], pzLast: null });
  }

  function pzHistogram(fr, xs) {
    if (!xs.length) return [];
    const bins = range(PZ_LO, PZ_HI - PZ_BIN, PZ_BIN).map(lo => ({ lo, n: 0 }));
    xs.forEach(x => { const b = Math.floor((x - PZ_LO) / PZ_BIN); if (b >= 0 && b < bins.length) bins[b].n += 1; });
    return bins.filter(b => b.n).map(b => {
      const y = fr.Y(b.n / (xs.length * PZ_BIN));
      return { x: f1(fr.X(b.lo)), y: f1(y), w: f1(fr.X(b.lo + PZ_BIN) - fr.X(b.lo) - 1), h: f1(fr.base - y) };
    });
  }

  function pzSteps(s) {
    const L = s.pzLast;
    if (!L) return ['Press “Generate 1” to watch one new data point being made, step by step.'];
    const n = s.pzPts.length, sign = L.nudge >= 0 ? '+' : '−', size = f2(Math.abs(L.nudge));
    return [
      'Step 1 · Pick a data point at random. Every point has the same 1-in-' + n + ' chance. This time: the point at x = ' + f2(L.from) + '.',
      'Step 2 · Draw a random nudge from that point’s ' + KINDS[s.pzSh].label + ' bump (width h = ' + f2(s.pzH) + '): ' + sign + size + '.',
      'Step 3 · New point = ' + f2(L.from) + ' ' + sign + ' ' + size + ' = ' + f2(L.x) + '. It is brand new, but it looks like it came from the same place as the real data.'
    ];
  }

  function pzNote(s, n) {
    if (!n) return 'No data yet. Click anywhere on the number line to drop a data point.';
    if (s.pzNew.length >= 300) return 'The bars are a histogram of the ' + s.pzNew.length + ' generated points. They follow the thick curve: “pick a point, then nudge it” really does draw from p̂(x).';
    if (s.pzH < 0.2) return 'With a very narrow bump, every new point lands almost on top of a real one. The model is basically copying its data. Widen h to make genuinely new values.';
    return 'Each thin bump is one data point’s share (1/' + n + ' of the total). The thick curve is all of them stacked: tall where points crowd together, flat where there are none.';
  }

  function parzen(c) {
    const s = c.state, D = M(), n = s.pzPts.length;
    const est = x => D.kde(s.pzPts, x, s.pzH, s.pzSh);
    let top = 0;
    for (let i = 0; i <= 200; i++) top = Math.max(top, est(PZ_LO + ((PZ_HI - PZ_LO) * i) / 200));
    const fr = frame(PZ_LO, PZ_HI, 0, Math.max(0.3, top * 1.15));
    const bump = xn => x => D.kernel(s.pzSh, x - xn, s.pzH) / n;
    const L = s.pzLast, newY = fr.base + 18;
    const inView = x => x >= PZ_LO && x <= PZ_HI;
    return {
      pzShapeBtns: shapeBtns(c, 'pzSh'),
      pzH: s.pzH, pzHLabel: f2(s.pzH), onPzH: e => c.setState({ pzH: parseFloat(e.target.value), pzNew: [], pzLast: null }),
      pzBumps: n ? s.pzPts.map(xn => ({ d: curvePath(fr, bump(xn), 300) })) : [],
      pzCurve: n ? curvePath(fr, est, 400) : '',
      pzFill: n ? areaPath(fr, est, 400) : '',
      pzPick: L ? [{ d: areaPath(fr, bump(L.from), 300) }] : [],
      pzArrow: L ? [{ x1: f1(fr.X(L.from)), y1: f1(fr.base), x2: f1(fr.X(clamp(L.x, PZ_LO, PZ_HI))), y2: f1(newY) }] : [],
      pzDots: s.pzPts.map((x, i) => ({ x: f1(fr.X(x)), y: f1(fr.base), ring: L && i === L.i ? 'var(--color-text)' : 'var(--color-surface)' })),
      pzNewDots: s.pzNew.slice(-PZ_SHOW_NEW).filter(inView).map(x => ({ x: f1(fr.X(x)), y: f1(newY) })),
      pzLastDot: L && inView(L.x) ? [{ x: f1(fr.X(L.x)), y: f1(newY) }] : [],
      pzHist: pzHistogram(fr, s.pzNew),
      pzTicks: ticks(fr, range(0, 10, 1)), pzBase: f1(fr.base),
      onPzDown: e => pzClick(c, e),
      onPzGen1: () => pzGenerate(c, 1), onPzGen100: () => pzGenerate(c, 100),
      onPzClearNew: () => c.setState({ pzNew: [], pzLast: null }),
      onPzClearPts: () => c.setState({ pzPts: [], pzNew: [], pzLast: null }),
      onPzReset: () => c.setState({ pzPts: PZ_START, pzNew: [], pzLast: null }),
      pzCount: n, pzNewCount: s.pzNew.length,
      pzSteps: pzSteps(s).map(t => ({ t })), pzNote: pzNote(s, n)
    };
  }

  // ---------- §16.3.3: choosing the bandwidth ----------

  function bwVerdict(h, best) {
    const ratio = h / best;
    if (ratio < 0.5) return { label: 'Too spiky', bg: 'var(--color-accent-200)', note: 'Overfitting. Each data point gets its own skinny spike, so the curve says “only these exact values are possible.” A new value between the spikes gets almost zero probability, which is why the leave-one-out score drops.' };
    if (ratio > 2) return { label: 'Too smooth', bg: 'var(--color-accent-100)', note: 'Underfitting. The bumps are so wide that the two humps blur together. The curve is smooth but hides real structure in the data.' };
    return { label: 'About right', bg: 'var(--color-accent-2-100)', note: 'Close to the h that predicts held-out points best. The curve keeps both humps without inventing extra wiggles.' };
  }

  function looChart(s) {
    const { grid, scores, best, bestScore } = s.bwLoo;
    const lo = Math.max(Math.min(...scores), bestScore - 3), hi = bestScore + 0.15;
    const X = h => 30 + (hToSlider(h) / 100) * 260, Y = v => 104 - ((clamp(v, lo, hi) - lo) / (hi - lo)) * 94;
    const d = grid.map((h, i) => (i ? 'L ' : 'M ') + f1(X(h)) + ' ' + f1(Y(scores[i]))).join(' ');
    const now = M().looScore(s.bwXs, s.bwH);
    return { path: d, now, nowPt: { x: f1(X(s.bwH)), y: f1(Y(now)) }, bestPt: { x: f1(X(best)), y: f1(Y(bestScore)) } };
  }

  function bandwidth(c) {
    const s = c.state, D = M(), fr = frame(BW_LO, BW_HI, 0, BW_YMAX);
    const v = bwVerdict(s.bwH, s.bwLoo.best), loo = looChart(s);
    return {
      bwTruth: curvePath(fr, D.truePdf, 300),
      bwCurve: curvePath(fr, x => D.kde(s.bwXs, x, s.bwH), 500),
      bwRug: s.bwXs.filter(x => x >= BW_LO && x <= BW_HI).map(x => ({ x: f1(fr.X(x)), y1: f1(fr.base), y2: f1(fr.base + 9) })),
      bwTicks: ticks(fr, range(-4, 5, 1)), bwBase: f1(fr.base),
      bwSlider: hToSlider(s.bwH).toFixed(1), onBwH: e => c.setState({ bwH: sliderToH(parseFloat(e.target.value)) }),
      bwHLabel: s.bwH.toFixed(3), bwSilLabel: s.bwSil.toFixed(3), bwBestLabel: s.bwLoo.best.toFixed(3),
      onBwSil: () => c.setState({ bwH: s.bwSil }), onBwBest: () => c.setState({ bwH: s.bwLoo.best }),
      bwN: s.bwN, onBwN: e => c.setState(freshBw(parseInt(e.target.value, 10), s.bwSeed)),
      onBwNew: () => c.setState(freshBw(s.bwN, s.bwSeed + 1)),
      bwVerdict: v.label, bwVerdictBg: v.bg, bwNote: v.note,
      bwLooPath: loo.path, bwLooNow: loo.nowPt, bwLooBest: loo.bestPt, bwScore: loo.now.toFixed(2)
    };
  }

  // ---------- §16.3.4: from KDE to KNN ----------

  const nnXY = p => [NN_C + p[0] * NN_S, NN_C - p[1] * NN_S];
  const nnWorld = e => [clamp((svgX(e, 300) - NN_C) / NN_S, -2.6, 2.6), clamp((NN_C - svgY(e, 300)) / NN_S, -2.6, 2.6)];

  function nnQuery(s, q) {
    const D = M(), d = s.nnData;
    return s.nnMode === 'balloon' ? D.balloon(d.X, d.y, q, s.nnK) : D.window2d(d.X, d.y, q, s.nnR);
  }

  // faint background: what the classifier would say at each spot
  function nnMap(s) {
    const cell = 300 / NN_GRID, out = [];
    for (let i = 0; i < NN_GRID; i++) {
      for (let j = 0; j < NN_GRID; j++) {
        const q = [((i + 0.5) * cell - NN_C) / NN_S, (NN_C - (j + 0.5) * cell) / NN_S], r = nnQuery(s, q);
        const fill = !r.probs ? 'var(--color-neutral-400)' : r.probs[0] === r.probs[1] ? 'transparent' : COLORS[r.probs[0] > r.probs[1] ? 0 : 1];
        out.push({ x: f1(i * cell), y: f1(j * cell), w: f1(cell), fill });
      }
    }
    return out;
  }

  function nnNote(s, r) {
    if (!r.probs) return 'Empty window: no data points inside the circle, so the estimate is 0 ⁄ 0 and the classifier cannot decide (grey areas on the map). Switch to the balloon, which always grows until it finds K points.';
    if (s.nnMode === 'window') return 'Fixed window (Parzen): the circle is always the same size, so in crowded places it holds lots of points and in empty places it may hold none. Drag the query into a corner and watch it go grey.';
    if (s.nnK === 1) return 'K = 1: the guess is just the colour of the single closest point. The map becomes a jigsaw of tiny islands, very sensitive to every individual point.';
    return 'Balloon (KNN): the radius adapts. Small where points are crowded, big where they are spread out. The vote is p(y = c | x) = (colour-c points inside) ⁄ K.';
  }

  function knn(c) {
    const s = c.state, d = s.nnData, r = nnQuery(s, s.nnQ), inside = new Set(r.inside), [qx, qy] = nnXY(s.nnQ);
    const balloonMode = s.nnMode === 'balloon';
    const move = e => {
      if (e.type === 'pointerdown') capture(e);
      else if (!e.buttons) return;
      c.setState({ nnQ: nnWorld(e) });
    };
    const decided = r.probs && r.probs[0] !== r.probs[1];
    return {
      nnMap: nnMap(s),
      nnDots: d.X.map((p, i) => {
        const [x, y] = nnXY(p), hit = inside.has(i);
        return { x: f1(x), y: f1(y), fill: COLORS[d.y[i]], r: hit ? 6.5 : 4.2, stroke: hit ? 'var(--color-text)' : 'var(--color-surface)', sw: hit ? 2 : 1 };
      }),
      nnQx: f1(qx), nnQy: f1(qy), nnRpx: f1(r.r * NN_S),
      onNnDown: move, onNnMove: move,
      nnModeBtns: [['balloon', 'Balloon · K nearest'], ['window', 'Fixed window · radius h']].map(([k, label]) => ({ label, ...pill(k === s.nnMode), onClick: () => c.setState({ nnMode: k }) })),
      nnSliderName: balloonMode ? 'K (points to collect)' : 'window radius h',
      nnSliderLabel: balloonMode ? String(s.nnK) : f2(s.nnR),
      nnMin: balloonMode ? 1 : 0.15, nnMax: balloonMode ? 25 : 1.5, nnStep: balloonMode ? 1 : 0.05, nnVal: balloonMode ? s.nnK : s.nnR,
      onNnSlider: e => c.setState(balloonMode ? { nnK: parseInt(e.target.value, 10) } : { nnR: parseFloat(e.target.value) }),
      nnBars: CLASS_NAMES.map((name, k) => ({
        label: name, sub: r.counts[k] + ' of ' + r.total + ' inside', fill: COLORS[k],
        pct: r.probs ? (r.probs[k] * 100).toFixed(0) : '0', pctLabel: r.probs ? (r.probs[k] * 100).toFixed(0) + '%' : '—'
      })),
      nnRadius: f2(r.r), nnInside: String(r.total), nnDensity: r.density.toFixed(3),
      nnWinner: decided ? CLASS_NAMES[r.probs[0] > r.probs[1] ? 0 : 1] : 'can’t decide',
      nnNote: nnNote(s, r)
    };
  }

  // ---------- §16.3.5: kernel regression ----------

  function regFit(s, x) {
    const D = M(), { xs, ys } = s.rgData;
    return s.rgMode === 'nw' ? D.nadarayaWatson(xs, ys, x, s.rgH, RG_SY) : D.localLine(xs, ys, x, s.rgH);
  }

  function regNote(s) {
    const edge = s.rgQ < RG_EDGE || s.rgQ > RG_HI - RG_EDGE, nw = s.rgMode === 'nw';
    if (edge && nw) return 'Edge problem: near the end of the data, every neighbour is on one side. The weighted average gets pulled toward them, so the prediction misses the dashed true curve. Try “Local line”.';
    if (edge) return 'The local line follows the trend of the neighbours on one side and extends it to the edge, so it stays much closer to the true curve than the plain average does.';
    if (s.rgH < 0.15) return 'Tiny h: only the one or two closest points get any weight, so the curve hops from dot to dot and chases the noise.';
    if (s.rgH > 1.5) return 'Huge h: every point gets nearly the same weight, so the prediction flattens toward the average of all the y values.';
    return nw
      ? 'The prediction is a weighted average of the dots’ heights. Bars under the plot show each dot’s weight (all the weights add up to 100%). The shaded band shows how much the weighted neighbours disagree.'
      : 'At each x, fit a straight line through the nearby dots (closer dots count more) and read the line at x. The short dark segment is the line fitted for the current query.';
  }

  function regression(c) {
    const s = c.state, D = M(), { xs, ys } = s.rgData, fr = frame(RG_LO, RG_HI, RG_YLO, RG_YHI), nw = s.rgMode === 'nw';
    const here = regFit(s, s.rgQ), wMax = Math.max(...here.w);
    const steps = 160, fits = range(0, steps, 1).map(i => { const x = RG_LO + ((RG_HI - RG_LO) * i) / steps; return [x, regFit(s, x)]; });
    const edge = sign => fits.map(([x, f]) => f1(fr.X(x)) + ' ' + f1(fr.Y(f.mean + sign * 2 * f.sd)));
    const half = 1.2 * s.rgH, a = Math.max(RG_LO, s.rgQ - half), b = Math.min(RG_HI, s.rgQ + half);
    const onLine = x => fr.Y(here.mean + here.slope * (x - s.rgQ));
    const drag = e => {
      if (e.type === 'pointerdown') capture(e);
      else if (!e.buttons) return;
      c.setState({ rgQ: clamp(fr.inv(svgX(e, W)), RG_LO, RG_HI) });
    };
    const truth = D.trend(s.rgQ);
    return {
      rgModeBtns: [['nw', 'Weighted average (Nadaraya–Watson)'], ['line', 'Local line (locally weighted regression)']].map(([k, label]) => ({ label, ...pill(k === s.rgMode), onClick: () => c.setState({ rgMode: k }) })),
      rgTruth: curvePath(fr, D.trend, 300),
      rgCurve: fits.map(([x, f], i) => (i ? 'L ' : 'M ') + f1(fr.X(x)) + ' ' + f1(fr.Y(f.mean))).join(' '),
      rgBand: nw ? 'M ' + edge(1).join(' L ') + ' L ' + edge(-1).reverse().join(' L ') + ' Z' : '',
      rgLocal: nw ? [] : [{ x1: f1(fr.X(a)), y1: f1(onLine(a)), x2: f1(fr.X(b)), y2: f1(onLine(b)) }],
      rgDots: xs.map((x, n) => ({ x: f1(fr.X(x)), y: f1(fr.Y(ys[n])), r: f1(3 + 4 * (here.w[n] / wMax)), op: (0.3 + 0.7 * (here.w[n] / wMax)).toFixed(2) })),
      rgBars: xs.map((x, n) => { const h = 46 * (here.w[n] / wMax); return { x: f1(fr.X(x) - 3), y: f1(54 - h), h: f1(h) }; }),
      rgQx: f1(fr.X(s.rgQ)), rgPy: f1(fr.Y(here.mean)), rgTop: PT, rgBase: H - PB,
      rgTicks: ticks(fr, range(0, 6, 1)),
      onRgDown: drag, onRgMove: drag,
      rgH: s.rgH, rgHLabel: f2(s.rgH), onRgH: e => c.setState({ rgH: parseFloat(e.target.value) }),
      onRgNew: () => c.setState({ rgData: D.regData(D.rng(Math.floor(Math.random() * 1e6))) }),
      rgPred: f2(here.mean), rgTrue: f2(truth), rgErr: f2(Math.abs(here.mean - truth)),
      rgSd: nw ? '± ' + f2(2 * here.sd) : '—', rgTopW: (wMax * 100).toFixed(0) + '%',
      rgNote: regNote(s)
    };
  }

  function render(c) {
    return { ...kernels(c), ...parzen(c), ...bandwidth(c), ...knn(c), ...regression(c) };
  }

  return { initState, render };
})();
