// View logic for Topic 10 (deep metric learning, §16.2.2 – 16.2.4).
// Loaded only by metric-learning.html. reading-log.js calls DMLView.initState()
// for the starting state and DMLView.render(component) from t10(); the math is in dml-math.js.
window.DMLView = (function () {
  const M = () => window.DMLMath;
  const COLORS = ['#c67139', '#7a8a5e', '#4f6d8a'];
  const LOSSES = {
    triplet: { label: 'Triplet loss · §16.2.4', note: 'Each step grabs 60 (anchor, positive, negative) triplets and asks: is the positive closer than the negative by at least the margin? Only triplets that fail get a push.' },
    contrastive: { label: 'Contrastive loss · §16.2.4', note: 'Each step grabs 60 pairs, half same-class and half different. Same-class pairs are pulled together; different-class pairs are pushed apart only while they are closer than the margin.' },
    proxy: { label: 'Classifier · §16.2.3', note: 'No pairs at all: a 3-way softmax classifier sits on the normalized embedding. Its weight vectors (arrows) act as one "proxy" point per class, and every point is pulled toward its own proxy.' }
  };
  const N_PER = 40, KNN = 5, STEPS_PER_TICK = 10, TICK_MS = 40, MAX_STEPS = 2000;
  const HALF = 150, IN_SCALE = 118, EMB_R = 112;
  const PX_THETAS = [90, 210, 330], PX_CONFIDENT = 0.9;
  const GEO_C = 160, GEO_S = 50, GEO_GRAB = 0.7, ARROW_SCALE = 0.22, ARROW_MAX = 1.6;
  const NP_SCALE = 4, NP_MAX = 6, NP_R = 112, NP_GRAB_DEG = 25;
  const RAD = Math.PI / 180;

  const f1 = v => v.toFixed(1);
  const pill = active => ({ bg: active ? 'var(--color-accent)' : 'transparent', fg: active ? 'var(--color-bg)' : 'var(--color-text)' });
  const capture = e => { try { e.currentTarget.setPointerCapture(e.pointerId); } catch (err) { /* capture is optional */ } };

  // ---------- state ----------

  function freshLab(kind, seed) {
    const D = M();
    return { dmlKind: kind, dmlSeed: seed, dmlModel: D.initModel(D.rng(seed)), dmlRand: D.rng(seed + 1), dmlStep: 0, dmlHist: [] };
  }

  function freshGeo() {
    return { triPts: { a: [-1.2, 0.2], p: [0.6, 1.3], n: [0.5, -0.7] }, conPts: { i: [-1.1, -0.2], j: [0.4, 0.5] } };
  }

  function initState() {
    const D = M(), data = D.spiral(N_PER, D.rng(473));
    return {
      dmlData: data, dmlBase: D.pairStats(data.X, data.y), dmlSel: 25, dmlRun: false, ...freshLab('triplet', 11),
      pxPhi: 45, pxCls: 0, pxM: 0, pxS: 8,
      geoMode: 'triplet', ...freshGeo(), conSame: false, conM: 2, triM: 1.5, geoDrag: null,
      npN: 3, npPos: 60, npNegs: [110, 200, 290, 160, 250, 330], npDrag: null
    };
  }

  // ---------- 16.2.2 lab: training loop ----------

  function stop(c) {
    clearInterval(c.dmlTimer);
    if (c.state.dmlRun) c.setState({ dmlRun: false });
  }

  function tick(c) {
    const s = c.state, D = M();
    let model = s.dmlModel, sum = 0;
    for (let k = 0; k < STEPS_PER_TICK; k++) {
      const r = D.step(model, s.dmlData, s.dmlKind, s.dmlRand);
      model = r.model;
      sum += r.loss;
    }
    const step = s.dmlStep + STEPS_PER_TICK;
    c.setState({ dmlModel: model, dmlStep: step, dmlHist: [...s.dmlHist, sum / STEPS_PER_TICK] });
    if (step >= MAX_STEPS) stop(c);
  }

  function toggleRun(c) {
    if (c.state.dmlRun) return stop(c);
    if (c.state.dmlStep >= MAX_STEPS) c.setState(freshLab(c.state.dmlKind, c.state.dmlSeed));
    c.setState({ dmlRun: true });
    c.dmlTimer = setInterval(() => tick(c), TICK_MS);
  }

  function restart(c, kind, seed) {
    stop(c);
    c.setState(freshLab(kind, seed));
  }

  function histBars(st, y0, h) {
    const bw = 280 / st.hSame.length, top = Math.max(...st.hSame, ...st.hDiff) || 1;
    const bars = (arr, fill) => arr.map((v, b) => {
      const bh = (v / top) * h;
      return { x: f1(10 + b * bw), y: f1(y0 + h - bh), w: f1(bw - 1), h: f1(bh), fill };
    });
    return [...bars(st.hDiff, 'var(--color-accent-2-500)'), ...bars(st.hSame, 'var(--color-accent)')];
  }

  function lab(c) {
    const s = c.state, D = M(), data = s.dmlData;
    const E = D.embed(s.dmlModel.P, data.X), st = D.pairStats(E, data.y, 20, 2), base = s.dmlBase;
    const inXY = p => [HALF + p[0] * IN_SCALE, HALF - p[1] * IN_SCALE];
    const emXY = e => [HALF + e[0] * EMB_R, HALF - e[1] * EMB_R];
    const dots = (pts, xy) => pts.map((p, i) => {
      const [x, y] = xy(p);
      return { x: f1(x), y: f1(y), fill: COLORS[data.y[i]], onClick: () => c.setState({ dmlSel: i }) };
    });
    const sel = s.dmlSel, cls = data.y[sel];
    const links = (pts, xy) => {
      const [sx, sy] = xy(pts[sel]);
      return D.nearest(pts, sel, KNN).map(j => {
        const [x, y] = xy(pts[j]), ok = data.y[j] === cls;
        return { x1: f1(sx), y1: f1(sy), x2: f1(x), y2: f1(y), dash: ok ? 'none' : '4 3', stroke: ok ? 'var(--color-neutral-800)' : 'var(--color-accent-700)' };
      });
    };
    const agree = pts => D.nearest(pts, sel, KNN).filter(j => data.y[j] === cls).length;
    const ring = (pts, xy) => { const [x, y] = xy(pts[sel]); return { x: f1(x), y: f1(y) }; };

    const hist = s.dmlHist, hMax = Math.max(0.05, ...hist), ticks = MAX_STEPS / STEPS_PER_TICK - 1;
    const spark = hist.map((v, i) => (i ? 'L ' : 'M ') + f1(10 + (i / ticks) * 280) + ' ' + f1(46 - (v / hMax) * 38)).join(' ');
    const proxyArrows = s.dmlKind === 'proxy'
      ? D.proxies(s.dmlModel.P).map((p, k) => ({ x2: f1(HALF + p.w[0] * (EMB_R + 22)), y2: f1(HALF - p.w[1] * (EMB_R + 22)), stroke: COLORS[k] }))
      : [];

    let note = 'Untrained network: the embedding is a random squish of the input, so the three colours are smeared around the circle. Press Train.';
    if (s.dmlStep > 0 && st.auc < 0.9) note = 'Training… watch the colours pull into arcs on the circle while the input picture never changes. The network is learning a new ruler, not moving the data.';
    if (s.dmlStep > 0 && st.auc >= 0.9) note = 'Untangled. In the input, a different-class point is often closer than a same-class one (the spiral arms are interleaved). In the embedding, same class ⇒ close and different class ⇒ far, almost without exception.';

    return {
      dmlLossBtns: Object.keys(LOSSES).map(k => ({ label: LOSSES[k].label, ...pill(k === s.dmlKind), onClick: () => restart(c, k, s.dmlSeed) })),
      dmlLossNote: LOSSES[s.dmlKind].note,
      dmlRunLabel: s.dmlRun ? 'Pause' : s.dmlStep >= MAX_STEPS ? 'Train again' : s.dmlStep ? 'Resume' : 'Train',
      onDmlRun: () => toggleRun(c),
      onDmlReset: () => restart(c, s.dmlKind, s.dmlSeed + 1),
      dmlStepLabel: s.dmlStep + ' / ' + MAX_STEPS,
      dmlLossLabel: hist.length ? hist[hist.length - 1].toFixed(3) : '–',
      dmlSpark: spark,
      dmlInDots: dots(data.X, inXY), dmlEmDots: dots(E, emXY),
      dmlInLinks: links(data.X, inXY), dmlEmLinks: links(E, emXY),
      dmlInRing: ring(data.X, inXY), dmlEmRing: ring(E, emXY),
      dmlInAgree: agree(data.X) + ' of ' + KNN, dmlEmAgree: agree(E) + ' of ' + KNN,
      dmlProxyArrows: proxyArrows,
      dmlInAuc: (base.auc * 100).toFixed(0) + '%', dmlEmAuc: (st.auc * 100).toFixed(0) + '%',
      dmlInHist: histBars(base, 6, 46), dmlEmHist: histBars(st, 6, 46),
      dmlNote: note
    };
  }

  // ---------- 16.2.3: normalized softmax with proxies on the circle ----------

  function arcPath(r, a0, a1) {
    const pt = a => f1(HALF + r * Math.cos(a * RAD)) + ' ' + f1(HALF - r * Math.sin(a * RAD));
    return 'M ' + pt(a0) + ' A ' + r + ' ' + r + ' 0 ' + (a1 - a0 > 180 ? 1 : 0) + ' 0 ' + pt(a1);
  }

  // whole-degree stretches of the circle where p(true class) > 90%
  function confidentArcs(s) {
    const D = M(), th = PX_THETAS.map(t => t * RAD), arcs = [];
    let start = null;
    for (let a = 0; a <= 360; a++) {
      const ok = a < 360 && D.cosSoftmax(a * RAD, th, s.pxCls, s.pxS, s.pxM).probs[s.pxCls] > PX_CONFIDENT;
      if (ok && start === null) start = a;
      if (!ok && start !== null) { arcs.push([start, a - 1]); start = null; }
    }
    return arcs;
  }

  function angleFromEvent(e) {
    const r = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * 300 - HALF, y = HALF - ((e.clientY - r.top) / r.height) * 300;
    return ((Math.atan2(y, x) / RAD) + 360) % 360;
  }

  function proxyLab(c) {
    const s = c.state, D = M(), th = PX_THETAS.map(t => t * RAD);
    const r = D.cosSoftmax(s.pxPhi * RAD, th, s.pxCls, s.pxS, s.pxM);
    const arcs = confidentArcs(s), width = arcs.reduce((a, [a0, a1]) => a + a1 - a0 + 1, 0);
    const ex = HALF + EMB_R * Math.cos(s.pxPhi * RAD), ey = HALF - EMB_R * Math.sin(s.pxPhi * RAD);
    const names = ['A', 'B', 'C'];
    const moveTo = e => {
      if (e.type === 'pointerdown') capture(e);
      else if (!e.buttons) return;
      c.setState({ pxPhi: angleFromEvent(e) });
    };

    let note = 'Drag the dot around the circle. Its class probabilities depend only on the angle to each proxy, because both the embedding and the proxies have length 1.';
    if (s.pxM > 0) note = 'With a margin, the true class must beat the others by m in cosine before it gets the same probability. The confident zone shrinks, so training has to pack each class into a tighter arc around its proxy: tighter clusters, bigger gaps between classes.';

    return {
      pxProxies: PX_THETAS.map((t, k) => ({
        x2: f1(HALF + (EMB_R + 20) * Math.cos(t * RAD)), y2: f1(HALF - (EMB_R + 20) * Math.sin(t * RAD)),
        lx: f1(HALF + (EMB_R + 34) * Math.cos(t * RAD)), ly: f1(HALF - (EMB_R + 34) * Math.sin(t * RAD) + 4),
        stroke: COLORS[k], label: 'w' + names[k], width: k === s.pxCls ? 4 : 2.5
      })),
      pxArcs: arcs.map(([a0, a1]) => ({ d: arcPath(EMB_R, a0, a1) })),
      pxX: f1(ex), pxY: f1(ey), pxFill: COLORS[s.pxCls],
      onPxDown: moveTo, onPxMove: moveTo,
      pxClsBtns: names.map((n, k) => ({ label: 'True class ' + n, ...pill(k === s.pxCls), onClick: () => c.setState({ pxCls: k }) })),
      pxM: s.pxM, pxS: s.pxS, pxMLabel: s.pxM.toFixed(2), pxSLabel: s.pxS.toFixed(0),
      onPxM: e => c.setState({ pxM: parseFloat(e.target.value) }),
      onPxS: e => c.setState({ pxS: parseFloat(e.target.value) }),
      pxRows: names.map((n, k) => ({
        name: n, cos: r.cos[k].toFixed(2), logit: r.logits[k].toFixed(2), pct: (r.probs[k] * 100).toFixed(1),
        fill: COLORS[k], weight: k === s.pxCls ? 700 : 400
      })),
      pxLoss: r.loss.toFixed(3), pxWidth: width + '°', pxClsName: names[s.pxCls],
      pxNote: note
    };
  }

  // ---------- 16.2.4: contrastive and triplet geometry ----------

  const toGeo = p => [GEO_C + p[0] * GEO_S, GEO_C - p[1] * GEO_S];

  function geoWorld(e) {
    const r = e.currentTarget.getBoundingClientRect();
    const x = ((e.clientX - r.left) / r.width) * 320, y = ((e.clientY - r.top) / r.height) * 320;
    const clamp = v => Math.max(-3, Math.min(3, v));
    return [clamp((x - GEO_C) / GEO_S), clamp((GEO_C - y) / GEO_S)];
  }

  // descent direction −g as an arrow with a small head
  function arrow(p, g, stroke) {
    const vx = -g[0] * ARROW_SCALE, vy = -g[1] * ARROW_SCALE, len = Math.hypot(vx, vy);
    if (len < 0.04) return null;
    const k = Math.min(1, ARROW_MAX / len), [x1, y1] = toGeo(p), [x2, y2] = toGeo([p[0] + vx * k, p[1] + vy * k]);
    const ang = Math.atan2(y2 - y1, x2 - x1), hd = 8;
    const head = [[x2, y2], [x2 - hd * Math.cos(ang - 0.45), y2 - hd * Math.sin(ang - 0.45)], [x2 - hd * Math.cos(ang + 0.45), y2 - hd * Math.sin(ang + 0.45)]];
    return { x1: f1(x1), y1: f1(y1), x2: f1(x2), y2: f1(y2), head: head.map(([a, b]) => f1(a) + ',' + f1(b)).join(' '), stroke };
  }

  function line(u, v, stroke) {
    const [x1, y1] = toGeo(u), [x2, y2] = toGeo(v);
    return { x1: f1(x1), y1: f1(y1), x2: f1(x2), y2: f1(y2), stroke };
  }

  // annulus between radii r1 < r2, drawn with fill-rule evenodd
  function ringPath([cx, cy], r1, r2) {
    const circ = r => 'M ' + f1(cx - r) + ' ' + f1(cy) + ' a ' + f1(r) + ' ' + f1(r) + ' 0 1 0 ' + f1(2 * r) + ' 0 a ' + f1(r) + ' ' + f1(r) + ' 0 1 0 ' + f1(-2 * r) + ' 0 Z';
    return circ(r2) + ' ' + circ(r1);
  }

  // loss as a function of distance, for the inset under the plane
  function curve(fn, dNow) {
    const X = d => 30 + (d / 4) * 260, Y = l => 104 - (Math.min(l, 6) / 6) * 94;
    let d = '';
    for (let i = 0; i <= 80; i++) d += (i ? 'L ' : 'M ') + f1(X(i / 20)) + ' ' + f1(Y(fn(i / 20))) + ' ';
    const dn = Math.min(dNow, 4);
    return { path: d, mx: f1(X(dn)), my: f1(Y(fn(dn))) };
  }

  function geoPoints(s) {
    const tri = s.geoMode === 'triplet', pts = tri ? s.triPts : s.conPts;
    const look = tri
      ? { a: ['anchor', 'var(--color-neutral-800)'], p: ['positive', COLORS[1]], n: ['negative', COLORS[0]] }
      : { i: ['xᵢ', 'var(--color-neutral-800)'], j: ['xⱼ', s.conSame ? COLORS[1] : COLORS[0]] };
    return Object.keys(pts).map(k => {
      const [x, y] = toGeo(pts[k]);
      return { x: f1(x), y: f1(y), lx: f1(x + 11), ly: f1(y - 9), label: look[k][0], fill: look[k][1] };
    });
  }

  function geoTriplet(s) {
    const D = M(), { a, p, n } = s.triPts, r = D.triplet(a, p, n, s.triM), ca = toGeo(a);
    const r1 = r.dap * GEO_S, r2 = Math.sqrt(r.dap * r.dap + s.triM) * GEO_S;
    const kinds = {
      easy: 'Easy negative: already outside the dashed ring, so the loss is 0 and nothing moves. Most random triplets look like this late in training, which is why they waste compute.',
      'semi-hard': 'Semi-hard negative: farther than the positive, but inside the margin band. Loss > 0, so it still teaches something. These are the ones FaceNet-style mining hunts for.',
      hard: 'Hard negative: closer to the anchor than the positive is. Biggest loss and biggest push, but training only on these early on tends to collapse everything to one point.'
    };
    const fn = d => Math.max(0, r.dap * r.dap - d * d + s.triM);
    return {
      geoRing: ringPath(ca, r1, r2), geoShowRing: true, geoShowMargin: false,
      geoInner: { cx: f1(ca[0]), cy: f1(ca[1]), r: f1(r1) }, geoOuter: { cx: f1(ca[0]), cy: f1(ca[1]), r: f1(r2) },
      geoArrows: [arrow(a, r.ga, 'var(--color-neutral-800)'), arrow(p, r.gp, COLORS[1]), arrow(n, r.gn, COLORS[0])].filter(Boolean),
      geoLines: [line(a, p, COLORS[1]), line(a, n, COLORS[0])],
      geoKind: r.kind, geoKindBg: r.kind === 'easy' ? 'var(--color-accent-2-100)' : r.kind === 'hard' ? 'var(--color-accent-200)' : 'var(--color-accent-100)',
      geoLoss: r.loss.toFixed(2),
      geoFormula: '[ ' + r.dap.toFixed(2) + '² − ' + r.dan.toFixed(2) + '² + ' + s.triM.toFixed(1) + ' ]₊ = ' + r.loss.toFixed(2),
      geoNote: kinds[r.kind],
      geoCurve: curve(fn, r.dan), geoCurveLabel: 'loss vs d(anchor, negative), positive held fixed',
      geoM: s.triM, geoMMin: 0.2, geoMMax: 4, geoMName: 'margin m (in squared distance)', geoMLabel: s.triM.toFixed(1)
    };
  }

  function geoContrastive(s) {
    const D = M(), { i, j } = s.conPts, r = D.contrastive(i, j, s.conSame, s.conM), ci = toGeo(i);
    const fn = d => (s.conSame ? d * d : Math.max(0, s.conM - d) ** 2);
    let note = 'Same class: the loss is d², so the pair is always pulled together, harder the farther apart they are.';
    if (!s.conSame) note = r.loss > 0
      ? 'Different class and inside the margin (dashed circle): pushed apart until the gap reaches m. The closer they are, the harder the push.'
      : 'Different class and already farther than m: loss is exactly 0. The margin stops the loss from shoving every negative pair off to infinity.';
    return {
      geoRing: '', geoShowRing: false, geoShowMargin: !s.conSame,
      geoInner: { cx: f1(ci[0]), cy: f1(ci[1]), r: 0 }, geoOuter: { cx: f1(ci[0]), cy: f1(ci[1]), r: f1(s.conM * GEO_S) },
      geoArrows: [arrow(i, r.gi, 'var(--color-neutral-800)'), arrow(j, [-r.gi[0], -r.gi[1]], s.conSame ? COLORS[1] : COLORS[0])].filter(Boolean),
      geoLines: [line(i, j, 'var(--color-neutral-600)')],
      geoKind: s.conSame ? 'same class' : r.loss > 0 ? 'different · inside margin' : 'different · past margin',
      geoKindBg: s.conSame ? 'var(--color-accent-2-100)' : 'var(--color-accent-100)',
      geoLoss: r.loss.toFixed(2),
      geoFormula: s.conSame ? r.d.toFixed(2) + '² = ' + r.loss.toFixed(2) : '[ ' + s.conM.toFixed(1) + ' − ' + r.d.toFixed(2) + ' ]₊² = ' + r.loss.toFixed(2),
      geoNote: note,
      geoCurve: curve(fn, r.d), geoCurveLabel: 'loss vs d(xᵢ, xⱼ)',
      geoM: s.conM, geoMMin: 0.5, geoMMax: 3, geoMName: 'margin m', geoMLabel: s.conM.toFixed(1)
    };
  }

  function geo(c) {
    const s = c.state, tri = s.geoMode === 'triplet', key = tri ? 'triPts' : 'conPts';
    const down = e => {
      const w = geoWorld(e), pts = s[key];
      const best = Object.keys(pts).map(k => [k, Math.hypot(pts[k][0] - w[0], pts[k][1] - w[1])]).sort((u, v) => u[1] - v[1])[0];
      if (best[1] > GEO_GRAB) return;
      capture(e);
      c.setState({ geoDrag: best[0] });
    };
    const move = e => {
      const k = c.state.geoDrag;
      if (!k || !e.buttons) return;
      c.setState({ [key]: { ...c.state[key], [k]: geoWorld(e) } });
    };
    return {
      ...(tri ? geoTriplet(s) : geoContrastive(s)),
      geoModeBtns: [['triplet', 'Triplet'], ['contrastive', 'Contrastive (pairs)']].map(([k, label]) => ({ label, ...pill(k === s.geoMode), onClick: () => c.setState({ geoMode: k, geoDrag: null }) })),
      geoPts: geoPoints(s), geoIsCon: !tri,
      geoSameLabel: s.conSame ? 'Pair label: same class' : 'Pair label: different class',
      onGeoSame: () => c.setState({ conSame: !s.conSame }),
      onGeoM: e => c.setState(tri ? { triM: parseFloat(e.target.value) } : { conM: parseFloat(e.target.value) }),
      onGeoReset: () => c.setState({ ...freshGeo(), geoDrag: null }),
      onGeoDown: down, onGeoMove: move, onGeoUp: () => c.setState({ geoDrag: null })
    };
  }

  // ---------- 16.2.4: N-pairs ----------

  function nPairsLab(c) {
    const s = c.state, D = M(), negs = s.npNegs.slice(0, s.npN);
    const r = D.nPairs(0, s.npPos * RAD, negs.map(a => a * RAD), NP_SCALE);
    const at = (a, rr) => [HALF + rr * Math.cos(a * RAD), HALF - rr * Math.sin(a * RAD)];
    const dot = (a, fill, label) => { const [x, y] = at(a, NP_R), [lx, ly] = at(a, NP_R + 20); return { x: f1(x), y: f1(y), lx: f1(lx), ly: f1(ly + 4), fill, label }; };
    const pts = [dot(s.npPos, COLORS[1], '+'), ...negs.map((a, k) => dot(a, COLORS[0], String(k + 1)))];
    const [ax, ay] = at(0, NP_R);

    const grab = e => {
      const a = angleFromEvent(e), gap = v => Math.abs(((v - a + 540) % 360) - 180);
      const best = [s.npPos, ...negs].map((v, k) => [k, gap(v)]).sort((u, v) => u[1] - v[1])[0];
      if (best[1] > NP_GRAB_DEG) return;
      capture(e);
      c.setState({ npDrag: best[0] });
    };
    const move = e => {
      const k = c.state.npDrag;
      if (k === null || !e.buttons) return;
      const a = angleFromEvent(e);
      if (k === 0) return c.setState({ npPos: a });
      c.setState({ npNegs: c.state.npNegs.map((v, i) => (i === k - 1 ? a : v)) });
    };
    const negP = r.probs.slice(1), hardest = negP.indexOf(Math.max(...negP));

    return {
      npPts: pts, npAX: f1(ax), npAY: f1(ay),
      npSpokes: pts.map(p => ({ x2: p.x, y2: p.y, stroke: p.fill })),
      onNpDown: grab, onNpMove: move, onNpUp: () => c.setState({ npDrag: null }),
      npN: s.npN, npNMax: NP_MAX, onNpN: e => c.setState({ npN: parseInt(e.target.value, 10) }),
      onNpShuffle: () => c.setState({ npNegs: s.npNegs.map(() => 40 + Math.random() * 280) }),
      npBars: r.probs.map((p, k) => ({
        label: k ? 'negative ' + k : 'positive',
        sub: 's·cos = ' + (k ? r.sn[k - 1] : r.sp).toFixed(2),
        pct: (p * 100).toFixed(1), fill: k ? COLORS[0] : COLORS[1]
      })),
      npLoss: r.loss.toFixed(3), npSingle: r.single.toFixed(3), npCount: s.npN,
      npNote: s.npN === 1
        ? 'With one negative, N-pairs is log(1 + e^(s_neg − s_pos)): a smooth, margin-free triplet loss. Add more negatives with the slider.'
        : 'Each negative is pushed with weight equal to its softmax share, so the hardest one here (negative ' + (hardest + 1) + ') automatically gets the biggest push. A triplet loss would see only one negative per update.'
    };
  }

  function render(c) {
    return { ...lab(c), ...proxyLab(c), ...geo(c), ...nPairsLab(c) };
  }

  return { initState, render, stop };
})();
