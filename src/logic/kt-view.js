// View logic for Topic 13 (the kernel trick, §17.3.4).
// Loaded only by kernel-trick.html. reading-log.js calls KTView.initState()
// for the starting state and KTView.render(component) from t13(); the math is in kt-math.js.
window.KTView = (function () {
  const M = () => window.KTMath;
  const POS = 'var(--color-accent)', NEG = 'var(--color-accent-2-600)';
  const pill = active => ({ bg: active ? 'var(--color-accent)' : 'transparent', fg: active ? 'var(--color-bg)' : 'var(--color-text)' });
  const patch = (c, key, change) => c.setState({ [key]: { ...c.state[key], ...change } });
  const num = e => parseFloat(e.target.value);
  const int = e => parseInt(e.target.value, 10);
  const f1 = v => v.toFixed(1);
  const f2 = v => v.toFixed(2);
  const big = v => v.toLocaleString('en-US');

  // ---------- part 1: lift 1-D data onto a parabola ----------

  const LIFT_W = 560, LIFT_H = 240, LIFT_PL = 34, LIFT_PR = 14, LIFT_PT = 14, LIFT_PB = 30;
  const LIFT_X = 2.6, LIFT_V_LO = -0.6, LIFT_V_HI = 7;
  // + in the middle, − on both sides: no single cut on the line separates them
  const LIFT_PTS = [[-0.85, 1], [-0.5, 1], [-0.15, 1], [0.25, 1], [0.6, 1], [0.95, 1],
    [-2.4, -1], [-1.95, -1], [-1.5, -1], [1.4, -1], [1.85, -1], [2.35, -1]];

  const lx = x => LIFT_PL + ((x + LIFT_X) / (2 * LIFT_X)) * (LIFT_W - LIFT_PL - LIFT_PR);
  const ly = v => LIFT_H - LIFT_PB - ((v - LIFT_V_LO) / (LIFT_V_HI - LIFT_V_LO)) * (LIFT_H - LIFT_PT - LIFT_PB);

  // best of "right of c is +" and "left of c is +"
  function flatCut(c) {
    const wrong = side => LIFT_PTS.filter(([x, l]) => (side * (x - c) > 0 ? 1 : -1) !== l).length;
    const right = wrong(1), left = wrong(-1);
    return right <= left ? { side: 1, errors: right } : { side: -1, errors: left };
  }

  function liftNote(s, errors) {
    if (!s.lifted) return 'One straight cut on a line splits it into a left half and a right half. The orange dots sit in the middle, so whichever way you cut, some dots land on the wrong side. Here: ' + errors + ' wrong.';
    if (errors) return 'Each dot x has been raised to height x². The orange dots are near 0, so they stay low; the green ones are far out, so they rise high. Slide the flat line into the gap between them.';
    return 'Zero mistakes with one flat line. Look at the baseline: that one line in the lifted picture is two cuts back on the original line, at x = ±√t. A straight boundary in the new space became a curved (here: two-piece) one in the old space.';
  }

  function lift(c) {
    const s = c.state.ktLift, flat = flatCut(s.cut);
    const inside = ([x]) => (s.lifted ? x * x < s.t : flat.side * (x - s.cut) > 0);
    const errors = s.lifted ? LIFT_PTS.filter(p => (inside(p) ? 1 : -1) !== p[1]).length : flat.errors;
    const xs = Array.from({ length: 81 }, (_, i) => -LIFT_X + (2 * LIFT_X * i) / 80);
    const root = Math.sqrt(Math.max(0, s.t));
    const shadeX = s.lifted ? [-root, root] : flat.side > 0 ? [s.cut, LIFT_X] : [-LIFT_X, s.cut];
    const shadeTop = s.lifted ? ly(Math.min(s.t, LIFT_V_HI)) : LIFT_PT;
    return {
      liftDots: LIFT_PTS.map(p => ({
        x: f1(lx(p[0])), y: f1(ly(s.lifted ? p[0] * p[0] : 0)),
        fill: p[1] > 0 ? POS : NEG, ring: (inside(p) ? 1 : -1) !== p[1] ? 'var(--color-text)' : 'transparent'
      })),
      liftStems: s.lifted ? LIFT_PTS.map(([x]) => ({ x: f1(lx(x)), y1: f1(ly(0)), y2: f1(ly(x * x)) })) : [],
      liftParabola: s.lifted ? xs.map((x, i) => (i ? 'L ' : 'M ') + f1(lx(x)) + ' ' + f1(ly(x * x))).join(' ') : '',
      liftBase: f1(ly(0)), liftTop: LIFT_PT, liftBottom: LIFT_H - LIFT_PB,
      liftShade: 'M ' + f1(lx(shadeX[0])) + ' ' + f1(shadeTop) + ' H ' + f1(lx(shadeX[1])) + ' V ' + f1(ly(LIFT_V_LO)) + ' H ' + f1(lx(shadeX[0])) + ' Z',
      liftShowFlat: !s.lifted, liftShowLifted: s.lifted,
      liftCutX: f1(lx(s.cut)), liftLineY: f1(ly(Math.min(s.t, LIFT_V_HI))),
      liftRootL: f1(lx(-root)), liftRootR: f1(lx(root)),
      liftTicks: [-2, -1, 0, 1, 2].map(v => ({ x: f1(lx(v)), label: String(v) })),
      liftModeBtns: [[false, 'Straight cut on the line'], [true, 'Lift to (x, x²), then cut flat']].map(([v, label]) => ({ label, ...pill(v === s.lifted), onClick: () => patch(c, 'ktLift', { lifted: v }) })),
      liftCut: s.cut, liftCutLabel: f2(s.cut), liftOnCut: e => patch(c, 'ktLift', { cut: num(e) }),
      liftT: s.t, liftTLabel: f2(s.t), liftOnT: e => patch(c, 'ktLift', { t: num(e) }),
      liftRootLabel: '±' + f2(root),
      liftErrors: String(errors), liftNote: liftNote(s, errors)
    };
  }

  // ---------- part 2: the same number, the long way and the short way ----------

  const DIMS = [2, 10, 100, 1000, 10000];

  function trick(c) {
    const s = c.state.ktTrick, D = M(), a = [s.a1, s.a2], b = [s.b1, s.b2];
    const pa = D.lift2(a), pb = D.lift2(b), ab = D.dot(a, b);
    const vec = v => '(' + v.map(f2).join(', ') + ')';
    const dim = DIMS[s.dim], longCost = D.featureCount(dim, s.deg), shortCost = dim + s.deg - 1;
    const slider = k => ({ ['tr' + k.toUpperCase()]: s[k], ['tr' + k.toUpperCase() + 'Label']: f1(s[k]), ['trOn' + k.toUpperCase()]: e => patch(c, 'ktTrick', { [k]: num(e) }) });
    return {
      ...slider('a1'), ...slider('a2'), ...slider('b1'), ...slider('b2'),
      trPhiA: vec(pa), trPhiB: vec(pb),
      trProducts: pa.map((v, i) => f2(v) + ' × ' + f2(pb[i])).join('  +  '),
      trLong: f2(D.dot(pa, pb)),
      trDot: f2(a[0]) + ' × ' + f2(b[0]) + ' + ' + f2(a[1]) + ' × ' + f2(b[1]) + ' = ' + f2(ab),
      trShort: f2(ab * ab),
      trDim: s.dim, trDimLabel: big(dim), trOnDim: e => patch(c, 'ktTrick', { dim: int(e) }),
      trDeg: s.deg, trDegLabel: String(s.deg), trOnDeg: e => patch(c, 'ktTrick', { deg: int(e) }),
      trLongCost: big(longCost), trShortCost: big(shortCost),
      trRatio: longCost >= 10 * shortCost ? '≈ ' + big(Math.round(longCost / shortCost)) + '× less work' : 'about the same work'
    };
  }

  // ---------- part 3: a kernel SVM you can poke ----------

  const LAB = 300, LAB_LIM = 3, CELLS = 50, GRAB = 10, LAB_MAX = 80, SMO_SEED = 473, DATA_SEED = 13, RBF_WIGGLY = 3;
  const C_VALUES = [0.1, 0.3, 1, 3, 10, 30, 100];
  const KERNELS = { linear: 'Linear: a·b', poly: 'Polynomial: (a·b)²', rbf: 'RBF: exp(−γ‖a − b‖²)' };
  const DATASETS = { rings: 'Rings', xor: 'XOR', blobs: 'Two blobs' };
  // decision value buckets: confident −, inside margin −, inside margin +, confident +
  const BUCKETS = [[-Infinity, -1, NEG, 0.34], [-1, 0, NEG, 0.13], [0, 1, POS, 0.13], [1, Infinity, POS, 0.34]];

  const px = v => ((v + LAB_LIM) / (2 * LAB_LIM)) * LAB;
  const py = v => LAB - ((v + LAB_LIM) / (2 * LAB_LIM)) * LAB;
  const unpx = p => (p / LAB) * 2 * LAB_LIM - LAB_LIM;
  const unpy = p => LAB_LIM - (p / LAB) * 2 * LAB_LIM;

  // retraining is cheap but not free; only redo it when the inputs change
  let cache = { key: null, model: null };
  function model(p, pts, C) {
    const key = JSON.stringify([p, pts, C]);
    if (cache.key !== key) cache = { key, model: M().trainSvm(p, pts, C, SMO_SEED) };
    return cache.model;
  }

  // one path per bucket instead of 2,500 rects
  function heat(p, m) {
    const size = LAB / CELLS, step = f1(size + 0.4), paths = BUCKETS.map(() => []);
    for (let i = 0; i < CELLS; i++) {
      for (let j = 0; j < CELLS; j++) {
        const x0 = i * size, y0 = j * size;
        const f = M().decision(p, m, [unpx(x0 + size / 2), unpy(y0 + size / 2)]);
        const k = BUCKETS.findIndex(([lo, hi]) => f >= lo && f < hi);
        paths[k].push('M' + f1(x0) + ' ' + f1(y0) + 'h' + step + 'v' + step + 'h-' + step + 'z');
      }
    }
    return BUCKETS.map(([, , color, opacity], k) => ({ d: paths[k].join(''), color, opacity }));
  }

  function labNote(s, nPos, nNeg, acc) {
    if (!nPos || !nNeg) return 'An SVM needs at least one dot of each colour. Switch the brush and add some.';
    if (s.kernel === 'linear' && acc < 0.95) return 'The linear kernel is the plain dot product: no lift at all, so the boundary must be a straight line. No straight line splits this data. Try another kernel.';
    if (s.kernel === 'linear') return 'A straight line is enough here, so the plain dot product works. You only need the kernel trick when the classes are tangled.';
    if (s.kernel === 'poly' && s.data === 'rings') return '(a·b)² is secretly the dot product of (x₁², √2 x₁x₂, x₂²). A circle x₁² + x₂² = r² is a flat cut in that space, so a curved boundary comes out for free.';
    if (s.kernel === 'poly' && s.data === 'xor') return 'XOR is just "is x₁·x₂ positive?", and x₁x₂ is one of the hidden features of (a·b)². So one flat cut in the lifted space solves it.';
    if (s.kernel === 'poly') return '(a·b)² can only draw boundaries built from squares and cross terms (ellipses, hyperbolas, pairs of lines through the centre). If the true boundary is wigglier than that, switch to RBF.';
    if (s.gamma > RBF_WIGGLY) return 'Large γ: each support vector\'s bump of similarity is very narrow, so the boundary hugs single dots and almost every dot becomes a support vector. That is overfitting.';
    return 'RBF: K(a, b) is near 1 when a and b are close and near 0 when far apart. Each support vector adds a bump (orange pushes +, green pushes −), and the boundary is where they balance. Its hidden φ has infinitely many features, yet each K is one exp().';
  }

  function labClick(c, s, e) {
    const r = e.currentTarget.getBoundingClientRect();
    const cx = ((e.clientX - r.left) / r.width) * LAB, cy = ((e.clientY - r.top) / r.height) * LAB;
    const near = s.pts.findIndex(q => Math.hypot(px(q[0]) - cx, py(q[1]) - cy) < GRAB);
    if (near >= 0) { patch(c, 'ktLab', { pts: s.pts.filter((_, i) => i !== near) }); return; }
    if (s.pts.length >= LAB_MAX) return;
    patch(c, 'ktLab', { pts: [...s.pts, [unpx(cx), unpy(cy), s.brush]] });
  }

  function svm(c) {
    const s = c.state.ktLab, D = M(), p = { type: s.kernel, gamma: s.gamma }, C = C_VALUES[s.cIdx];
    const m = model(p, s.pts, C);
    const correct = s.pts.filter(q => (D.decision(p, m, [q[0], q[1]]) >= 0 ? 1 : -1) === q[2]).length;
    const nPos = s.pts.filter(q => q[2] > 0).length, nNeg = s.pts.length - nPos;
    const acc = s.pts.length ? correct / s.pts.length : 0;
    return {
      labHeat: heat(p, m),
      labDots: s.pts.map(q => ({ x: f1(px(q[0])), y: f1(py(q[1])), fill: q[2] > 0 ? POS : NEG })),
      labSv: m.sv.map(n => ({ x: f1(px(m.X[n][0])), y: f1(py(m.X[n][1])) })),
      labOnDown: e => labClick(c, s, e),
      labDataBtns: Object.keys(DATASETS).map(k => ({ label: DATASETS[k], ...pill(k === s.data), onClick: () => patch(c, 'ktLab', { data: k, pts: D.dataset(k, DATA_SEED) }) })),
      labKernelBtns: Object.keys(KERNELS).map(k => ({ label: KERNELS[k], ...pill(k === s.kernel), onClick: () => patch(c, 'ktLab', { kernel: k }) })),
      labBrushBtns: [[1, '+ orange'], [-1, '− green']].map(([v, label]) => ({ label: 'Add ' + label, ...pill(v === s.brush), onClick: () => patch(c, 'ktLab', { brush: v }) })),
      labOnClear: () => patch(c, 'ktLab', { pts: [] }),
      labShowGamma: s.kernel === 'rbf',
      labGamma: s.gamma, labGammaLabel: f2(s.gamma), labOnGamma: e => patch(c, 'ktLab', { gamma: num(e) }),
      labC: s.cIdx, labCLabel: String(C), labOnC: e => patch(c, 'ktLab', { cIdx: int(e) }),
      labCount: s.pts.length, labSvCount: String(m.sv.length),
      labAcc: s.pts.length ? Math.round(acc * 100) + '%' : '—',
      labW0: f2(m.b),
      labNote: labNote(s, nPos, nNeg, acc)
    };
  }

  // ---------- state ----------

  function initState() {
    return {
      ktLift: { lifted: false, cut: 0, t: 0.5 },
      ktTrick: { a1: 1, a2: 2, b1: 3, b2: -1, dim: 0, deg: 2 },
      ktLab: { data: 'rings', pts: M().dataset('rings', DATA_SEED), kernel: 'linear', gamma: 0.8, cIdx: 4, brush: 1 }
    };
  }

  function render(c) {
    return { ...lift(c), ...trick(c), ...svm(c) };
  }

  return { initState, render };
})();
