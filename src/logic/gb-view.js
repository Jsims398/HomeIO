// View logic for Topic 17, part 1 (§18.5.5): step-by-step walkthroughs of gradient tree boosting on 1-D data
// and of XGBoost rounds on 2-D data. Loaded only by gradient-boosting.html. reading-log.js calls
// GBView.initState() and GBView.render(component) from t17(); both also merge GBIView (§18.6, gbi-view.js).
// The math is in gb-math.js; the round/step controls are in walkthrough.js.
window.GBView = (function () {
  const M = () => window.GBMath, W = () => window.Walkthrough;
  const SEED = 16, GB_ROUNDS = 40, XG_ROUNDS = 25, GB_DEPTH = 2, NU = 1, ETA = 0.5;
  const POS = 'var(--color-accent)', NEG = '#4f6d8a', FIT = 'var(--color-accent-2-600)';
  const SUBS = '₀₁₂₃₄₅₆₇₈₉', FEAT = ['x₁', 'x₂'];

  const num = e => parseFloat(e.target.value);
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const sub = n => String(n).split('').map(d => SUBS[+d]).join('');
  const f1 = v => v.toFixed(1);
  const fx = d => v => (v < 0 ? '−' : '') + Math.abs(v).toFixed(d);
  const f2 = fx(2), f3 = fx(3);
  const sg = v => (v < 0 ? '−' : '+') + Math.abs(v).toFixed(2);
  const pct = v => Math.round(v * 100) + '%';
  const argmax = arr => arr.reduce((b, v, i) => (v > arr[b] ? i : b), 0);
  const minMax = arr => [Math.min(...arr), Math.max(...arr)];
  const sum = arr => arr.reduce((s, v) => s + v, 0);
  const toggleBg = on => ({ bg: on ? 'var(--color-accent)' : 'var(--color-neutral-200)', fg: on ? 'var(--color-bg)' : 'var(--color-text)' });
  const interval = iv => '(' + f2(iv[0]) + ', ' + f2(iv[1]) + ']';

  // ---------- ① gradient tree boosting on 1-D data ----------

  const RW = 560, RPL = 36, RPR = 12, LIM = 10;
  const TOP_H = 240, TOP_PT = 12, TOP_PB = 24, YMAX = 7;
  const RES_H = 170, RES_PT = 14, RES_PB = 14, CURVE_STEP = 0.025;
  const lx = x => RPL + (x / LIM) * (RW - RPL - RPR);
  const ly = y => TOP_H - TOP_PB - (y / YMAX) * (TOP_H - TOP_PT - TOP_PB);
  // the residual panel's vertical scale: ±1.5 for absolute-loss pseudo-residuals (all ±1), ±3 otherwise
  const resScale = rmax => r => RES_PT + ((rmax - clamp(r, -rmax, rmax)) / (2 * rmax)) * (RES_H - RES_PT - RES_PB);
  const TICKS = { 1.5: [1, 0, -1], 3: [2, 0, -2] };

  function curvePath(fn) {
    const xs = Array.from({ length: Math.round(LIM / CURVE_STEP) + 1 }, (_, i) => i * CURVE_STEP);
    return xs.map((x, i) => (i ? 'L ' : 'M ') + f1(lx(x)) + ' ' + f1(ly(clamp(fn(x), 0, YMAX)))).join(' ');
  }

  const LOSS_TEXT = {
    squared: { name: 'squared loss', f0: 'the mean ȳ', r: 'r_i = y_i − f(x_i), the ordinary residual.', leaf: 'mean', metric: 'Mean ½(y − f)²' },
    absolute: { name: 'absolute loss', f0: 'the median of y', r: 'r_i = sign(y_i − f(x_i)): just +1 (up) or −1 (down).', leaf: 'median', metric: 'Mean |y − f|' }
  };

  const leafList = (tree, key) => M().leafBoxes(tree, 1).map(b => interval(b.box[0]) + ' → ' + sg(b.leaf[key])).join(', ');

  function gbPhases(run, st, s) {
    const T = LOSS_TEXT[s.loss];
    return [
      { title: 'Compute the pseudo-residuals', formula: 'r_im = −∂ℓ(y_i, f) / ∂f  at  f = f_{m−1}(x_i)',
        body: (s.k === 0 ? 'f₀ = ' + T.f0 + ' = ' + f2(run.f0) + '. ' : '') + 'For ' + T.name + ', ' + T.r },
      { title: 'Fit a regression tree to them', formula: 'F_m = argmin_F Σᵢ (r_im − F(x_i))²',
        body: 'Depth-' + GB_DEPTH + ' tree: ' + leafList(st.tree, 'fitV') + '.' },
      { title: 'Re-solve each leaf for the real loss', formula: 'γ_jm = argmin_γ Σ_{x_i ∈ R_jm} ℓ(y_i, f_{m−1}(x_i) + γ)',
        body: 'γ = ' + T.leaf + ' of the residuals y − f in each leaf: ' + leafList(st.tree, 'v') +
          (s.loss === 'squared' ? '. Same as step 2 for squared loss.' : '. Back in y units, not ±1.') },
      { title: 'Add it', formula: 'f_m(x) = f_{m−1}(x) + Σ_j γ_jm I(x ∈ R_jm)',
        body: T.metric + ': ' + f3(st.lossBefore) + ' → ' + f3(st.lossAfter) + '.' }
    ];
  }

  function gbInspect(data, st, i, s) {
    const d = data[i], lf = M().leafAt(st.tree, d.x);
    return W().upTo(s.p, [
      'Point #' + (i + 1) + ': x = ' + f1(d.x[0]) + ', y = ' + f1(d.y) + ', f = ' + f2(st.fBefore[i]) + ', so y − f = ' + sg(st.res[i]) + ' and r = ' + sg(st.r[i]) + '.',
      'Its leaf predicts F = ' + sg(lf.fitV) + '.',
      'The ' + LOSS_TEXT[s.loss].leaf + ' residual of its ' + lf.idx.length + ' leaf points is γ = ' + sg(lf.v) + '.',
      'New f = ' + f2(st.fBefore[i]) + ' + ' + sg(lf.v) + ' = ' + f2(st.fAfter[i]) + '.'
    ]);
  }

  const GB_NOTES = [
    'Gradient boosting is gradient descent in function space: −∂ℓ/∂f says which way each prediction f(x_i) should move. Only this step depends on the loss.',
    'The gradient only exists at the training points; fitting a tree to it turns it into a function of x. The tree is always fit by least squares.',
    'Gradient tree boosting (§18.5.5.1): keep the tree\'s regions, but pick each leaf\'s value by minimising the actual loss there.',
    'Repeat. In practice the step is often shrunk by a learning rate ν < 1, as in §18.5.2.'
  ];

  // the lower panel: pseudo-residuals (steps 1–2), real residuals (step 3), residuals after the update (step 4)
  function gbPanel(st, s, y) {
    const pseudo = s.p <= 1, rmax = pseudo && s.loss === 'absolute' ? 1.5 : 3, lr = resScale(rmax);
    const vals = s.p === 3 ? y.map((v, i) => v - st.fAfter[i]) : pseudo ? st.r : st.res;
    const boxes = M().leafBoxes(st.tree, 1), key = s.p === 1 ? 'fitV' : 'v';
    const titles = ['Pseudo-residuals r_i = −∂ℓ/∂f', 'Pseudo-residuals, with the tree F fit to them', 'Residuals y − f, with each leaf\'s γ', 'Residuals y − f after the update'];
    return {
      vals: vals.map(v => f1(lr(v))),
      gbTicks: TICKS[rmax].map(t => ({ y: f1(lr(t)), label: t > 0 ? '+' + t : t < 0 ? '−' + Math.abs(t) : '0' })),
      gbShowTree: s.p === 1 || s.p === 2,
      gbSegs: boxes.map(b => ({ x1: f1(lx(b.box[0][0])), x2: f1(lx(b.box[0][1])), y: f1(lr(b.leaf[key])), color: s.p === 1 ? FIT : POS })),
      gbResTitle: titles[s.p]
    };
  }

  function gbView(c) {
    const s = c.state.gb1, data = window.BoostMath.regData(SEED), y = data.map(d => d.y), xs = data.map(d => d.x[0]);
    const run = M().gradBoost(data, s.k + 1, s.loss, NU, GB_DEPTH), st = run.steps[s.k], after = s.p === 3;
    const fOf = n => x => run.steps.slice(0, n).reduce((f, q) => f + NU * M().predict(q.tree, [x]), run.f0);
    const sel = s.sel === null || s.sel >= data.length ? argmax(st.res.map(Math.abs)) : s.sel;
    const pick = i => () => W().patch(c, 'gb1', { sel: i });
    const ring = i => ({ stroke: i === sel ? 'var(--color-text)' : 'var(--color-surface)', sw: i === sel ? 2.6 : 1.2 });
    const { vals, ...panel } = gbPanel(st, s, y);
    const setLoss = loss => () => W().patch(c, 'gb1', { loss });
    const sq = toggleBg(s.loss === 'squared'), ab = toggleBg(s.loss === 'absolute');
    return {
      ...W().bindings(c, 'gb1', 'gb', gbPhases(run, st, s), GB_ROUNDS),
      ...panel,
      gbDots: xs.map((x, i) => ({ x: f1(lx(x)), y: f1(ly(y[i])), ...ring(i), onClick: pick(i) })),
      gbResDots: xs.map((x, i) => ({ x: f1(lx(x)), y: vals[i], ...ring(i), onClick: pick(i) })),
      gbOld: curvePath(fOf(s.k)), gbOldDash: after ? '6 5' : 'none', gbOldOpacity: after ? '.5' : '1',
      gbShowNew: after, gbNew: curvePath(fOf(s.k + 1)),
      gbInspect: gbInspect(data, st, sel, s),
      gbNote: GB_NOTES[s.p],
      gbLoss: f3(after ? st.lossAfter : st.lossBefore),
      gbLossLabel: LOSS_TEXT[s.loss].metric,
      gbSq: setLoss('squared'), gbSqBg: sq.bg, gbSqFg: sq.fg,
      gbAbs: setLoss('absolute'), gbAbsBg: ab.bg, gbAbsFg: ab.fg
    };
  }

  // ---------- ② XGBoost: one regularised, second-order tree per round ----------

  const BOX = 300;
  const px = v => (v / LIM) * BOX;
  const py = v => BOX - (v / LIM) * BOX;
  const rect = (x0, x1, y0, y1) => ({ x: f1(px(x0)), y: f1(py(y1)), w: f1(px(x1) - px(x0)), h: f1(py(y0) - py(y1)) });
  const cls = v => (v >= 0 ? POS : NEG);
  const rule = nd => FEAT[nd.j] + ' ≤ ' + f2(nd.t);
  const trainErr = (f, ys) => (ys.length ? ys.filter((v, i) => (f[i] >= 0 ? 1 : 0) !== v).length / ys.length : 0);
  const splitBox = (box, nd) => [box.map((iv, k) => (k === nd.j ? [iv[0], nd.t] : iv)), box.map((iv, k) => (k === nd.j ? [nd.t, iv[1]] : iv))];

  // the ensemble so far is constant on the grid cut by every split threshold of every tree
  function modelRects(trees) {
    if (!trees.length) return [];
    const cuts = j => [0, ...[...new Set(trees.flatMap(t => M().nodes(t).filter(n => !n.leaf && n.j === j).map(n => n.t)))].sort((a, b) => a - b), LIM];
    const bx = cuts(0), by = cuts(1), f = x => trees.reduce((acc, t) => acc + ETA * M().predict(t, x), 0);
    return bx.slice(1).flatMap((x1, i) => by.slice(1).map((y1, k) => {
      const v = f([(bx[i] + x1) / 2, (by[k] + y1) / 2]);
      return { ...rect(bx[i], x1, by[k], y1), fill: cls(v), opacity: (0.05 + 0.33 * Math.abs(2 * M().sigmoid(v) - 1)).toFixed(2) };
    }));
  }

  // the new tree's own leaves, tinted by the sign of w* and stronger for bigger |w*|
  const leafRects = tree => M().leafBoxes(tree, 2).map(b => ({
    ...rect(b.box[0][0], b.box[0][1], b.box[1][0], b.box[1][1]), fill: cls(b.leaf.v), opacity: (0.08 + 0.3 * Math.min(Math.abs(b.leaf.v) / 2, 1)).toFixed(2)
  }));

  // each split drawn only inside its own node's box
  function treeCuts(tree) {
    const walk = (nd, box) => {
      if (nd.leaf) return [];
      const [[x0, x1], [y0, y1]] = box, [lo, hi] = splitBox(box, nd);
      const line = nd.j === 0 ? { x1: f1(px(nd.t)), y1: f1(py(y1)), x2: f1(px(nd.t)), y2: f1(py(y0)) } : { x1: f1(px(x0)), y1: f1(py(nd.t)), x2: f1(px(x1)), y2: f1(py(nd.t)) };
      return [line, ...walk(nd.left, lo), ...walk(nd.right, hi)];
    };
    return walk(tree, [[0, LIM], [0, LIM]]);
  }

  function xgPhases(st, s) {
    const [gLo, gHi] = minMax(st.g), [hLo, hHi] = minMax(st.h);
    const lv = M().leaves(st.tree), pruned = M().nodes(st.tree).filter(n => n.pruned).length, ys = st.ys;
    return [
      { title: 'Gradient and Hessian of every point', formula: 'g_i = π_i − y*_i,   h_i = π_i(1 − π_i)',
        body: s.k === 0 ? 'f₀ = 0, so π = 0.5 everywhere: g = ±0.5, h = 0.25.' : 'g from ' + f2(gLo) + ' to ' + f2(gHi) + ', h from ' + f3(hLo) + ' to ' + f3(hHi) + ' (dot size).' },
      { title: 'Grow a tree using G and H', formula: 'gain = ½ [G_L²/(H_L+λ) + G_R²/(H_R+λ) − G²/(H+λ)]',
        body: 'Each split maximises the gain. Root: ' + (st.tree.leaf ? 'not split' : rule(st.tree) + ' (gain ' + f2(st.tree.gain) + ')') + '.' },
      { title: 'Prune with γ, set leaf weights', formula: 'drop a split if gain < γ;   w*_j = −G_j / (H_j + λ)',
        body: lv.length + (lv.length === 1 ? ' leaf' : ' leaves') + (pruned ? ', ' + pruned + ' split' + (pruned === 1 ? '' : 's') + ' pruned' : '') + '. See the table below.' },
      { title: 'Add it with learning rate η', formula: 'f_m = f_{m−1} + η · tree_m,   η = ' + ETA,
        body: 'Log loss ' + f3(M().logLoss(st.fBefore, ys)) + ' → ' + f3(M().logLoss(st.fAfter, ys)) + '. Train error ' + pct(trainErr(st.fAfter, ys)) + '.' }
    ];
  }

  const XG_NOTES = [
    'XGBoost uses a second-order expansion: ℓ(y, f + F) ≈ ℓ(y, f) + gF + ½hF². g says which way to move, h how curved the loss is.',
    'A split is good when its two children together score better than the parent. G and H are just sums of g and h over the points in a node.',
    'Ω(f) = γJ + ½λΣw²: λ shrinks leaf weights toward 0, γ is the price of each extra leaf.',
    'η shrinks each tree, like ν in gradient boosting.'
  ];

  function xgView(c) {
    const s = c.state.gb2, data = window.BoostMath.clsData(SEED, false), N = data.length;
    const run = M().xgBoost(data, s.k + 1, s.lambda, s.gamma, ETA), st = run.steps[s.k], after = s.p === 3;
    const trees = n => run.steps.slice(0, n).map(q => q.tree);
    const f = after ? st.fAfter : st.fBefore;
    return {
      ...W().bindings(c, 'gb2', 'xg', xgPhases(st, s), XG_ROUNDS),
      xgRects: s.p === 2 ? leafRects(st.tree) : modelRects(trees(after ? s.k + 1 : s.k)),
      xgCuts: s.p >= 1 && s.p <= 2 ? treeCuts(st.tree) : [],
      xgDots: data.map((d, i) => ({ x: f1(px(d.x[0])), y: f1(py(d.x[1])), fill: cls(d.y), r: f1(2.5 + 8 * Math.sqrt(st.h[i] / 0.25)) })),
      xgShowTable: s.p >= 2,
      xgLeaves: M().leafBoxes(st.tree, 2).map((b, i) => ({ name: 'Leaf ' + (i + 1), n: String(b.leaf.idx.length), G: f2(b.leaf.G), H: f2(b.leaf.H), w: f3(b.leaf.v) })),
      xgNote: XG_NOTES[s.p],
      xgLambda: s.lambda, xgLambdaText: s.lambda.toFixed(1), xgOnLambda: e => W().patch(c, 'gb2', { lambda: num(e) }),
      xgGamma: s.gamma, xgGammaText: s.gamma.toFixed(1), xgOnGamma: e => W().patch(c, 'gb2', { gamma: num(e) }),
      xgLoss: f3(M().logLoss(f, st.ys)), xgTrain: pct(trainErr(f, st.ys)),
      xgTrainLabel: 'Train error of f' + sub(after ? s.k + 1 : s.k)
    };
  }

  // ---------- state ----------

  function initState() {
    return {
      gb1: { k: 0, p: 0, sel: null, loss: 'squared' },
      gb2: { k: 0, p: 0, lambda: 1, gamma: 0 },
      ...window.GBIView.initState()
    };
  }

  function render(c) {
    return { ...gbView(c), ...xgView(c), ...window.GBIView.render(c) };
  }

  return { initState, render };
})();
