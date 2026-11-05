// View logic for Topic 15 (classification trees: D_L / D_R, Gini, greedy growth, §18.1).
// Loaded only by cart.html. reading-log.js calls CARTView.initState()
// for the starting state and CARTView.render(component) from t15(); the math is in cart-math.js.
window.CARTView = (function () {
  const M = () => window.CARTMath;
  const COLORS = ['var(--color-accent)', 'var(--color-accent-2-600)', '#4f6d8a'];
  const NAMES = ['A', 'B', 'C'];
  const FEAT = ['x₁', 'x₂'];
  const KINDS = { gini: 'Gini', entropy: 'Entropy', error: 'Error rate' };
  const SYMBOL = { gini: 'G', entropy: 'H', error: 'E' };
  const SEED = 15, EPS = 1e-9;

  const pill = active => ({ bg: active ? 'var(--color-accent)' : 'transparent', fg: active ? 'var(--color-bg)' : 'var(--color-text)' });
  const patch = (c, key, change) => c.setState({ [key]: { ...c.state[key], ...change } });
  const int = e => parseInt(e.target.value, 10);
  const num = e => parseFloat(e.target.value);
  const f1 = v => v.toFixed(1);
  const f2 = v => v.toFixed(2);
  const f3 = v => v.toFixed(3);
  const pct = v => Math.round(v * 100) + '%';

  // shared plot of [0, 10]² in a 300 × 300 box
  const BOX = 300;
  const px = v => (v / M().LIM) * BOX;
  const py = v => BOX - (v / M().LIM) * BOX;
  const dots = (pts, onClick) => pts.map((q, n) => ({ x: f1(px(q[0])), y: f1(py(q[1])), fill: COLORS[q[2]], onClick: onClick ? () => onClick(n) : () => {} }));

  const kindBtns = c => Object.keys(KINDS).map(k => ({ label: KINDS[k], ...pill(k === c.state.ct.kind), onClick: () => patch(c, 'ct', { kind: k }) }));

  // the impurity of one node written out with its numbers, e.g. 1 − (0.50² + 0.42² + 0.08²) = 0.569
  function worked(kind, cnt) {
    const N = cnt.reduce((s, v) => s + v, 0);
    if (!N) return 'empty, so 0';
    const p = M().proportions(cnt), val = f3(M().impurity(kind, cnt));
    if (kind === 'gini') return '1 − (' + p.map(v => f2(v) + '²').join(' + ') + ') = ' + val;
    if (kind === 'entropy') return '−(' + p.filter(v => v > 0).map(v => f2(v) + '·log₂ ' + f2(v)).join(' + ') + ') = ' + val;
    return '1 − max(' + p.map(f2).join(', ') + ') = ' + val;
  }

  // one row of the counts table: N_c and π̂_c = N_c / |D| for each class
  function countRow(name, cnt, kind) {
    const N = cnt.reduce((s, v) => s + v, 0);
    return {
      name, n: String(N),
      cells: cnt.map((v, k) => ({ text: v + (N ? ' (' + f2(v / N) + ')' : ''), color: COLORS[k] })),
      bar: cnt.map((v, k) => ({ w: N ? (100 * v) / N + '%' : '0%', bg: COLORS[k] })),
      imp: f3(M().impurity(kind, cnt))
    };
  }

  // ---------- part 1: one split, by hand ----------

  const CW = 560, CH = 200, CPL = 44, CPR = 16, CPT = 14, CPB = 30, CSTEP = 0.05;
  const cx = t => CPL + (t / M().LIM) * (CW - CPL - CPR);
  const cy = (v, top) => CH - CPB - (v / top) * (CH - CPT - CPB);

  // lowest-cost threshold for one feature (the dip in the curve)
  function bestOnFeature(kind, pts, j) {
    return M().thresholds(pts, j).reduce((b, t) => {
      const cost = M().splitCost(kind, pts, j, t).cost;
      return !b || cost < b.cost - EPS ? { j, t, cost } : b;
    }, null);
  }

  function cycleClass(c, n) {
    const pts = c.state.ct.pts.map((q, i) => (i === n ? [q[0], q[1], (q[2] + 1) % M().CLASSES] : q));
    patch(c, 'ct', { pts });
  }

  // shaded half-planes, the cut line and the D_L / D_R labels for x_j ≤ t
  function halves(j, t) {
    if (j === 0) {
      const at = px(t);
      return { sLeft: { x: 0, y: 0, w: f1(at), h: BOX }, sRight: { x: f1(at), y: 0, w: f1(BOX - at), h: BOX },
        sCut: { x1: f1(at), y1: 0, x2: f1(at), y2: BOX }, sLabL: { x: f1(at - 8), y: 18, anchor: 'end' }, sLabR: { x: f1(at + 8), y: 18, anchor: 'start' } };
    }
    const at = py(t);
    return { sLeft: { x: 0, y: f1(at), w: BOX, h: f1(BOX - at) }, sRight: { x: 0, y: 0, w: BOX, h: f1(at) },
      sCut: { x1: 0, y1: f1(at), x2: BOX, y2: f1(at) }, sLabL: { x: 292, y: f1(at + 18), anchor: 'end' }, sLabR: { x: 292, y: f1(at - 8), anchor: 'end' } };
  }

  // cost(t) on a fine grid, so the curve shows the flat steps between data values
  function costCurve(kind, pts, j, parent) {
    const ts = Array.from({ length: Math.round(M().LIM / CSTEP) + 1 }, (_, i) => i * CSTEP);
    const vals = ts.map(t => M().splitCost(kind, pts, j, t).cost);
    const top = Math.max(parent, ...vals, EPS) * 1.1;
    const path = ts.map((t, i) => (i ? 'L ' : 'M ') + f1(cx(t)) + ' ' + f1(cy(vals[i], top))).join(' ');
    return { path, top };
  }

  function splitNote(kind, r, parent, isBest, nL, nR) {
    if (!nL || !nR) return 'One side is empty, so nothing has been split: the cost is just c(D) again. Move t inside the data.';
    if (isBest) return 'This is the best split there is, over both features and every threshold. It is the one the tree makes first.';
    if (kind === 'error' && Math.abs(parent - r.cost) < EPS) return 'The error rate did not move, even though the children may be purer. Error rate only looks at the majority class, so many useful splits look like no progress. That is why trees use Gini or entropy.';
    if (r.cL < EPS || r.cR < EPS) return 'One child is pure: its impurity is 0, so only the other child adds to the cost.';
    return 'Each child\'s impurity is weighted by its share of the points. A tiny pure child barely helps; a large child that got purer helps a lot.';
  }

  function oneSplit(c) {
    const { pts, kind } = c.state.ct, { j, t } = c.state.ctSplit, N = pts.length;
    const r = M().splitCost(kind, pts, j, t), cntD = M().counts(pts), parent = M().impurity(kind, cntD);
    const best = M().bestSplit(kind, pts, 1) || { j, t, cost: parent };
    const here = bestOnFeature(kind, pts, j) || { t, cost: r.cost };
    const nL = r.left.length, nR = r.right.length;
    const isBest = nL > 0 && nR > 0 && Math.abs(r.cost - best.cost) < EPS;
    const curve = costCurve(kind, pts, j, parent);
    return {
      ...halves(j, t),
      sDots: dots(pts, n => cycleClass(c, n)),
      sFeatBtns: FEAT.map((f, k) => ({ label: 'Split on ' + f, ...pill(k === j), onClick: () => patch(c, 'ctSplit', { j: k }) })),
      sKindBtns: kindBtns(c),
      sT: t, sOnT: e => patch(c, 'ctSplit', { t: num(e) }),
      sRule: FEAT[j] + ' ≤ ' + f2(t), sRuleR: FEAT[j] + ' > ' + f2(t),
      sOnBest: () => patch(c, 'ctSplit', { j: best.j, t: best.t }),
      sOnResample: () => patch(c, 'ct', { seed: c.state.ct.seed + 1, pts: M().dataset(c.state.ct.seed + 1) }),
      sRows: [countRow('D', cntD, kind), countRow('D_L', r.cntL, kind), countRow('D_R', r.cntR, kind)],
      sSym: SYMBOL[kind], sKind: KINDS[kind],
      sWorkD: worked(kind, cntD), sWorkL: worked(kind, r.cntL), sWorkR: worked(kind, r.cntR),
      sNL: String(nL), sNR: String(nR), sN: String(N),
      sCostLine: nL + '/' + N + ' · ' + f3(r.cL) + ' + ' + nR + '/' + N + ' · ' + f3(r.cR) + ' = ' + f3(r.cost),
      sCost: f3(r.cost), sParent: f3(parent), sGain: f3(parent - r.cost),
      sBestRule: FEAT[best.j] + ' ≤ ' + f2(best.t), sBestCost: f3(best.cost),
      sCurve: curve.path,
      sParentY: f1(cy(parent, curve.top)),
      sMarkX: f1(cx(t)),
      sBestX: f1(cx(here.t)), sBestY: f1(cy(here.cost, curve.top)),
      sCands: M().thresholds(pts, j).map(v => ({ x: f1(cx(v)) })),
      sCYTicks: [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5].filter(v => v < curve.top).map(v => ({ y: f1(cy(v, curve.top)) })),
      sFeat: FEAT[j],
      sNote: splitNote(kind, r, parent, isBest, nL, nR)
    };
  }

  // ---------- part 2: grow the whole tree greedily ----------

  const TW = 560, TPAD = 24, TROW = 66, TTOP = 26, NODE_W = 80, NODE_H = 36, LEAF_R = 15;
  const REASON = {
    pure: 'it is pure (impurity 0), so there is nothing left to separate.',
    depth: 'it hit the maximum depth.',
    noGain: 'no split lowers the weighted impurity, so splitting would not help.'
  };

  // x of each node: leaves evenly spaced left to right, a parent sits midway over its two children
  function layout(tree) {
    const leaves = M().leaves(tree), gap = (TW - 2 * TPAD) / leaves.length, pos = new Map();
    leaves.forEach((lf, i) => pos.set(lf.id, TPAD + (i + 0.5) * gap));
    const place = n => {
      if (n.leaf) return;
      place(n.left);
      place(n.right);
      pos.set(n.id, (pos.get(n.left.id) + pos.get(n.right.id)) / 2);
    };
    place(tree);
    return pos;
  }

  function diagram(c, tree, sel) {
    const pos = layout(tree), all = M().nodes(tree), ny = d => TTOP + d * TROW, pick = id => () => patch(c, 'ctGrow', { sel: id });
    const depth = Math.max(...all.map(n => n.depth)), sym = SYMBOL[c.state.ct.kind], H = ny(depth) + 44;
    // the page runtime does not fill bound text inside SVG <text>, so labels are HTML laid over the SVG at % positions
    const at = (x, y) => ({ left: ((100 * x) / TW).toFixed(2) + '%', top: ((100 * y) / H).toFixed(2) + '%' });
    const edges = all.filter(n => !n.leaf).flatMap(n => [n.left, n.right].map((ch, side) => ({
      x1: f1(pos.get(n.id)), y1: f1(ny(n.depth) + NODE_H / 2), x2: f1(pos.get(ch.id)), y2: f1(ny(ch.depth) - (ch.leaf ? LEAF_R : NODE_H / 2)),
      ...at((pos.get(n.id) + pos.get(ch.id)) / 2 + (side ? 10 : -10), (ny(n.depth) + ny(ch.depth)) / 2), label: side ? 'no' : 'yes'
    })));
    return {
      gTreeBox: '0 0 ' + TW + ' ' + H,
      gTreeRatio: TW + ' / ' + H,
      gEdges: edges,
      gInner: all.filter(n => !n.leaf).map(n => ({
        x: f1(pos.get(n.id) - NODE_W / 2), y: f1(ny(n.depth) - NODE_H / 2), ...at(pos.get(n.id), ny(n.depth)),
        rule: FEAT[n.j] + ' ≤ ' + f2(n.t), sub: 'n=' + n.n + ' · ' + sym + '=' + f2(n.imp),
        stroke: n.id === sel ? 'var(--color-text)' : 'var(--color-neutral-400)', sw: n.id === sel ? 2.4 : 1, onClick: pick(n.id)
      })),
      gLeaves: M().leaves(tree).map(n => ({
        x: f1(pos.get(n.id)), y: f1(ny(n.depth)), ...at(pos.get(n.id), ny(n.depth)), subTop: at(0, ny(n.depth) + LEAF_R + 9).top,
        fill: COLORS[n.label], n: String(n.n), sub: sym + '=' + f2(n.imp),
        stroke: n.id === sel ? 'var(--color-text)' : 'var(--color-surface)', sw: n.id === sel ? 3 : 1.5, onClick: pick(n.id)
      }))
    };
  }

  // leaf rectangles tinted by their majority class (stronger = purer), plus every split line clipped to its node's box
  function regions(tree, chosen) {
    const rect = b => ({ x: f1(px(b[0])), y: f1(py(b[3])), w: f1(px(b[1]) - px(b[0])), h: f1(py(b[2]) - py(b[3])) });
    const purity = n => (Math.max(...M().proportions(n.cnt)) - 1 / 3) * 1.5;
    return {
      gCells: M().leaves(tree).map(n => ({ ...rect(n.box), fill: COLORS[n.label], opacity: f2(0.08 + 0.3 * purity(n)) })),
      gCuts: M().nodes(tree).filter(n => !n.leaf).map(n => {
        const [x0, x1, y0, y1] = n.box;
        return n.j === 0 ? { x1: f1(px(n.t)), y1: f1(py(y1)), x2: f1(px(n.t)), y2: f1(py(y0)) } : { x1: f1(px(x0)), y1: f1(py(n.t)), x2: f1(px(x1)), y2: f1(py(n.t)) };
      }),
      gSelBox: rect(chosen.box)
    };
  }

  function nodeDetail(kind, n, minLeaf) {
    const sym = SYMBOL[kind], counts = n.cnt.map((v, k) => NAMES[k] + ' ' + v).join(', ');
    const head = (n.depth === 0 ? 'Root' : 'Node at depth ' + n.depth) + ': ' + n.n + ' points (' + counts + '), ' + sym + ' = ' + f3(n.imp) + '. ';
    if (n.leaf) {
      const why = n.reason === 'minLeaf' ? 'every possible split would leave a child with fewer than ' + minLeaf + ' points.' : REASON[n.reason];
      return head + 'Leaf: predicts class ' + NAMES[n.label] + ' (the majority). It stopped because ' + why;
    }
    return head + 'Best split ' + FEAT[n.j] + ' ≤ ' + f2(n.t) + ' sends ' + n.left.n + ' points to D_L (' + sym + ' = ' + f3(n.left.imp) + ') and ' + n.right.n +
      ' to D_R (' + sym + ' = ' + f3(n.right.imp) + '). Weighted cost ' + n.left.n + '/' + n.n + ' · ' + f3(n.left.imp) + ' + ' + n.right.n + '/' + n.n + ' · ' + f3(n.right.imp) +
      ' = ' + f3(n.cost) + ', down from ' + f3(n.imp) + '.';
  }

  function growNote(maxDepth, leafCount, err) {
    if (maxDepth === 0) return 'Depth 0 is just the root: one leaf that predicts the majority class for everything. This is the "no split" baseline.';
    if (err < EPS) return 'Every training point is classified correctly. The small boxes are chasing single points in the overlap: this tree has memorised the sample. §18.1.3 is about stopping or pruning before this.';
    if (leafCount <= 4) return 'A few cuts already catch the three blobs. Each cut is the best (D_L, D_R) for the points that reach that node, found exactly as in part ①.';
    return 'Deeper nodes see fewer points, so their splits are fitted to smaller, noisier samples. Training error keeps falling; that alone does not mean the tree is getting better.';
  }

  function growTree(c) {
    const { pts, kind } = c.state.ct, g = c.state.ctGrow;
    const tree = M().grow(kind, pts, g.depth, g.minLeaf), all = M().nodes(tree);
    const chosen = all.find(n => n.id === g.sel) || tree, leaves = M().leaves(tree), err = M().trainError(tree, pts);
    return {
      ...regions(tree, chosen), ...diagram(c, tree, chosen.id),
      gDots: dots(pts),
      gKindBtns: kindBtns(c),
      gDepth: g.depth, gOnDepth: e => patch(c, 'ctGrow', { depth: int(e) }),
      gMin: g.minLeaf, gOnMin: e => patch(c, 'ctGrow', { minLeaf: int(e) }),
      gLeafCount: String(leaves.length), gErr: pct(err), gUsed: String(Math.max(...all.map(n => n.depth))),
      gDetail: nodeDetail(kind, chosen, g.minLeaf),
      gNote: growNote(g.depth, leaves.length, err)
    };
  }

  // ---------- state ----------

  function initState() {
    return {
      ct: { seed: SEED, pts: M().dataset(SEED), kind: 'gini' },
      ctSplit: { j: 0, t: 4 },
      ctGrow: { depth: 2, minLeaf: 1, sel: 0 }
    };
  }

  function render(c) {
    return { ...oneSplit(c), ...growTree(c) };
  }

  return { initState, render };
})();
