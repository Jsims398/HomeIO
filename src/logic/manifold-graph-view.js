// View logic for Topic 20 ③ (approaches to manifold learning, §20.4.3): the nonparametric recipe on the
// spiral from ①: k-NN graph → shortest-path distances → classical MDS, and how a big k short-circuits it.
// Loaded only by manifold.html after manifold-roll-view.js (reuses its data and geom); reading-log.js
// merges render(c) into t20(). Math is in manifold-math.js; Next/Back is Walkthrough.steps.
window.ManifoldGraphView = (function () {
  const M = () => window.ManifoldMath, W = () => window.Walkthrough;
  const R = () => window.ManifoldRollView, G = () => R().geom;
  const KEY = 'mg';
  const K_MIN = 2, K_MAX = 14, SHORTCUT_RATIO = 1.5;
  const STEP = { GRAPH: 1, PATH: 2, EMBED: 3, BREAK: 4 };
  const COL = { edge: 'var(--color-neutral-400)', shortcut: '#b3412e' };

  const f2 = v => v.toFixed(2);

  // an edge is a shortcut when the curve distance between its ends is much longer than the edge
  const isShortcut = (d, e) => Math.abs(d.t[e.i] - d.t[e.j]) > SHORTCUT_RATIO * e.w;

  let firstBreak = null;
  function breakK(d) {
    if (firstBreak !== null) return firstBreak;
    const ks = Array.from({ length: K_MAX - K_MIN + 1 }, (_, i) => K_MIN + i);
    firstBreak = ks.find(k => M().knnGraph(d.X, k).edges.some(e => isShortcut(d, e))) || K_MAX + 1;
    return firstBreak;
  }

  // full pipeline per k, cached because MDS on a 120×120 matrix takes a moment
  let runs = {};
  function run(d, k) {
    if (runs[k]) return runs[k];
    const r = M().isomap1(d.X, k), last = d.X.length - 1;
    const flip = r.z && r.z[last] < r.z[0] ? -1 : 1;
    const z = r.z ? r.z.map(v => v * flip) : null;
    const res = {
      ...r, z, path: r.connected ? M().pathTo(r.runs[0].prev, last) : [], pathLen: r.D[0][last],
      rho: z ? M().spearman(z, d.t) : null, shortcuts: r.edges.filter(e => isShortcut(d, e)).length
    };
    runs = { ...runs, [k]: res };
    return res;
  }

  function phases(d, k, r, kb) {
    const straight = M().dist(d.X[0], d.X[d.X.length - 1]), ok = r.shortcuts === 0;
    return [
      { title: 'Two families of methods', formula: 'parametric: z = f(x)    nonparametric: z₁, …, z_N only',
        body: 'Parametric methods (PCA, autoencoders) learn a map f, so a new x gets its z for free. Nonparametric methods only output z_n for the training points. ' +
          'Most nonparametric ones follow one recipe: build a neighbour graph, then find z that keeps some property of that graph. Here is that recipe on the spiral from ①.' },
      { title: 'Build the k-nearest-neighbour graph', formula: 'edge i — j  if j is among the ' + k + ' nearest of i (or i of j)',
        body: 'Link each point to its ' + k + ' nearest neighbours. Only short, local distances are kept, the ones ① showed we can trust. ' +
          r.edges.length + ' edges, ' + (ok ? 'all following the curve.' : r.shortcuts + ' of them (red) jumping across a gap between turns.') },
      { title: 'Distances along the graph', formula: 'D_ij = length of the shortest path i → j  (Dijkstra)',
        body: 'Add up edge lengths along the shortest route. From the inner end to the outer end: ' + f2(r.pathLen) + ' through the graph, vs the true length along the curve ' +
          f2(d.total) + ' and the straight line ' + f2(straight) + '. ' + (ok ? 'The graph path hugs the curve, so it estimates distance on the manifold.' : 'The path cuts across turns, so it underestimates.') },
      { title: 'Embed: classical MDS', formula: 'B = −½ J D⁽²⁾ J,   z = √λ₁ u₁',
        body: 'Find 1-D numbers whose gaps match the graph distances: double-centre the squared distances and take the top eigenvector. That is classical MDS; running it on graph distances is Isomap (§20.4.5). ' +
          (r.z ? 'Rank correlation with the true order: ' + f2(r.rho) + ', against ' + f2(R().model().rhoPca) + ' for PCA in ①.' : 'The graph is in pieces, so some distances are infinite and MDS can\'t run.') },
      { title: 'Choose k badly: short-circuit', formula: 'k ≥ ' + kb + ' here  ⇒  an edge jumps between turns',
        body: 'Drag k up to ' + kb + ' or more. Near the ends of the curve the ' + kb + 'th nearest point is on the next turn, a red edge appears, and many shortest paths use it. ' +
          'The embedding folds and the rank correlation drops. Too small a k and the graph can fall apart; too big and it short-circuits. Every graph-based method shares this weakness.' },
      { title: 'The landscape', formula: 'linear or not · parametric or not · convex or not',
        body: 'The table under this box sorts the methods coming up in §20.4. Convex ones (an eigenvalue problem) always give the same answer; non-convex ones (t-SNE, autoencoders) can land in different local optima from different starts.' }
    ];
  }

  function stats(d, p, k, r) {
    const base = [{ k: 'k', v: String(k) }, { k: 'edges', v: String(r.edges.length) }, { k: 'shortcuts', v: String(r.shortcuts) }];
    if (p === STEP.PATH) return [...base, { k: 'graph path', v: f2(r.pathLen) }, { k: 'true length', v: f2(d.total) }];
    if (p >= STEP.EMBED) return [...base, { k: 'rank corr.', v: r.rho === null ? '—' : f2(r.rho) }];
    return base;
  }

  function render(c) {
    const d = R().model(), s = c.state[KEY], p = s.p, k = s.k, r = run(d, k), kb = breakK(d);
    const st = r.z && p >= STEP.EMBED ? G().strip(r.z, d.u, []) : { dots: [] };
    return {
      ...W().steps(c, KEY, 'mg', phases(d, k, r, kb)),
      mgK: k, mgKMin: K_MIN, mgKMax: K_MAX,
      mgOnK: e => W().patch(c, KEY, { k: parseInt(e.target.value, 10) }),
      mgDots: d.X.map((x, n) => ({ ...G().pt(x), color: G().color(d.u[n]) })),
      mgShowEdges: p >= STEP.GRAPH,
      mgEdges: r.edges.map(e => ({ ...G().seg(d.X[e.i], d.X[e.j]), color: isShortcut(d, e) ? COL.shortcut : COL.edge, w: isShortcut(d, e) ? 3 : 1.2 })),
      mgShowPath: p >= STEP.PATH && r.connected, mgPath: G().polyline(r.path.map(n => d.X[n])),
      mgShowStrip: p >= STEP.EMBED && r.z !== null, mgStrip: st.dots,
      mgStripTitle: 'MDS coordinate z (k = ' + k + '): ' + (r.shortcuts === 0 ? 'colours in order' : 'folded by the shortcut'),
      mgShowBroken: p >= STEP.EMBED && r.z === null,
      mgStats: stats(d, p, k, r)
    };
  }

  function initState() {
    return { [KEY]: { p: 0, k: 6 } };
  }

  return { initState, render };
})();
