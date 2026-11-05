// Pure math for Topic 15 (classification trees: splitting D into D_L and D_R, Gini, greedy growth, §18.1).
// Loaded only by cart.html, after kt-math.js (it reuses KTMath.rng). cart-view.js calls it.
window.CARTMath = (function () {
  const T = () => window.KTMath;
  const CLASSES = 3, LIM = 10, PER_CLASS = 12, SPREAD = 1.9, EPS = 1e-9;
  const CENTERS = [[3, 3], [7, 4.5], [3.5, 7.5]];

  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const gauss = rand => Math.sqrt(-2 * Math.log(rand() || 1e-12)) * Math.cos(2 * Math.PI * rand());
  const round1 = v => Math.round(v * 10) / 10;

  // three overlapping blobs on [0, 10]²; values rounded to 0.1 so the arithmetic is easy to follow by hand
  function dataset(seed) {
    const rand = T().rng(seed), pts = [];
    CENTERS.forEach(([cx, cy], c) => {
      for (let n = 0; n < PER_CLASS; n++) {
        pts.push([round1(clamp(cx + SPREAD * gauss(rand), 0.2, LIM - 0.2)), round1(clamp(cy + SPREAD * gauss(rand), 0.2, LIM - 0.2)), c]);
      }
    });
    return pts;
  }

  // N_c for each class c
  const counts = pts => pts.reduce((acc, q) => acc.map((v, c) => v + (q[2] === c ? 1 : 0)), new Array(CLASSES).fill(0));

  // π̂_c = N_c / |D|
  function proportions(cnt) {
    const N = cnt.reduce((s, v) => s + v, 0);
    return N ? cnt.map(v => v / N) : cnt.map(() => 0);
  }

  // impurity of one node, from its class counts:
  //   gini    G = Σ π̂_c (1 − π̂_c) = 1 − Σ π̂_c²
  //   entropy H = − Σ π̂_c log₂ π̂_c
  //   error   1 − max_c π̂_c
  const IMPURITY = {
    gini: p => 1 - p.reduce((s, v) => s + v * v, 0),
    entropy: p => -p.reduce((s, v) => s + (v > 0 ? v * Math.log2(v) : 0), 0),
    error: p => 1 - Math.max(...p)
  };
  // an empty node has no impurity (and gets weight |D_L|/|D| = 0 in the split cost anyway)
  const impurity = (kind, cnt) => (cnt.some(v => v > 0) ? IMPURITY[kind](proportions(cnt)) : 0);

  // D_L = {n : x_nj ≤ t},  D_R = {n : x_nj > t}
  const split = (pts, j, t) => ({ left: pts.filter(q => q[j] <= t), right: pts.filter(q => q[j] > t) });

  // cost(j, t) = |D_L|/|D| · c(D_L) + |D_R|/|D| · c(D_R)
  function splitCost(kind, pts, j, t) {
    const { left, right } = split(pts, j, t), N = pts.length;
    const cntL = counts(left), cntR = counts(right);
    const cL = impurity(kind, cntL), cR = impurity(kind, cntR);
    const cost = N ? (left.length / N) * cL + (right.length / N) * cR : 0;
    return { left, right, cntL, cntR, cL, cR, cost };
  }

  // candidate thresholds T_j: midpoints between consecutive sorted unique values of x_nj.
  // Any t between two neighbours gives the same D_L / D_R, so these are all the splits there are
  function thresholds(pts, j) {
    const vals = [...new Set(pts.map(q => q[j]))].sort((a, b) => a - b);
    return vals.slice(1).map((v, i) => (vals[i] + v) / 2);
  }

  // argmin over features j and thresholds t ∈ T_j, keeping each child at least minLeaf points
  function bestSplit(kind, pts, minLeaf) {
    let best = null;
    [0, 1].forEach(j => {
      thresholds(pts, j).forEach(t => {
        const r = splitCost(kind, pts, j, t);
        if (r.left.length < minLeaf || r.right.length < minLeaf) return;
        if (!best || r.cost < best.cost - EPS) best = { j, t, cost: r.cost };
      });
    });
    return best;
  }

  // greedy recursive growth. A node becomes a leaf when it is pure, hits maxDepth,
  // has no split that keeps minLeaf points per side, or the best split does not lower the impurity
  function grow(kind, pts, maxDepth, minLeaf) {
    let nextId = 0;
    function node(subset, depth, box) {
      const cnt = counts(subset), imp = impurity(kind, cnt);
      const base = { id: nextId++, depth, box, n: subset.length, cnt, imp, label: cnt.indexOf(Math.max(...cnt)) };
      if (imp <= EPS) return { ...base, leaf: true, reason: 'pure' };
      if (depth >= maxDepth) return { ...base, leaf: true, reason: 'depth' };
      const best = bestSplit(kind, subset, minLeaf);
      if (!best) return { ...base, leaf: true, reason: 'minLeaf' };
      if (best.cost >= imp - EPS) return { ...base, leaf: true, reason: 'noGain' };
      const { left, right } = split(subset, best.j, best.t);
      const [x0, x1, y0, y1] = box;
      const boxL = best.j === 0 ? [x0, best.t, y0, y1] : [x0, x1, y0, best.t];
      const boxR = best.j === 0 ? [best.t, x1, y0, y1] : [x0, x1, best.t, y1];
      return { ...base, leaf: false, j: best.j, t: best.t, cost: best.cost, left: node(left, depth + 1, boxL), right: node(right, depth + 1, boxR) };
    }
    return node(pts, 0, [0, LIM, 0, LIM]);
  }

  const leaves = tree => (tree.leaf ? [tree] : [...leaves(tree.left), ...leaves(tree.right)]);
  const nodes = tree => (tree.leaf ? [tree] : [tree, ...nodes(tree.left), ...nodes(tree.right)]);

  // walk from the root to the leaf that x lands in
  const predict = (tree, x) => (tree.leaf ? tree.label : predict(x[tree.j] <= tree.t ? tree.left : tree.right, x));

  const trainError = (tree, pts) => (pts.length ? pts.filter(q => predict(tree, q) !== q[2]).length / pts.length : 0);

  return { CLASSES, LIM, dataset, counts, proportions, impurity, split, splitCost, thresholds, bestSplit, grow, leaves, nodes, predict, trainError };
})();
