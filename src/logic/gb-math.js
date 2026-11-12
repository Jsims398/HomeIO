// Pure math for Topic 17 (gradient boosting, gradient tree boosting and XGBoost, §18.5.5; feature importance
// and partial dependence plots, §18.6). Loaded only by gradient-boosting.html, after kt-math.js and
// boost-math.js (it reuses KTMath.rng and BoostMath.regData / clsData). gb-view.js and gbi-view.js call it.
window.GBMath = (function () {
  const T = () => window.KTMath;
  const LIM = 10, EPS = 1e-9, H_FLOOR = 1e-6;
  const INT_N = 200, INT_FEATS = 4, INT_NOISE = 0.5, STEP_AT = 6;

  const gauss = rand => Math.sqrt(-2 * Math.log(rand() || 1e-12)) * Math.cos(2 * Math.PI * rand());
  const round1 = v => Math.round(v * 10) / 10;
  const sum = arr => arr.reduce((s, v) => s + v, 0);
  const mean = arr => (arr.length ? sum(arr) / arr.length : 0);
  function median(arr) {
    const s = [...arr].sort((a, b) => a - b), n = s.length;
    if (!n) return 0;
    return n % 2 ? s[(n - 1) / 2] : (s[n / 2 - 1] + s[n / 2]) / 2;
  }
  const sigmoid = f => 1 / (1 + Math.exp(-f));
  // log(1 + e^{−m}) without overflow
  const softplusNeg = m => (m > -30 ? Math.log1p(Math.exp(-m)) : -m);

  // §18.6 demo: y = 2 sin(0.6 x₁) + 0.3 x₂ + 1·[x₃ > 6] + noise; x₄ is pure noise
  const TRUE_EFFECTS = [x => 2 * Math.sin(0.6 * x), x => 0.3 * x, x => (x > STEP_AT ? 1 : 0), () => 0];

  function interpData(seed) {
    const rand = T().rng(seed);
    return Array.from({ length: INT_N }, () => {
      const x = Array.from({ length: INT_FEATS }, () => round1(rand() * LIM));
      return { x, y: sum(x.map((v, k) => TRUE_EFFECTS[k](v))) + INT_NOISE * gauss(rand) };
    });
  }

  // ---------- one tree from gradients g and Hessians h (the XGBoost view, §18.5.5.2) ----------
  // A leaf holding points I has G = Σ_{i∈I} g_i and H = Σ_{i∈I} h_i, best weight w* = −G / (H + λ),
  // and a split of a node into L and R has
  //   gain = ½ [G_L²/(H_L + λ) + G_R²/(H_R + λ) − (G_L + G_R)²/(H_L + H_R + λ)].
  // Squared loss with λ = 0 has g = f − y and h = 1, so w* is the mean residual and gain is half the
  // drop in squared error: the plain regression tree of §18.1 is the special case.

  const scoreOf = (G, H, lambda) => (G * G) / (H + lambda);

  // running sums of g and h along a sorted order (a local loop: this runs thousands of times per ensemble)
  function prefixSums(order, g, h) {
    const out = new Array(order.length);
    let GL = 0, HL = 0;
    for (let p = 0; p < order.length; p++) {
      GL += g[order[p]];
      HL += h[order[p]];
      out[p] = { GL, HL };
    }
    return out;
  }

  // every candidate split of one node on feature j (midpoints between distinct sorted values), with its gain
  function scanFeature(X, idx, g, h, j, lambda, minLeaf) {
    const order = [...idx].sort((a, b) => X[a][j] - X[b][j]);
    const prefix = prefixSums(order, g, h), n = order.length;
    if (n < 2) return [];
    const { GL: G, HL: H } = prefix[n - 1], parent = scoreOf(G, H, lambda);
    return order.slice(0, -1).flatMap((i, pos) => {
      const a = X[i][j], b = X[order[pos + 1]][j], nL = pos + 1, nR = n - nL;
      if (b - a < EPS || nL < minLeaf || nR < minLeaf) return [];
      const { GL, HL } = prefix[pos], GR = G - GL, HR = H - HL;
      return [{ j, t: (a + b) / 2, GL, HL, GR, HR, gain: 0.5 * (scoreOf(GL, HL, lambda) + scoreOf(GR, HR, lambda) - parent) }];
    });
  }

  function bestSplit(X, idx, g, h, lambda, minLeaf) {
    return X[0].reduce((best, _, j) => scanFeature(X, idx, g, h, j, lambda, minLeaf)
      .reduce((b, s) => (!b || s.gain > b.gain + EPS ? s : b), best), null);
  }

  // XGBoost-style growth: split greedily down to maxDepth, then prune bottom-up any split whose
  // children are both leaves and whose gain is below γ (the per-leaf penalty in Ω = γJ + ½λΣw²)
  function grow(X, g, h, opts) {
    const { maxDepth, lambda = 0, gamma = 0, minLeaf = 1 } = opts;
    let nextId = 0;
    function leafOf(idx, depth) {
      const G = sum(idx.map(i => g[i])), H = sum(idx.map(i => h[i]));
      return { id: nextId++, depth, idx, G, H, v: -G / (H + lambda), leaf: true };
    }
    function node(idx, depth) {
      const base = leafOf(idx, depth);
      if (depth >= maxDepth) return base;
      const best = bestSplit(X, idx, g, h, lambda, minLeaf);
      if (!best) return base;
      const left = node(idx.filter(i => X[i][best.j] <= best.t), depth + 1);
      const right = node(idx.filter(i => X[i][best.j] > best.t), depth + 1);
      if (left.leaf && right.leaf && best.gain < gamma) return { ...base, pruned: true };
      return { ...base, ...best, leaf: false, left, right };
    }
    return node(X.map((_, i) => i), 0);
  }

  const leaves = tree => (tree.leaf ? [tree] : [...leaves(tree.left), ...leaves(tree.right)]);
  const nodes = tree => (tree.leaf ? [tree] : [tree, ...nodes(tree.left), ...nodes(tree.right)]);
  const leafAt = (tree, x) => (tree.leaf ? tree : leafAt(x[tree.j] <= tree.t ? tree.left : tree.right, x));
  const predict = (tree, x) => leafAt(tree, x).v;

  // copy of the tree with every leaf passed through fn
  const mapLeaves = (tree, fn) => (tree.leaf ? fn(tree) : { ...tree, left: mapLeaves(tree.left, fn), right: mapLeaves(tree.right, fn) });

  // each leaf with its box: one [lo, hi] interval per feature
  function leafBoxes(tree, nDims) {
    const walk = (nd, box) => {
      if (nd.leaf) return [{ leaf: nd, box }];
      const lo = box.map((iv, k) => (k === nd.j ? [iv[0], nd.t] : iv)), hi = box.map((iv, k) => (k === nd.j ? [nd.t, iv[1]] : iv));
      return [...walk(nd.left, lo), ...walk(nd.right, hi)];
    };
    return walk(tree, Array.from({ length: nDims }, () => [0, LIM]));
  }

  // ---------- §18.5.5.1 gradient tree boosting on 1-D data ----------
  // negGrad is the pseudo-residual −∂ℓ/∂f; best(r) = argmin_γ Σ ℓ(r_i − γ) is the leaf line search
  const LOSSES = {
    squared: { value: (y, f) => 0.5 * (y - f) ** 2, negGrad: (y, f) => y - f, best: mean },
    absolute: { value: (y, f) => Math.abs(y - f), negGrad: (y, f) => Math.sign(y - f), best: median }
  };
  const meanLoss = (L, y, f) => mean(y.map((v, i) => L.value(v, f[i])));

  // f₀ = argmin_γ Σ ℓ(y_i, γ). Each round: r = −gradient, fit a least-squares tree to r (regions R_jm),
  // then re-solve each leaf for the real loss, γ_jm = argmin_γ Σ_{x_i∈R_jm} ℓ(y_i, f(x_i) + γ), and add ν Σ_j γ_jm I(x ∈ R_jm)
  function gradBoost(data, rounds, lossName, nu, depth) {
    const L = LOSSES[lossName], X = data.map(d => d.x), y = data.map(d => d.y), ones = y.map(() => 1), f0 = L.best(y);
    const steps = [];
    let f = y.map(() => f0);
    for (let m = 0; m < rounds; m++) {
      const fm = f;
      const r = y.map((v, i) => L.negGrad(v, fm[i]));
      const fit = grow(X, r.map(v => -v), ones, { maxDepth: depth, minLeaf: 2 });
      const tree = mapLeaves(fit, lf => ({ ...lf, fitV: lf.v, v: L.best(lf.idx.map(i => y[i] - fm[i])) }));
      const fAfter = fm.map((v, i) => v + nu * predict(tree, X[i]));
      steps.push({ fBefore: fm, res: y.map((v, i) => v - fm[i]), r, tree, fAfter,
        lossBefore: meanLoss(L, y, fm), lossAfter: meanLoss(L, y, fAfter) });
      f = fAfter;
    }
    return { f0, steps };
  }

  // ---------- §18.5.5.2 XGBoost on 2-D binary data (log loss, π = σ(f)) ----------
  // g_i = π_i − y*_i, h_i = π_i(1 − π_i); each round grows a depth-2 tree with λ and γ, then f ← f + η·tree
  const XGB_DEPTH = 2;
  const logLoss = (f, ys) => mean(f.map((v, i) => softplusNeg((2 * ys[i] - 1) * v)));

  function xgBoost(data, rounds, lambda, gamma, eta) {
    const X = data.map(d => d.x), ys = data.map(d => (d.y > 0 ? 1 : 0)), steps = [];
    let f = ys.map(() => 0);
    for (let m = 0; m < rounds; m++) {
      const fm = f, pi = fm.map(sigmoid);
      const g = pi.map((p, i) => p - ys[i]), h = pi.map(p => Math.max(p * (1 - p), H_FLOOR));
      const tree = grow(X, g, h, { maxDepth: XGB_DEPTH, lambda, gamma });
      const fAfter = fm.map((v, i) => v + eta * predict(tree, X[i]));
      steps.push({ fBefore: fm, pi, g, h, tree, fAfter, ys });
      f = fAfter;
    }
    return { steps };
  }

  // ---------- §18.6 a boosted ensemble of regression trees, its feature importance and partial dependence ----------

  // squared-loss gradient boosting with depth-limited trees (g = f − y, h = 1)
  function fitEnsemble(data, M, nu, depth, minLeaf) {
    const X = data.map(d => d.x), y = data.map(d => d.y), ones = y.map(() => 1), f0 = mean(y), trees = [], mse = [];
    let f = y.map(() => f0);
    for (let m = 0; m < M; m++) {
      const fm = f;
      const tree = grow(X, fm.map((v, i) => v - y[i]), ones, { maxDepth: depth, minLeaf });
      f = fm.map((v, i) => v + nu * predict(tree, X[i]));
      trees.push(tree);
      mse.push(mean(y.map((v, i) => (v - f[i]) ** 2)));
    }
    return { f0, nu, trees, mse, X, y };
  }

  const predictEnsemble = (model, m, x) => model.trees.slice(0, m).reduce((f, tr) => f + model.nu * predict(tr, x), model.f0);

  // R_k(T) = Σ_{internal nodes j} G_j I(v_j = k), with G_j the drop in squared error at node j (2 × gain);
  // averaged over the first m trees and rescaled so the top feature scores 100
  function importance(model, m, nFeat) {
    const splits = model.trees.slice(0, m).flatMap(tree => nodes(tree).filter(nd => !nd.leaf).map(nd => ({ j: nd.j, drop: 2 * nd.gain })));
    const raw = Array.from({ length: nFeat }, (_, k) => sum(splits.filter(s => s.j === k).map(s => s.drop)) / Math.max(m, 1));
    const top = Math.max(...raw);
    return raw.map(v => (top > 0 ? (100 * v) / top : 0));
  }

  // f(x with x_k replaced by v) for every training point, at every grid value v.
  // The partial dependence f̄_k(v) = (1/N) Σ_n f(v, x_{n,−k}) is the mean of each column
  function partialDependence(model, m, k, grid) {
    const curves = model.X.map(x => grid.map(v => predictEnsemble(model, m, x.map((u, j) => (j === k ? v : u)))));
    const pd = grid.map((_, g) => mean(curves.map(c => c[g])));
    return { curves, pd };
  }

  return {
    LIM, TRUE_EFFECTS, sigmoid, mean, interpData, grow, leaves, nodes, leafAt, predict, leafBoxes,
    LOSSES, gradBoost, xgBoost, logLoss, fitEnsemble, importance, partialDependence
  };
})();
