// View logic for Topic 17, part 2 (§18.6 interpreting tree ensembles): feature importance and partial
// dependence of one boosted ensemble of regression trees. Loaded only by gradient-boosting.html, before
// gb-view.js, which merges GBIView.initState() and GBIView.render(component) into its own.
// The math is in gb-math.js.
window.GBIView = (function () {
  const M = () => window.GBMath;
  const SEED = 17, MAX_M = 100, NU = 0.1, DEPTH = 3, MIN_LEAF = 5;
  const GRID_STEP = 0.25, LIM = 10;
  const GRID = Array.from({ length: Math.round(LIM / GRID_STEP) + 1 }, (_, i) => i * GRID_STEP);
  const NAMES = ['x₁', 'x₂', 'x₃', 'x₄'];
  const TRUTH = ['2 sin(0.6 x₁)', '0.3 x₂', '1 if x₃ > 6', 'none (noise)'];

  const num = e => parseFloat(e.target.value);
  const fx = d => v => (v < 0 ? '−' : '') + Math.abs(v).toFixed(d);
  const f1 = fx(1), f2 = fx(2);
  const patch = (c, change) => c.setState({ gbI: { ...c.state.gbI, ...change } });

  // fitting 100 trees takes ~50 ms and one partial dependence curve ~30 ms, so keep each result once computed
  const cache = new Map();
  const memo = (key, make) => {
    if (!cache.has(key)) cache.set(key, make());
    return cache.get(key);
  };
  const model = () => memo('model', () => M().fitEnsemble(M().interpData(SEED), MAX_M, NU, DEPTH, MIN_LEAF));
  const pdFor = (m, k) => memo('pd|' + m + '|' + k, () => M().partialDependence(model(), m, k, GRID));

  // ---------- ③ feature importance ----------

  function fiView(c) {
    const s = c.state.gbI, rel = M().importance(model(), s.m, NAMES.length);
    return {
      fiM: s.m, fiMText: String(s.m), fiOnM: e => patch(c, { m: Math.round(num(e)) }),
      fiBars: rel.map((v, j) => ({ name: NAMES[j], truth: TRUTH[j], w: Math.max(v, 0.6).toFixed(1), val: f1(v) }))
    };
  }

  // ---------- ④ partial dependence ----------

  const PW = 560, PH = 260, PPL = 36, PPR = 12, PPT = 12, PPB = 24, YLO = -3, YHI = 7;
  const sx = v => PPL + (v / LIM) * (PW - PPL - PPR);
  const sy = v => PH - PPB - ((Math.max(YLO, Math.min(YHI, v)) - YLO) / (YHI - YLO)) * (PH - PPT - PPB);
  const path = vals => vals.map((v, i) => (i ? 'L ' : 'M ') + sx(GRID[i]).toFixed(1) + ' ' + sy(v).toFixed(1)).join(' ');

  // the true effect of x_k, shifted to sit at the same average height as the partial dependence curve
  function truthPath(k, pd) {
    const eff = M().TRUE_EFFECTS[k], shift = M().mean(pd) - M().mean(GRID.map(eff));
    return path(GRID.map(v => eff(v) + shift));
  }

  function pdView(c) {
    const s = c.state.gbI, k = s.feat, { curves, pd } = pdFor(s.m, k);
    const gi = Math.round(s.v / GRID_STEP), at = curves.map(cv => cv[gi]), name = NAMES[k], v = GRID[gi].toFixed(2);
    const firsts = at.slice(0, 3).map((p, i) => 'row ' + (i + 1) + ' → ' + f2(p)).join(', ');
    return {
      pdFeats: NAMES.map((nm, j) => ({
        name: nm, bg: j === k ? 'var(--color-accent)' : 'var(--color-neutral-200)', fg: j === k ? 'var(--color-bg)' : 'var(--color-text)',
        onClick: () => patch(c, { feat: j })
      })),
      pdName: name,
      pdPath: path(pd), pdTruth: truthPath(k, pd),
      pdV: s.v, pdVText: v, pdOnV: e => patch(c, { v: num(e) }),
      pdMarkX: sx(GRID[gi]).toFixed(1), pdMarkY: sy(pd[gi]).toFixed(1),
      pdInspect: 'Set ' + name + ' = ' + v + ' in all ' + at.length + ' training rows, keep their other features, and predict: ' + firsts +
        ', … The average of all ' + at.length + ' predictions is ' + f2(pd[gi]) + '.',
      pdMText: String(s.m)
    };
  }

  function initState() {
    return { gbI: { m: 30, feat: 0, v: 5 } };
  }

  function render(c) {
    return { ...fiView(c), ...pdView(c) };
  }

  return { initState, render };
})();
