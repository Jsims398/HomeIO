// View logic for Topic 14 (choosing C, kernel ridge regression and SVM regression, §17.3.6 – 17.3.8).
// Loaded only by kernel-machines.html. reading-log.js calls KMView.initState()
// for the starting state and KMView.render(component) from t14(); the math is in km-math.js (and kt-math.js).
window.KMView = (function () {
  const M = () => window.KMMath;
  const T = () => window.KTMath;
  const POS = 'var(--color-accent)', NEG = 'var(--color-accent-2-600)';
  const RIDGE = 'var(--color-accent-2-600)', SVR = 'var(--color-accent)';
  const pill = active => ({ bg: active ? 'var(--color-accent)' : 'transparent', fg: active ? 'var(--color-bg)' : 'var(--color-text)' });
  const patch = (c, key, change) => c.setState({ [key]: { ...c.state[key], ...change } });
  const int = e => parseInt(e.target.value, 10);
  const num = e => parseFloat(e.target.value);
  const f1 = v => v.toFixed(1);
  const f2 = v => v.toFixed(2);
  const pct = v => Math.round(v * 100) + '%';
  const short = v => (v >= 1 ? String(v) : String(+v.toPrecision(2)));
  const line = (vals, fx, fy) => vals.map((v, i) => (i ? 'L ' : 'M ') + f1(fx(v, i)) + ' ' + f1(fy(v, i))).join(' ');

  // ---------- part 1: choosing C by cross-validation (§17.3.6) ----------

  const BOX = 300, BOX_LIM = 3, CELLS = 50, CV_SEED = 473, CV_DEFER_MS = 30;
  const C_GRID = [0.03, 0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000];
  const GAMMAS = [[0.3, 'Wide bumps · γ = 0.3'], [1, 'Medium · γ = 1'], [5, 'Narrow · γ = 5']];
  const BUCKETS = [[-Infinity, -1, NEG, 0.34], [-1, 0, NEG, 0.13], [0, 1, POS, 0.13], [1, Infinity, POS, 0.34]];
  const CV_W = 560, CV_H = 220, CV_PL = 44, CV_PR = 16, CV_PT = 14, CV_PB = 36, CV_TOP = 0.6;

  const bx = v => ((v + BOX_LIM) / (2 * BOX_LIM)) * BOX;
  const by = v => BOX - ((v + BOX_LIM) / (2 * BOX_LIM)) * BOX;
  const unbx = p => (p / BOX) * 2 * BOX_LIM - BOX_LIM;
  const unby = p => BOX_LIM - (p / BOX) * 2 * BOX_LIM;
  const cvx = i => CV_PL + (i / (C_GRID.length - 1)) * (CV_W - CV_PL - CV_PR);
  const cvy = e => CV_H - CV_PB - (Math.min(e, CV_TOP) / CV_TOP) * (CV_H - CV_PT - CV_PB);

  // one CV curve is ~50 SMO runs (~1 s), so keep one per (γ, sample); moving the C slider is then free.
  // The first time, return null and run it after the page repaints, then re-render
  const curves = new Map(), pending = new Set();
  function curve(c, gamma, seed) {
    const key = gamma + '|' + seed;
    if (curves.has(key)) return curves.get(key);
    if (!pending.has(key)) {
      pending.add(key);
      setTimeout(() => {
        curves.set(key, M().cvCurve({ type: 'rbf', gamma }, M().classData(seed), C_GRID, CV_SEED));
        pending.delete(key);
        patch(c, 'kmC', {});
      }, CV_DEFER_MS);
    }
    return null;
  }
  let fitCache = { key: null, model: null };
  function fit(gamma, seed, C) {
    const key = [gamma, seed, C].join('|');
    if (fitCache.key !== key) fitCache = { key, model: T().trainSvm({ type: 'rbf', gamma }, M().classData(seed), C, CV_SEED) };
    return fitCache.model;
  }

  // one path per bucket instead of 2,500 rects
  function heat(p, m) {
    const size = BOX / CELLS, step = f1(size + 0.4), paths = BUCKETS.map(() => []);
    for (let i = 0; i < CELLS; i++) {
      for (let j = 0; j < CELLS; j++) {
        const x0 = i * size, y0 = j * size, f = T().decision(p, m, [unbx(x0 + size / 2), unby(y0 + size / 2)]);
        paths[BUCKETS.findIndex(([lo, hi]) => f >= lo && f < hi)].push('M' + f1(x0) + ' ' + f1(y0) + 'h' + step + 'v' + step + 'h-' + step + 'z');
      }
    }
    return BUCKETS.map(([, , color, opacity], k) => ({ d: paths[k].join(''), color, opacity }));
  }

  function cvNote(i, best, rows) {
    const r = rows[i];
    if (r.train > 0.35) return 'C is so small that the SVM barely cares about mistakes: it would rather keep a wide margin than fit anything, so it calls almost everything one class. Underfitting.';
    if (i === best) return 'This C has the lowest cross-validation error: the best guess for how this SVM will do on new points. Training error is not zero, and that is fine; a few labels are noise.';
    if (i < best) return 'Getting better: a larger C lets the boundary follow the wave more closely. Keep going until the cross-validation error stops dropping.';
    if (r.cv - r.train > 0.15) return 'Training error keeps falling as C grows, but cross-validation error has turned back up. The boundary is bending around noisy points that new data will not repeat. Overfitting.';
    return 'Past the best C. Once C is big enough to fit (nearly) every training point, raising it further changes little, so the curves go flat.';
  }

  // everything that needs the CV curve; placeholders while it is still running
  function cvParts(c, s, rows) {
    if (!rows) {
      return { cTrain: '…', cCv: '…', cBestLabel: '…', cBestCv: '…', cTrainPath: '', cCvPath: '', cPoints: [], cBestX: -50, cBestY: -50,
        cNote: 'Running 5-fold cross-validation: 10 values of C × 5 folds = 50 SVMs…' };
    }
    const best = rows.reduce((b, r, i) => (r.cv < rows[b].cv ? i : b), 0), r = rows[s.cIdx];
    return {
      cTrain: pct(r.train), cCv: pct(r.cv), cBestLabel: short(C_GRID[best]), cBestCv: pct(rows[best].cv),
      cOnBest: () => patch(c, 'kmC', { cIdx: best }),
      cTrainPath: line(rows, (_, i) => cvx(i), v => cvy(v.train)),
      cCvPath: line(rows, (_, i) => cvx(i), v => cvy(v.cv)),
      cPoints: rows.map((v, i) => ({ x: f1(cvx(i)), yT: f1(cvy(v.train)), yC: f1(cvy(v.cv)), r: i === s.cIdx ? 7 : 4.5, onClick: () => patch(c, 'kmC', { cIdx: i }) })),
      cBestX: f1(cvx(best)), cBestY: f1(cvy(rows[best].cv)),
      cNote: cvNote(s.cIdx, best, rows)
    };
  }

  function chooseC(c) {
    const s = c.state.kmC, p = { type: 'rbf', gamma: s.gamma }, pts = M().classData(s.seed);
    const m = fit(s.gamma, s.seed, C_GRID[s.cIdx]);
    const wave = Array.from({ length: 61 }, (_, i) => -BOX_LIM + i / 10);
    return {
      cOnBest: () => {},
      ...cvParts(c, s, curve(c, s.gamma, s.seed)),
      cHeat: heat(p, m),
      cWave: line(wave, x => bx(x), x => by(0.8 * Math.sin(1.5 * x))),
      cDots: pts.map(q => ({ x: f1(bx(q[0])), y: f1(by(q[1])), fill: q[2] > 0 ? POS : NEG })),
      cNoisy: pts.filter(q => M().trueSide(q[0], q[1]) !== q[2]).map(q => ({ x: f1(bx(q[0])), y: f1(by(q[1])) })),
      cSv: m.sv.map(n => ({ x: f1(bx(m.X[n][0])), y: f1(by(m.X[n][1])) })),
      cGammaBtns: GAMMAS.map(([g, label]) => ({ label, ...pill(g === s.gamma), onClick: () => patch(c, 'kmC', { gamma: g }) })),
      cOnResample: () => patch(c, 'kmC', { seed: s.seed + 1 }),
      cIdx: s.cIdx, cLabel: short(C_GRID[s.cIdx]), cOnC: e => patch(c, 'kmC', { cIdx: int(e) }),
      cSvCount: String(m.sv.length), cCount: String(pts.length),
      cMarkX: f1(cvx(s.cIdx)),
      cXTicks: C_GRID.map((v, i) => ({ x: f1(cvx(i)), label: short(v) })),
      cYTicks: [0, 0.2, 0.4, 0.6].map(v => ({ y: f1(cvy(v)), label: pct(v) }))
    };
  }

  // ---------- part 2: kernel ridge vs SVM regression (§17.3.7 – 17.3.8) ----------

  const RW = 560, RH = 260, RPL = 34, RPR = 14, RPT = 14, RPB = 28, RX = 3, RY = 2.8;
  const WROW = 52, WGAP = 14, GRAB = 10, REG_MAX = 50, OUTLIER_GAP = 1.5;
  const OUTLIER = [1.1, -2.4];
  const LAMBDAS = [0.001, 0.01, 0.1, 1, 10];
  const SVR_CS = [0.1, 0.3, 1, 3, 10, 100];
  const SHOW = { ridge: 'Kernel ridge', svr: 'SVM regression', both: 'Both' };
  const GRID = Array.from({ length: 121 }, (_, i) => -RX + i / 20);

  const rx = x => RPL + ((x + RX) / (2 * RX)) * (RW - RPL - RPR);
  const ry = y => RH - RPB - ((Math.max(-RY, Math.min(RY, y)) + RY) / (2 * RY)) * (RH - RPT - RPB);
  const unrx = p => ((p - RPL) / (RW - RPL - RPR)) * 2 * RX - RX;
  const unry = p => ((RH - RPB - p) / (RH - RPT - RPB)) * 2 * RY - RY;

  // one row of weight bars, centred on `mid`: height ∝ |weight|, a flat grey tick where the weight is exactly 0
  function bars(model, mid, color) {
    const peak = Math.max(...model.alpha.map(Math.abs), M().ZERO);
    return model.alpha.map((a, n) => {
      const zero = Math.abs(a) <= M().ZERO, h = zero ? 1.5 : Math.max((Math.abs(a) / peak) * (WROW / 2 - 2), 1.5);
      return { x: f1(rx(model.xs[n]) - 3), y: f1(a >= 0 ? mid - h : mid), h: f1(h), fill: zero ? 'var(--color-neutral-400)' : color };
    });
  }

  function regClick(c, s, e) {
    const r = e.currentTarget.getBoundingClientRect();
    const cx = ((e.clientX - r.left) / r.width) * RW, cy = ((e.clientY - r.top) / r.height) * RH;
    const near = s.pts.findIndex(q => Math.hypot(rx(q[0]) - cx, ry(q[1]) - cy) < GRAB);
    if (near >= 0) { patch(c, 'kmReg', { pts: s.pts.filter((_, i) => i !== near) }); return; }
    const x = unrx(cx);
    if (s.pts.length >= REG_MAX || x < -RX || x > RX) return;
    patch(c, 'kmReg', { pts: [...s.pts, [x, unry(cy)]] });
  }

  function regNote(s, nR, nS, N, hasOutlier) {
    if (N < 2) return 'Add a few points: click anywhere on the plot.';
    if (hasOutlier && s.show !== 'ridge') return 'Squared loss grows like (error)², so ridge bends hard toward the outlier. SVR\'s loss grows only linearly outside the tube and each weight is capped at C, so one bad point can only pull so far. Try a smaller C.';
    if (s.show === 'ridge') return 'Kernel ridge uses squared loss, so every point that is not fitted exactly pushes on the curve. All ' + nR + ' weights are nonzero: to predict, you need K between x and every training point. λ trades smoothness for fit.';
    if (s.eps < 0.05) return 'With a tube this thin, almost every point sits outside it, so almost every point is a support vector (' + nS + ' of ' + N + '). SVR now behaves a lot like ridge.';
    if (nS <= 4) return 'The tube is so wide that only ' + nS + ' points touch it. Very sparse, but the fit is now too flat to follow the wave: ε is a bias-variance knob too.';
    return 'Points inside the tube cost nothing, so their weights are exactly 0 (grey ticks below). Only the ' + nS + ' points on or outside the tube, the support vectors, are needed to predict. Ridge needs all ' + nR + '.';
  }

  function regression(c) {
    const s = c.state.kmReg, D = M(), lambda = LAMBDAS[s.lamIdx], C = SVR_CS[s.cIdx], N = s.pts.length;
    const ridge = D.ridge(s.gamma, s.pts, lambda), svr = D.svr(s.gamma, s.pts, s.eps, C);
    const fitR = GRID.map(x => D.predict(s.gamma, ridge, x)), fitS = GRID.map(x => D.predict(s.gamma, svr, x));
    const path = ys => line(GRID, x => rx(x), (_, i) => ry(ys[i]));
    const back = [...GRID].reverse().map((x, i) => 'L ' + f1(rx(x)) + ' ' + f1(ry(fitS[GRID.length - 1 - i] - s.eps))).join(' ');
    const nR = D.active(ridge), nS = D.active(svr);
    const hasOutlier = s.pts.some(q => Math.abs(q[1] - D.truth(q[0])) > OUTLIER_GAP);
    return {
      rShowRidge: s.show !== 'svr', rShowSvr: s.show !== 'ridge',
      rTruth: line(GRID, x => rx(x), x => ry(D.truth(x))),
      rRidgePath: N ? path(fitR) : '', rSvrPath: N ? path(fitS) : '',
      rTube: N ? path(fitS.map(v => v + s.eps)) + ' ' + back + ' Z' : '',
      rDots: s.pts.map(q => ({ x: f1(rx(q[0])), y: f1(ry(q[1])) })),
      rSv: s.show !== 'ridge' ? s.pts.filter((_, n) => Math.abs(svr.alpha[n]) > D.ZERO).map(q => ({ x: f1(rx(q[0])), y: f1(ry(q[1])) })) : [],
      rZeroY: f1(ry(0)),
      rXTicks: [-3, -2, -1, 0, 1, 2, 3].map(v => ({ x: f1(rx(v)), label: String(v) })),
      rOnDown: e => regClick(c, s, e),
      rRidgeBars: bars(ridge, 4 + WROW / 2, RIDGE), rSvrBars: bars(svr, 4 + WROW + WGAP + WROW / 2, SVR),
      rShowBtns: Object.keys(SHOW).map(k => ({ label: SHOW[k], ...pill(k === s.show), onClick: () => patch(c, 'kmReg', { show: k }) })),
      rGamma: s.gamma, rGammaLabel: f2(s.gamma), rOnGamma: e => patch(c, 'kmReg', { gamma: num(e) }),
      rLam: s.lamIdx, rLamMax: LAMBDAS.length - 1, rLamLabel: short(lambda), rOnLam: e => patch(c, 'kmReg', { lamIdx: int(e) }),
      rEps: s.eps, rEpsLabel: f2(s.eps), rOnEps: e => patch(c, 'kmReg', { eps: num(e) }),
      rC: s.cIdx, rCMax: SVR_CS.length - 1, rCLabel: short(C), rOnC: e => patch(c, 'kmReg', { cIdx: int(e) }),
      rOnOutlier: () => patch(c, 'kmReg', { pts: [...s.pts, OUTLIER] }),
      rOnResample: () => patch(c, 'kmReg', { seed: s.seed + 1, pts: D.regData(s.seed + 1) }),
      rCount: String(N), rRidgeActive: String(nR), rSvrActive: String(nS),
      rRidgeMse: N ? f2(D.truthMse(s.gamma, ridge, GRID)) : '—', rSvrMse: N ? f2(D.truthMse(s.gamma, svr, GRID)) : '—',
      rNote: regNote(s, nR, nS, N, hasOutlier)
    };
  }

  // ---------- state ----------

  function initState() {
    return {
      kmC: { gamma: 1, seed: 14, cIdx: 0 },
      kmReg: { seed: 14, pts: M().regData(14), gamma: 1, lamIdx: 2, eps: 0.3, cIdx: 4, show: 'both' }
    };
  }

  function render(c) {
    return { ...chooseC(c), ...regression(c) };
  }

  return { initState, render };
})();
