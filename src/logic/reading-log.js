// Shared logic for every Reading Log page. Each page's inline
// <script data-dc-script> calls createReadingLog(DCLogic, <topic number>).
window.createReadingLog = function (DCLogic, topic) {
const PAGES = { 1: 'probability.html', 2: 'distributions.html', 3: 'mahalanobis.html', 4: 'bias-variance.html', 5: 'roc.html' };

class Component extends DCLogic {
  state = { car: null, picked: null, opened: null, phase: 'pick', choice: null, won: null, round: 1,
            tally: { stayPlays: 0, stayWins: 0, switchPlays: 0, switchWins: 0 },
            topic, theta: 0.35, logits: [2.2, 1.1, -0.4], temp: 1, n: 12, samples: null,
            s1: 1.5, s2: 0.8, rho: 0.6, ptx: 1.7, pty: 1.1, cloud: null,
            deg: 3, nTrain: 12, noise: 0.28, sets: topic === 4 ? this.makeSets(12, 0.28) : null,
            cm: { tp: '40', fn: '10', fp: '15', tn: '35' }, rocD: null, rocT: null };

  goTopic(n) { if (n !== this.state.topic) location.href = PAGES[n]; }

  componentDidMount() { this.draw(12); this.cloud(); }

  cloud() {
    const pts = [];
    for (let i = 0; i < 90; i++) pts.push([this.gauss(0, 1), this.gauss(0, 1)]);
    this.setState({ cloud: pts });
  }

  makeSets(n, sigma) {
    const M = 16, sets = [];
    for (let m = 0; m < M; m++) {
      const pts = [];
      for (let i = 0; i < n; i++) {
        const x = Math.random();
        pts.push([x, Math.sin(2 * Math.PI * x) + this.gauss(0, sigma)]);
      }
      sets.push(pts);
    }
    return sets;
  }

  polyfit(pts, deg) {
    const k = deg + 1, A = [], b = [];
    for (let i = 0; i < k; i++) { A.push(new Array(k).fill(0)); b.push(0); }
    pts.forEach(([x, y]) => {
      const ph = []; let v = 1;
      for (let j = 0; j < k; j++) { ph.push(v); v *= x; }
      for (let i = 0; i < k; i++) { b[i] += ph[i] * y; for (let j = 0; j < k; j++) A[i][j] += ph[i] * ph[j]; }
    });
    for (let i = 0; i < k; i++) A[i][i] += 1e-6;
    // gaussian elimination with partial pivoting
    for (let c = 0; c < k; c++) {
      let p = c;
      for (let r = c + 1; r < k; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
      [A[c], A[p]] = [A[p], A[c]]; [b[c], b[p]] = [b[p], b[c]];
      if (Math.abs(A[c][c]) < 1e-12) continue;
      for (let r = c + 1; r < k; r++) {
        const f = A[r][c] / A[c][c];
        for (let j = c; j < k; j++) A[r][j] -= f * A[c][j];
        b[r] -= f * b[c];
      }
    }
    const w = new Array(k).fill(0);
    for (let r = k - 1; r >= 0; r--) {
      let acc = b[r];
      for (let j = r + 1; j < k; j++) acc -= A[r][j] * w[j];
      w[r] = Math.abs(A[r][r]) < 1e-12 ? 0 : acc / A[r][r];
    }
    return w;
  }

  evalPoly(w, x) { let v = 0, p = 1; for (let i = 0; i < w.length; i++) { v += w[i] * p; p *= x; } return v; }

  bvStats(sets, deg, grid) {
    const fits = sets.map(pts => this.polyfit(pts, deg));
    const preds = fits.map(w => grid.map(x => this.evalPoly(w, x)));
    const M = fits.length;
    let bias = 0, varr = 0;
    grid.forEach((x, gi) => {
      const avg = preds.reduce((a, p) => a + p[gi], 0) / M;
      bias += (avg - Math.sin(2 * Math.PI * x)) ** 2;
      varr += preds.reduce((a, p) => a + (p[gi] - avg) ** 2, 0) / M;
    });
    return { bias: bias / grid.length, varr: varr / grid.length, preds, fits };
  }


  ncdf(x) {
    const z = Math.abs(x) / Math.SQRT2, t = 1 / (1 + 0.3275911 * z);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-z * z);
    return x >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y);
  }

  ninv(p) {
    const a = [-39.69683028665376, 220.9460984245205, -275.9285104469687, 138.3577518672690, -30.66479806614716, 2.506628277459239];
    const b = [-54.47609879822406, 161.5858368580409, -155.6989798598866, 66.80131188771972, -13.28068155288572];
    const c = [-0.007784894002430293, -0.3223964580411365, -2.400758277161838, -2.549732539343734, 4.374664141464968, 2.938163982698783];
    const d = [0.007784695709041462, 0.3224671290700398, 2.445134137142996, 3.754408661907416];
    const lo = 0.02425;
    if (p < lo) { const q = Math.sqrt(-2 * Math.log(p)); return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1); }
    if (p > 1 - lo) { const q = Math.sqrt(-2 * Math.log(1 - p)); return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1); }
    const q = p - 0.5, r = q * q;
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
  }

  t5() {
    const s = this.state, on = s.topic === 5;
    const nav = {
      showT5: on,
      goT5: () => this.goTopic(5),
      t5Bg: on ? 'var(--color-accent)' : 'transparent',
      t5Fg: on ? 'var(--color-bg)' : 'var(--color-text)'
    };
    if (!on) return nav;

    const cm = s.cm || { tp: '40', fn: '10', fp: '15', tn: '35' };
    const num = k => { const v = parseInt(cm[k], 10); return isFinite(v) && v > 0 ? v : 0; };
    const TP = num('tp'), FN = num('fn'), FP = num('fp'), TN = num('tn');
    const P = TP + FN, N = FP + TN, ok = P > 0 && N > 0;
    const tpr = P ? TP / P : 0, fpr = N ? FP / N : 0;

    // equal-variance binormal fit through the observed point
    const clamp = (r, n) => Math.min(Math.max(r, 0.5 / n), 1 - 0.5 / n);
    let dFit = 0, tFit = 0;
    if (ok) { const zt = this.ninv(clamp(tpr, P)), zf = this.ninv(clamp(fpr, N)); dFit = zt - zf; tFit = -zf; }
    const d = Math.max(-5, Math.min(5, s.rocD != null ? s.rocD : dFit));
    const lo = Math.min(0, d) - 3.5, hi = Math.max(0, d) + 3.5;
    const thr = Math.max(lo, Math.min(hi, s.rocT != null ? s.rocT : tFit));

    const set = next => this.setState({ cm: { ...cm, ...next }, rocD: null, rocT: null });
    const edit = k => e => set({ [k]: String(e.target.value).replace(/[^0-9]/g, '').slice(0, 6) });
    const bump = (k, dl) => () => set({ [k]: String(Math.max(0, num(k) + dl)) });

    // ROC plot
    const RX = f => 46 + f * 228, RY = t => 246 - t * 228;
    const pt = (f, t) => RX(f).toFixed(1) + ' ' + RY(t).toFixed(1);
    let curve = 'M ' + pt(0, 0);
    for (let i = 0; i <= 160; i++) {
      const t = (hi + 3) - ((hi - lo + 6) * i) / 160;
      curve += ' L ' + pt(1 - this.ncdf(t), 1 - this.ncdf(t - d));
    }
    curve += ' L ' + pt(1, 1);
    const auc = this.ncdf(d / Math.SQRT2);
    const eer = 1 - this.ncdf(d / 2);

    // score distributions
    const DX = v => 10 + ((v - lo) / (hi - lo)) * 320, DY = p => 132 - (p / 0.4) * 112;
    const pdf = (v, m) => Math.exp(-0.5 * (v - m) * (v - m)) / Math.sqrt(2 * Math.PI);
    const line = m => { let p = ''; for (let i = 0; i <= 100; i++) { const v = lo + ((hi - lo) * i) / 100; p += (i ? ' L ' : 'M ') + DX(v).toFixed(1) + ' ' + DY(pdf(v, m)).toFixed(1); } return p; };
    const tail = m => { let p = 'M ' + DX(thr).toFixed(1) + ' 132'; for (let i = 0; i <= 60; i++) { const v = thr + ((hi - thr) * i) / 60; p += ' L ' + DX(v).toFixed(1) + ' ' + DY(pdf(v, m)).toFixed(1); } return p + ' L ' + DX(hi).toFixed(1) + ' 132 Z'; };

    const onThr = e => {
      if (!ok) return;
      const t = parseFloat(e.target.value);
      const tp = Math.round(P * (1 - this.ncdf(t - d))), fp = Math.round(N * (1 - this.ncdf(t)));
      this.setState({ cm: { tp: String(tp), fn: String(P - tp), fp: String(fp), tn: String(N - fp) }, rocD: d, rocT: t });
    };

    const fmt = (a, b) => (b ? (a / b).toFixed(3) : '—');
    const pct = (a, b) => (b ? (100 * a / b).toFixed(1) : '0');
    const rateBars = [
      { name: 'TPR', alias: 'recall', frac: TP + ' / ' + P, val: fmt(TP, P), pct: pct(TP, P), color: 'var(--color-accent)' },
      { name: 'FNR', alias: 'miss rate', frac: FN + ' / ' + P, val: fmt(FN, P), pct: pct(FN, P), color: 'var(--color-accent-300)' },
      { name: 'FPR', alias: 'false alarms', frac: FP + ' / ' + N, val: fmt(FP, N), pct: pct(FP, N), color: 'var(--color-neutral-600)' },
      { name: 'TNR', alias: 'specificity', frac: TN + ' / ' + N, val: fmt(TN, N), pct: pct(TN, N), color: 'var(--color-accent-2-500)' }
    ];

    let verdict;
    if (!ok) verdict = P === 0 && N === 0 ? 'Enter some counts to get started.'
      : P === 0 ? 'There are no actual positives, so TPR and FNR are undefined. Put something in the top row.'
      : 'There are no actual negatives, so FPR and TNR are undefined. Put something in the bottom row.';
    else if (tpr === 1 && fpr === 0) verdict = "Top-left corner: every positive caught, no false alarms. Nothing on the ROC plot beats this.";
    else if (tpr - fpr < -0.02) verdict = 'Below the diagonal: worse than guessing. Flip every prediction and this point moves to (' + (1 - fpr).toFixed(2) + ', ' + (1 - tpr).toFixed(2) + ').';
    else if (Math.abs(tpr - fpr) <= 0.02) verdict = 'On the diagonal: TPR ≈ FPR, so a "+" is just as likely for a negative as for a positive. A biased coin would do the same.';
    else verdict = "Your table sits at (FPR " + fpr.toFixed(2) + ", TPR " + tpr.toFixed(2) + "), " + (tpr - fpr).toFixed(2) + " above the diagonal (Youden's J). Slide the threshold to trade misses for false alarms.";

    const preset = (label, a, b, c, e) => ({ label, onClick: () => set({ tp: String(a), fn: String(b), fp: String(c), tn: String(e) }) });

    return {
      ...nav,
      tpVal: cm.tp, fnVal: cm.fn, fpVal: cm.fp, tnVal: cm.tn,
      onTP: edit('tp'), onFN: edit('fn'), onFP: edit('fp'), onTN: edit('tn'),
      tpInc: bump('tp', 1), tpDec: bump('tp', -1), fnInc: bump('fn', 1), fnDec: bump('fn', -1),
      fpInc: bump('fp', 1), fpDec: bump('fp', -1), tnInc: bump('tn', 1), tnDec: bump('tn', -1),
      pTot: P, nTot: N, predPos: TP + FP, predNeg: FN + TN,
      presets: [preset('Example', 40, 10, 15, 35), preset('Perfect', 50, 0, 0, 50), preset('Coin flip', 25, 25, 25, 25), preset('Always +', 50, 0, 50, 0), preset('Always −', 0, 50, 0, 50)],
      rateBars, verdict,
      accLabel: fmt(TP + TN, P + N), precLabel: fmt(TP, TP + FP),
      tprShort: fmt(TP, P), fprShort: fmt(FP, N),
      rocCurve: curve, aucArea: curve + ' L ' + pt(1, 0) + ' Z',
      showPt: ok, ptX: RX(fpr).toFixed(1), ptY: RY(tpr).toFixed(1),
      polyPath: 'M ' + pt(0, 0) + ' L ' + pt(fpr, tpr) + ' L ' + pt(1, 1),
      eerX: RX(eer).toFixed(1), eerY: RY(1 - eer).toFixed(1),
      aucLabel: auc.toFixed(3), dLabel: d.toFixed(2), eerLabel: (eer * 100).toFixed(1) + '%',
      pAucLabel: ok ? ((1 + tpr - fpr) / 2).toFixed(3) : '—',
      thr, thrMin: lo.toFixed(2), thrMax: hi.toFixed(2), thrLabel: thr.toFixed(2), onThr,
      lockNote: s.rocD != null
        ? "Curve held at d′ = " + d.toFixed(2) + " while you slide (P and N stay fixed). Edit any cell to refit."
        : 'Curve fitted through your point, assuming Gaussian scores with equal spread.',
      negLine: line(0), posLine: line(d), negTail: tail(0), posTail: tail(d), thrX: DX(thr).toFixed(1),
      onImbalance: () => set({ fp: String(FP * 10), tn: String(TN * 10) })
    };
  }

  t4() {
    const s = this.state;
    if (s.topic !== 4) {
      return {
        showT4: false,
        goT4: () => this.goTopic(4),
        t4Bg: 'transparent', t4Fg: 'var(--color-text)'
      };
    }
    const sets = s.sets || this.makeSets(s.nTrain, s.noise);
    const grid = []; for (let i = 0; i <= 48; i++) grid.push(i / 48);
    const X = x => 10 + x * 320, Y = y => 100 - y * 48;

    const cur = this.bvStats(sets, s.deg, grid);
    const M = sets.length;
    const path = arr => arr.map((y, i) => (i ? 'L ' : 'M ') + X(grid[i]).toFixed(1) + ' ' + Y(Math.max(-1.9, Math.min(1.9, y))).toFixed(1)).join(' ');
    const avgArr = grid.map((x, gi) => cur.preds.reduce((a, p) => a + p[gi], 0) / M);

    const sweep = [];
    for (let d = 0; d <= 9; d++) { const st = this.bvStats(sets, d, grid); sweep.push({ d, b: st.bias, v: st.varr, t: st.bias + st.varr + s.noise * s.noise }); }
    const cap = 1.6;
    const vmax = Math.max(0.15, Math.min(cap, Math.max(...sweep.map(p => p.t))));
    const CX = d => 16 + (d / 9) * 308, CY = v => 155 - Math.min(v / vmax, 1) * 140;
    const curve = key => sweep.map((p, i) => (i ? 'L ' : 'M ') + CX(p.d).toFixed(1) + ' ' + CY(p[key]).toFixed(1)).join(' ');
    const best = sweep.reduce((a, p) => (p.t < a.t ? p : a), sweep[0]);

    const noiseSq = s.noise * s.noise;
    const total = cur.bias + cur.varr + noiseSq;
    const scale = Math.max(cur.bias, cur.varr, noiseSq, 0.05);
    const degNote = s.deg <= 1 ? 'Underfitting: every sample gives nearly the same wrong answer. High bias, low variance.'
      : s.deg <= 4 ? 'About right — flexible enough to bend with the truth, stiff enough to ignore the noise.'
      : 'Overfitting: the fits scatter wildly. Low bias, high variance.';

    return {
      showT4: true,
      goT4: () => this.goTopic(4),
      t4Bg: 'var(--color-accent)', t4Fg: 'var(--color-bg)',
      deg: s.deg, degLabel: s.deg, degNote, nSets: M,
      nTrain: s.nTrain, noise: s.noise, noiseLabel: s.noise.toFixed(2),
      onDeg: e => this.setState({ deg: parseInt(e.target.value, 10) }),
      onN4: e => { const n = parseInt(e.target.value, 10); this.setState({ nTrain: n, sets: this.makeSets(n, s.noise) }); },
      onNoise: e => { const v = parseFloat(e.target.value); this.setState({ noise: v, sets: this.makeSets(s.nTrain, v) }); },
      onNewSets: () => this.setState({ sets: this.makeSets(s.nTrain, s.noise) }),

      truePath: path(grid.map(x => Math.sin(2 * Math.PI * x))),
      avgPath: path(avgArr),
      fitPaths: cur.preds.slice(0, 12).map(p => ({ d: path(p) })),
      shownPts: sets[0].map(([x, y]) => ({ x: X(x).toFixed(1), y: Y(Math.max(-1.9, Math.min(1.9, y))).toFixed(1) })),

      biasLabel: cur.bias.toFixed(3), varLabel: cur.varr.toFixed(3),
      noiseSqLabel: noiseSq.toFixed(3), totalLabel: total.toFixed(3),
      biasPct: Math.min(100, (cur.bias / scale) * 100).toFixed(1),
      varPct: Math.min(100, (cur.varr / scale) * 100).toFixed(1),
      noisePct: Math.min(100, (noiseSq / scale) * 100).toFixed(1),

      biasCurve: curve('b'), varCurve: curve('v'), totalCurve: curve('t'),
      markX: CX(s.deg).toFixed(1), bestDeg: best.d,
      bestX: CX(best.d).toFixed(1), bestY: CY(best.t).toFixed(1)
    };
  }

  t3() {
    const s = this.state, S = 46;
    const s1 = s.s1, s2 = s.s2, rho = s.rho;
    const c11 = s1 * s1, c22 = s2 * s2, c12 = rho * s1 * s2;

    // eigendecomposition of the 2x2 symmetric matrix
    const tr = c11 + c22, det = c11 * c22 - c12 * c12;
    const disc = Math.sqrt(Math.max(tr * tr / 4 - det, 0));
    const l1 = tr / 2 + disc, l2 = tr / 2 - disc;
    const angle = 0.5 * Math.atan2(2 * c12, c11 - c22);
    const deg = -angle * 180 / Math.PI;

    const inv = det > 1e-9
      ? [c22 / det, -c12 / det, c11 / det]
      : [1, 0, 1];
    const dm = Math.sqrt(Math.max(inv[0] * s.ptx * s.ptx + 2 * inv[1] * s.ptx * s.pty + inv[2] * s.pty * s.pty, 0));
    const eu = Math.sqrt(s.ptx * s.ptx + s.pty * s.pty);

    const verdict = dm < 1 ? 'Well inside the cloud — nothing surprising about this point.'
      : dm < 2 ? 'Ordinary. Roughly the middle band of the distribution.'
      : dm < 3 ? 'Getting unusual — past the 2σ ring.'
      : 'Outlier territory. Note how far this can be from the centre in one direction and how close in another.';

    // rotate a unit-variance cloud by Σ^{1/2} = U Λ^{1/2} U^T
    const ca = Math.cos(angle), sa = Math.sin(angle);
    const r1 = Math.sqrt(Math.max(l1, 0)), r2 = Math.sqrt(Math.max(l2, 0));
    const samplePts = (s.cloud || []).map(([a, b]) => {
      const u = r1 * a, v = r2 * b;
      return { x: (160 + (ca * u - sa * v) * S).toFixed(1), y: (160 - (sa * u + ca * v) * S).toFixed(1) };
    });

    return {
      showT3: s.topic === 3,
      goT3: () => this.goTopic(3),
      t3Bg: s.topic === 3 ? 'var(--color-accent)' : 'transparent',
      t3Fg: s.topic === 3 ? 'var(--color-bg)' : 'var(--color-text)',

      s1, s2, rho,
      s1Label: s1.toFixed(2), s2Label: s2.toFixed(2), rhoLabel: rho.toFixed(2),
      onS1: e => this.setState({ s1: parseFloat(e.target.value) }),
      onS2: e => this.setState({ s2: parseFloat(e.target.value) }),
      onRho: e => this.setState({ rho: parseFloat(e.target.value) }),
      onIso: () => this.setState({ s1: 1, s2: 1, rho: 0 }),
      onResample3: () => this.cloud(),

      c11: c11.toFixed(2), c22: c22.toFixed(2), c12: c12.toFixed(2),
      lam1: r1.toFixed(2), lam2: r2.toFixed(2), angleLabel: deg.toFixed(0),

      px: (160 + s.ptx * S).toFixed(1), py: (160 - s.pty * S).toFixed(1),
      euclid: eu.toFixed(2), maha: dm.toFixed(2), verdict,
      onSvgMove: e => {
        if (e.buttons === 0 && e.type === 'mousemove') return;
        const r = e.currentTarget.getBoundingClientRect();
        const x = ((e.clientX - r.left) / r.width) * 320, y = ((e.clientY - r.top) / r.height) * 320;
        this.setState({ ptx: (x - 160) / S, pty: (160 - y) / S });
      },

      euclidRings: [1, 2, 3].map(k => ({ r: (k * S).toFixed(1) })),
      ellipses: [1, 2, 3].map(k => ({
        rx: (k * r1 * S).toFixed(1), ry: (k * r2 * S).toFixed(1),
        rot: 'rotate(' + deg.toFixed(2) + ' 160 160)',
        w: k === 1 ? 3 : 2, op: k === 1 ? 1 : 0.55
      })),
      samplePts
    };
  }

  gauss(mu, sd) {
    const u = 1 - Math.random(), v = Math.random();
    return mu + sd * Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }
  draw(n) {
    const s = [];
    for (let i = 0; i < n; i++) s.push(this.gauss(this.state.mu ?? 0, this.state.sigma ?? 1));
    this.setState({ samples: s, n });
  }
  X(x) { return ((x + 6) / 12) * 340; }
  phi(x) {
    const t = 1 / (1 + 0.2316419 * Math.abs(x) / Math.SQRT2);
    const p = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    const c = 1 - p * Math.exp(-x * x / 2) / Math.sqrt(2 * Math.PI);
    return x >= 0 ? c : 1 - c;
  }

  t2() {
    const s = this.state;
    const mu = s.mu ?? 0, sigma = s.sigma ?? 1;
    const theta = s.theta, T = s.temp;
    const labels = ['cat', 'dog', 'fox'];
    const colors = ['var(--color-accent)', 'var(--color-accent-2-500)', 'var(--color-neutral-600)'];

    const scaled = s.logits.map(z => z / T);
    const m = Math.max(...scaled);
    const ex = scaled.map(z => Math.exp(z - m));
    const tot = ex.reduce((a, b) => a + b, 0);
    const probs = ex.map(e => e / tot);

    const pdf = x => Math.exp(-((x - mu) ** 2) / (2 * sigma * sigma)) / (sigma * Math.sqrt(2 * Math.PI));
    const Y = p => 160 - Math.min(p / 1.05, 1) * 136;
    let gaussPath = '';
    for (let i = 0; i <= 90; i++) {
      const x = -6 + (12 * i) / 90;
      gaussPath += (i ? ' L ' : 'M ') + this.X(x).toFixed(1) + ' ' + Y(pdf(x)).toFixed(1);
    }
    let bandPath = 'M ' + this.X(mu - sigma).toFixed(1) + ' 160';
    for (let i = 0; i <= 40; i++) {
      const x = mu - sigma + (2 * sigma * i) / 40;
      bandPath += ' L ' + this.X(x).toFixed(1) + ' ' + Y(pdf(x)).toFixed(1);
    }
    bandPath += ' L ' + this.X(mu + sigma).toFixed(1) + ' 160 Z';

    const samples = (s.samples || []).slice().sort((a, b) => a - b);
    const N = samples.length || 1;
    const CY = c => 140 - c * 118;
    let empCdfPath = 'M 0 ' + CY(0);
    samples.forEach((v, i) => {
      const x = this.X(v).toFixed(1);
      empCdfPath += ' L ' + x + ' ' + CY(i / N) + ' L ' + x + ' ' + CY((i + 1) / N);
    });
    empCdfPath += ' L 340 ' + CY(1);
    let trueCdfPath = '';
    for (let i = 0; i <= 90; i++) {
      const x = -6 + (12 * i) / 90;
      trueCdfPath += (i ? ' L ' : 'M ') + this.X(x).toFixed(1) + ' ' + CY(this.phi((x - mu) / sigma)).toFixed(1);
    }

    const tempNote = T < 0.5 ? 'Cold: the argmax takes almost everything.'
      : T > 2.4 ? 'Hot: the logits barely matter, it is drifting toward 1/3 each.'
      : 'T = 1 is the plain softmax — logit gaps translate straight into odds ratios.';

    return {
      showT1: s.topic === 1, showT2: s.topic === 2,
      goT1: () => this.goTopic(1), goT2: () => this.goTopic(2),
      t1Bg: s.topic === 1 ? 'var(--color-accent)' : 'transparent',
      t1Fg: s.topic === 1 ? 'var(--color-bg)' : 'var(--color-text)',
      t2Bg: s.topic === 2 ? 'var(--color-accent)' : 'transparent',
      t2Fg: s.topic === 2 ? 'var(--color-bg)' : 'var(--color-text)',

      theta, thetaLabel: theta.toFixed(2), bernVar: (theta * (1 - theta)).toFixed(3),
      onTheta: e => this.setState({ theta: parseFloat(e.target.value) }),
      bernBars: [
        { y: 0, val: (1 - theta).toFixed(2), h: Math.round(4 + (1 - theta) * 150), color: 'var(--color-neutral-600)' },
        { y: 1, val: theta.toFixed(2), h: Math.round(4 + theta * 150), color: 'var(--color-accent)' }
      ],

      logitCtrls: s.logits.map((z, i) => ({
        label: labels[i], z, val: z.toFixed(1),
        onChange: e => { const L = s.logits.slice(); L[i] = parseFloat(e.target.value); this.setState({ logits: L }); }
      })),
      softBars: probs.map((p, i) => ({ label: labels[i], color: colors[i], pct: (p * 100).toFixed(1), pctLabel: (p * 100).toFixed(1) + '%' })),
      softSum: probs.reduce((a, b) => a + b, 0).toFixed(2),
      temp: T, tempLabel: T.toFixed(2), tempNote,
      onTemp: e => this.setState({ temp: parseFloat(e.target.value) }),
      onSoftReset: () => this.setState({ logits: [2.2, 1.1, -0.4], temp: 1 }),

      mu, sigma, muLabel: mu.toFixed(1), sigmaLabel: sigma.toFixed(2), muX: this.X(mu).toFixed(1),
      gaussPath, bandPath, trueCdfPath, empCdfPath,
      onMu: e => this.setState({ mu: parseFloat(e.target.value) }),
      onSigma: e => this.setState({ sigma: parseFloat(e.target.value) }),

      nLabel: samples.length, sampleTicks: samples.map(v => ({ x: this.X(v).toFixed(1) })),
      onResample: () => this.draw(s.n),
      onMoreN: () => this.draw(Math.min(s.n * 2, 400)),
      onFewN: () => this.draw(Math.max(3, Math.round(s.n / 2)))
    };
  }

  pick(i) {
    if (this.state.phase !== 'pick') return;
    const car = Math.floor(Math.random() * 3);
    const opts = [0, 1, 2].filter(d => d !== i && d !== car);
    const opened = opts[Math.floor(Math.random() * opts.length)];
    this.setState({ car, picked: i, opened, phase: 'decide' });
  }

  decide(kind) {
    const { picked, opened, car, tally } = this.state;
    const other = [0, 1, 2].find(d => d !== picked && d !== opened);
    const final = kind === 'switch' ? other : picked;
    const won = final === car;
    const t = { ...tally };
    if (kind === 'switch') { t.switchPlays++; if (won) t.switchWins++; }
    else { t.stayPlays++; if (won) t.stayWins++; }
    this.setState({ phase: 'done', choice: kind, won, tally: t });
  }

  pct(w, p) { return p ? Math.round((w / p) * 1000) / 10 : 0; }

  renderVals() {
    const s = this.state, t = s.tally;
    const other = s.picked === null ? null : [0, 1, 2].find(d => d !== s.picked && d !== s.opened);
    const finalPick = s.phase === 'done' ? (s.choice === 'switch' ? other : s.picked) : null;

    const doors = [0, 1, 2].map(i => {
      const isOpen = i === s.opened;
      const revealed = isOpen || s.phase === 'done';
      const isCar = i === s.car;
      const isPick = i === s.picked;
      let bg = 'var(--color-neutral-100)', border = 'var(--color-neutral-400)', note = '', noteColor = 'var(--color-neutral-700)';
      if (isPick) { border = 'var(--color-accent)'; note = s.phase === 'done' && s.choice === 'stay' ? 'Your final answer' : 'Your first pick'; noteColor = 'var(--color-accent-700)'; }
      if (isOpen) { bg = 'var(--color-neutral-300)'; border = 'var(--color-neutral-400)'; note = 'Opened by the host'; noteColor = 'var(--color-neutral-700)'; }
      if (s.phase === 'done' && i === finalPick && !isPick) { border = 'var(--color-accent)'; note = 'You switched here'; noteColor = 'var(--color-accent-700)'; }
      if (revealed && isCar) { bg = 'var(--color-accent-200)'; }
      if (revealed && isCar && s.phase === 'done') { bg = 'var(--color-accent-300)'; }
      return {
        num: i + 1,
        face: revealed ? (isCar ? 'Car' : 'Goat') : '?',
        faceColor: revealed && isCar ? 'var(--color-accent-800)' : 'var(--color-neutral-700)',
        bg, border, note, noteColor,
        disabled: s.phase === 'done' || isOpen || (s.phase === 'decide' && isPick),
        cursor: s.phase === 'done' || isOpen ? 'default' : 'pointer',
        onClick: s.phase === 'pick' ? () => this.pick(i) : (s.phase === 'decide' && i === other ? () => this.decide('switch') : () => {})
      };
    });

    let status = 'Pick a door. The car is behind one of the three.';
    if (s.phase === 'decide') status = 'The host opened door ' + (s.opened + 1) + ' \u2014 a goat. Door ' + (s.picked + 1) + ' is still yours and door ' + (other + 1) + ' is still closed. Stay or switch?';
    if (s.phase === 'done') status = (s.won ? 'You won the car' : 'A goat. Bad luck') + ' \u2014 you ' + (s.choice === 'switch' ? 'switched to door ' + (finalPick + 1) : 'stayed on door ' + (s.picked + 1)) + ', and the car was behind door ' + (s.car + 1) + '.';

    return {
      ...this.t2(), ...this.t3(), ...this.t4(), ...this.t5(),
      doors, status, round: s.round,
      deciding: s.phase === 'decide', finished: s.phase === 'done',
      pickedNum: s.picked === null ? '' : s.picked + 1,
      otherNum: other === null ? '' : other + 1,
      onStay: () => this.decide('stay'),
      onSwitch: () => this.decide('switch'),
      onReset: () => this.setState({ car: null, picked: null, opened: null, phase: 'pick', choice: null, won: null, round: s.round + 1 }),
      onClear: () => this.setState({ tally: { stayPlays: 0, stayWins: 0, switchPlays: 0, switchWins: 0 }, round: 1, car: null, picked: null, opened: null, phase: 'pick', choice: null, won: null }),
      hasHistory: t.stayPlays + t.switchPlays > 0,
      stayPct: this.pct(t.stayWins, t.stayPlays),
      switchPct: this.pct(t.switchWins, t.switchPlays),
      stayLabel: t.stayPlays ? t.stayWins + ' / ' + t.stayPlays + '  \u00b7  ' + this.pct(t.stayWins, t.stayPlays) + '%' : 'no rounds yet',
      switchLabel: t.switchPlays ? t.switchWins + ' / ' + t.switchPlays + '  \u00b7  ' + this.pct(t.switchWins, t.switchPlays) + '%' : 'no rounds yet',
      showHostRule: this.props.showHostRule ?? true,
      showTally: this.props.showTally ?? true,
      showBayes: this.props.showBayes ?? true
    };
  }
}

return Component;
};
