// View logic for Topic 16 (boosting, §18.5.1 – 18.5.4): step-by-step walkthroughs of least squares
// boosting, AdaBoost and LogitBoost, plus the loss comparison. Loaded only by boosting.html.
// reading-log.js calls BoostView.initState() for the starting state and BoostView.render(component)
// from t16(); the math is in boost-math.js.
window.BoostView = (function () {
  const M = () => window.BoostMath;
  const SEED = 16, LS_ROUNDS = 40, CLS_ROUNDS = 30, SKIP = 5;
  const POS = 'var(--color-accent)', NEG = '#4f6d8a';
  const SUBS = '₀₁₂₃₄₅₆₇₈₉';
  const FEAT = ['x₁', 'x₂'];

  const patch = (c, key, change) => c.setState({ [key]: { ...c.state[key], ...change } });
  const num = e => parseFloat(e.target.value);
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const sub = n => String(n).split('').map(d => SUBS[+d]).join('');
  const f1 = v => v.toFixed(1);
  const fx = d => v => (v < 0 ? '−' : '') + Math.abs(v).toFixed(d);
  const f2 = fx(2), f3 = fx(3);
  const sg = v => (v < 0 ? '−' : '+') + Math.abs(v).toFixed(2);
  const pm = v => (v > 0 ? '+1' : '−1');
  const pct = v => Math.round(v * 100) + '%';
  const argmax = arr => arr.reduce((b, v, i) => (v > arr[b] ? i : b), 0);
  const minMax = arr => [Math.min(...arr), Math.max(...arr)];

  // ---------- shared stepper: k = rounds already finished, p = step inside round k + 1 ----------

  function controls(c, key, nPhases, maxRounds) {
    const s = c.state[key], lastStep = s.p + 1 >= nPhases, lastRound = s.k + 1 >= maxRounds;
    return {
      next: () => {
        if (!lastStep) patch(c, key, { p: s.p + 1 });
        else if (!lastRound) patch(c, key, { k: s.k + 1, p: 0 });
      },
      back: () => {
        if (s.p > 0) patch(c, key, { p: s.p - 1 });
        else if (s.k > 0) patch(c, key, { k: s.k - 1, p: nPhases - 1 });
      },
      skip: () => patch(c, key, { k: Math.min(s.k + SKIP, maxRounds - 1), p: nPhases - 1 }),
      reset: () => patch(c, key, { k: 0, p: 0 }),
      label: 'Round m = ' + (s.k + 1) + ' · step ' + (s.p + 1) + ' of ' + nPhases,
      nextLabel: !lastStep ? 'Next step →' : lastRound ? 'Last round' : 'Start round ' + (s.k + 2) + ' →'
    };
  }

  // the step cards down the side of each walkthrough; steps after the current one are greyed out
  function stepCards(c, key, phases) {
    const p = c.state[key].p;
    return phases.map((ph, i) => ({
      num: 'Step ' + (i + 1), title: ph.title, formula: ph.formula,
      body: i <= p ? ph.body : 'Press Next step to work this out.',
      bg: i === p ? 'var(--color-accent-100)' : 'var(--color-neutral-100)',
      border: i === p ? 'var(--color-accent)' : 'transparent',
      opacity: i > p ? '.5' : '1',
      onClick: () => patch(c, key, { p: i })
    }));
  }

  // bindings every stepper needs, under a prefix: aNext, aBack, aSkip, aReset, aLabel, aNextLabel, aCards
  function stepperBindings(c, key, prefix, phases, maxRounds) {
    const ctl = controls(c, key, phases.length, maxRounds);
    return {
      [prefix + 'Next']: ctl.next, [prefix + 'Back']: ctl.back, [prefix + 'Skip']: ctl.skip, [prefix + 'Reset']: ctl.reset,
      [prefix + 'Label']: ctl.label, [prefix + 'NextLabel']: ctl.nextLabel, [prefix + 'Cards']: stepCards(c, key, phases)
    };
  }

  // the inspector text grows one sentence per step reached
  const upTo = (p, sentences) => sentences.slice(0, p + 1).join(' ');

  // ---------- ② least squares boosting on 1-D data ----------

  const RW = 560, RPL = 36, RPR = 12, LIM = 10;
  const TOP_H = 240, TOP_PT = 12, TOP_PB = 24, YMAX = 7;
  const RES_H = 170, RES_PT = 14, RES_PB = 14, RMAX = 3;
  const lx = x => RPL + (x / LIM) * (RW - RPL - RPR);
  const ly = y => TOP_H - TOP_PB - (y / YMAX) * (TOP_H - TOP_PT - TOP_PB);
  const lr = r => RES_PT + ((RMAX - clamp(r, -RMAX, RMAX)) / (2 * RMAX)) * (RES_H - RES_PT - RES_PB);
  const CURVE_STEP = 0.025;

  function curvePath(fn) {
    const xs = Array.from({ length: Math.round(LIM / CURVE_STEP) + 1 }, (_, i) => i * CURVE_STEP);
    return xs.map((x, i) => (i ? 'L ' : 'M ') + f1(lx(x)) + ' ' + f1(ly(clamp(fn(x), 0, YMAX)))).join(' ');
  }

  function lsPhases(run, st, k, nu, y) {
    const absR = st.r.map(Math.abs), big = argmax(absR), s = st.stump, m = k + 1;
    const sse0 = st.r.reduce((acc, v) => acc + v * v, 0);
    return [
      { title: 'Compute the residuals', formula: 'r_im = y_i − f_{m−1}(x_i)',
        body: (k === 0 ? 'Start from the best constant: f₀ = ȳ = ' + f2(run.f0) + ' everywhere. ' : '') +
          'The vertical sticks are the residuals r_i. Mean |r| = ' + f2(M().mean(absR)) + '; the largest is point #' + (big + 1) + ' with r = ' + sg(st.r[big]) + '.' },
      { title: 'Fit a stump to the residuals', formula: 'F_m = argmin_F Σᵢ (r_im − F(x_i))²',
        body: 'Best stump: x ≤ ' + f2(s.t) + ' → ' + sg(s.a) + ', else ' + sg(s.b) + '. Each side predicts the mean residual on that side. Squared error of the residuals: ' + f2(sse0) + ' → ' + f2(s.sse) + '.' },
      { title: 'Add it, scaled by ν', formula: 'f_m(x) = f_{m−1}(x) + ν F_m(x)',
        body: 'f' + sub(m) + ' = f' + sub(k) + ' + ' + nu.toFixed(1) + ' · F' + sub(m) + '. Training MSE ' + f3(M().mse(st.fBefore, y)) + ' → ' + f3(M().mse(st.fAfter, y)) + '.' }
    ];
  }

  function lsInspect(data, st, i, p, nu, k) {
    const x = data[i].x[0], y = data[i].y, left = x <= st.stump.t;
    return upTo(p, [
      'Point #' + (i + 1) + ': x = ' + f1(x) + ', y = ' + f1(y) + '. The model so far says f' + sub(k) + '(x) = ' + f2(st.fBefore[i]) + ', so r = ' + f1(y) + ' − ' + f2(st.fBefore[i]) + ' = ' + sg(st.r[i]) + '.',
      'It is on the ' + (left ? 'left (x ≤ ' : 'right (x > ') + f2(st.stump.t) + ') side, so F(x) = ' + sg(st.F[i]) + '.',
      'New f = ' + f2(st.fBefore[i]) + ' + ' + nu.toFixed(1) + ' · ' + sg(st.F[i]) + ' = ' + f2(st.fAfter[i]) + ', and its residual becomes ' + sg(st.rAfter[i]) + '.'
    ]);
  }

  const LS_NOTES = [
    'Why residuals? With squared loss, (y − (f + F))² = ((y − f) − F)². So "find the best F to add" is just "fit F to r = y − f": the part of y the model still gets wrong.',
    'The stump is a regression tree with one split (§18.1), fit to r instead of y. One step can only fix part of the curve; the next round fits whatever is left.',
    'ν = 1 is plain least squares boosting. A smaller ν (shrinkage) takes only part of each step: more rounds are needed, but the fit grows more smoothly and overfits more slowly.'
  ];

  function lsView(c) {
    const s = c.state.bLs, data = M().regData(c.state.bReg.seed), y = data.map(d => d.y), xs = data.map(d => d.x[0]);
    const run = M().lsBoost(data, s.k + 1, s.nu), st = run.steps[s.k];
    const terms = n => run.steps.slice(0, n).map(q => ({ stump: q.stump, coef: s.nu }));
    const fOld = x => M().ensemble(terms(s.k), [x], run.f0), fNew = x => M().ensemble(terms(s.k + 1), [x], run.f0);
    const sel = s.sel === null || s.sel >= data.length ? argmax(st.r.map(Math.abs)) : s.sel;
    const pick = i => () => patch(c, 'bLs', { sel: i });
    const ring = i => ({ stroke: i === sel ? 'var(--color-text)' : 'var(--color-surface)', sw: i === sel ? 2.6 : 1.2 });
    const after = s.p === 2, { stump } = st;
    return {
      ...stepperBindings(c, 'bLs', 'l', lsPhases(run, st, s.k, s.nu, y), LS_ROUNDS),
      lDots: xs.map((x, i) => ({ x: f1(lx(x)), y: f1(ly(y[i])), ...ring(i), onClick: pick(i) })),
      lOld: curvePath(fOld), lOldDash: after ? '6 5' : 'none', lOldOpacity: after ? '.5' : '1',
      lShowNew: after, lNew: curvePath(fNew),
      lSticks: after ? [] : xs.map((x, i) => ({ x1: f1(lx(x)), y1: f1(ly(st.fBefore[i])), x2: f1(lx(x)), y2: f1(ly(y[i])) })),
      lRes: xs.map((x, i) => ({ x: f1(lx(x)), y: f1(lr(after ? st.rAfter[i] : st.r[i])), ...ring(i), onClick: pick(i) })),
      lShowStump: s.p >= 1,
      lStumpSegs: [{ x1: f1(lx(0)), x2: f1(lx(stump.t)), y: f1(lr(stump.a)) }, { x1: f1(lx(stump.t)), x2: f1(lx(LIM)), y: f1(lr(stump.b)) }],
      lCutX: f1(lx(stump.t)),
      lResTitle: after ? 'New residuals r − νF: what round ' + (s.k + 2) + ' will fit' : 'Residuals r_i = y_i − f(x_i)' + (s.p >= 1 ? ', with the stump F fit to them' : ''),
      lInspect: lsInspect(data, st, sel, s.p, s.nu, s.k),
      lNote: LS_NOTES[s.p],
      lNu: s.nu, lNuText: s.nu.toFixed(1), lOnNu: e => patch(c, 'bLs', { nu: num(e) }),
      lMse: f3(M().mse(after ? st.fAfter : st.fBefore, y)),
      lMseLabel: 'Train MSE of f' + sub(after ? s.k + 1 : s.k),
      lRounds: String(after ? s.k + 1 : s.k),
      lResample: () => c.setState({ bReg: { seed: c.state.bReg.seed + 1 }, bLs: { ...c.state.bLs, sel: null } })
    };
  }

  // ---------- shared 2-D plot for ③ and ④ ----------

  const BOX = 300;
  const px = v => (v / LIM) * BOX;
  const py = v => BOX - (v / LIM) * BOX;
  const rect = (x0, x1, y0, y1) => ({ x: f1(px(x0)), y: f1(py(y1)), w: f1(px(x1) - px(x0)), h: f1(py(y0) - py(y1)) });
  const cls = v => (v >= 0 ? POS : NEG);

  // the two half-planes of a stump, tinted by the class (sign) each side predicts
  function stumpHalves(st) {
    const { j, t, a, b } = st;
    const left = j === 0 ? rect(0, t, 0, LIM) : rect(0, LIM, 0, t), right = j === 0 ? rect(t, LIM, 0, LIM) : rect(0, LIM, t, LIM);
    return [{ ...left, fill: cls(a), opacity: '.22' }, { ...right, fill: cls(b), opacity: '.22' }];
  }

  // the additive model's regions; stronger tint = more confident
  function modelRects(terms, strength) {
    if (!terms.length) return [];
    return M().ensembleCells(terms).map(cell => ({ ...rect(cell.x0, cell.x1, cell.y0, cell.y1), fill: cls(cell.f), opacity: (0.05 + 0.33 * strength(cell.f)).toFixed(2) }));
  }

  function cutLine(st) {
    return st.j === 0 ? { x1: f1(px(st.t)), y1: 0, x2: f1(px(st.t)), y2: BOX } : { x1: 0, y1: f1(py(st.t)), x2: BOX, y2: f1(py(st.t)) };
  }

  const ruleText = st => FEAT[st.j] + ' ≤ ' + f2(st.t);
  const where = d => '(' + f1(d.x[0]) + ', ' + f1(d.x[1]) + ')';

  function outlierBtn(c) {
    const on = c.state.bCls.outlier;
    return {
      oLabel: on ? 'Remove the outlier' : 'Add an outlier',
      oToggle: () => c.setState({ bCls: { outlier: !on }, bAda: { ...c.state.bAda, sel: null }, bLog: { ...c.state.bLog, sel: null } })
    };
  }

  // ---------- ③ AdaBoost ----------

  function adaPhases(st, k, N, y) {
    const m = k + 1, big = argmax(st.w), misses = st.miss.filter(Boolean).length;
    return [
      { title: 'Look at the weights', formula: 'ω_i,m ∝ exp(−ỹ_i f_{m−1}(x_i)),  Σᵢ ω_i = 1',
        body: k === 0 ? 'Round 1: f₀ = 0, so every point has ω = 1/N = ' + f3(1 / N) + '. Dot size shows ω.'
          : 'Largest ω = ' + f3(st.w[big]) + ' on point #' + (big + 1) + ' (uniform would be ' + f3(1 / N) + '). Points the model gets wrong have grown.' },
      { title: 'Fit a stump to the weighted data', formula: 'F_m = argmin_F Σᵢ ω_i I(ỹ_i ≠ F(x_i))',
        body: 'Best stump: ' + ruleText(st.stump) + ' → ' + pm(st.stump.a) + ', else ' + pm(st.stump.b) + '. It misses ' + misses + ' point' + (misses === 1 ? '' : 's') +
          ' (dashed rings), whose weights add up to err' + sub(m) + ' = ' + f3(st.err) + '.' },
      { title: 'Give the stump a say', formula: 'β_m = ½ log((1 − err_m) / err_m)',
        body: 'β' + sub(m) + ' = ½ log(' + f3(1 - st.err) + ' / ' + f3(st.err) + ') = ' + f2(st.beta) + '. A stump no better than a coin (err = 0.5) would get β = 0.' },
      { title: 'Reweight and add', formula: 'ω_i ← ω_i e^{−β_m ỹ_i F_m(x_i)} / Z,   f_m = f_{m−1} + β_m F_m',
        body: 'Misses × e^β = ' + f2(Math.exp(st.beta)) + ', hits × e^−β = ' + f2(Math.exp(-st.beta)) + ', then ÷ Z = ' + f3(st.Z) +
          ' so the weights sum to 1. The misses now hold exactly half the weight. Train error of sign(f' + sub(m) + '): ' + pct(M().classError(st.fAfter, y)) + '.' }
    ];
  }

  function adaInspect(d, st, i, p, k) {
    const yi = d.y, F = st.F[i], factor = Math.exp(-st.beta * yi * F);
    return upTo(p, [
      'Point #' + (i + 1) + (d.outlier ? ' (the outlier)' : '') + ' at ' + where(d) + ', label ỹ = ' + pm(yi) + '. f' + sub(k) + '(x) = ' + f2(st.fBefore[i]) + ', so its weight is ω = ' + f3(st.w[i]) + '.',
      'The stump says F(x) = ' + pm(F) + ': ' + (st.miss[i] ? 'a miss, so ω counts toward err.' : 'a hit.'),
      'β = ' + f2(st.beta) + '.',
      'ω · e^{−β·ỹ·F} = ' + f3(st.w[i]) + ' · e^{−' + f2(st.beta) + '·(' + pm(yi) + ')(' + pm(F) + ')} = ' + f3(st.w[i]) + ' × ' + f2(factor) + ' = ' + f3(st.raw[i]) +
        ', ÷ Z = ' + f3(st.wNext[i]) + '. New f(x) = ' + f2(st.fAfter[i]) + ', so sign(f) predicts ' + pm(st.fAfter[i] >= 0 ? 1 : -1) + '.'
    ]);
  }

  const ADA_NOTES = [
    'With exponential loss, the factor exp(−ỹ_i f_{m−1}(x_i)) does not depend on the new learner, so it can be pulled out as a weight. Boosting the loss = fitting to reweighted data.',
    'For ±1 outputs, minimising Σ ω_i exp(−β ỹ_i F(x_i)) over F is the same as minimising the weighted error, whatever β > 0 turns out to be.',
    'Setting the derivative in β to zero gives this closed form. The better the stump on the weighted data, the bigger its vote in the final sign(Σ β_m F_m).',
    'Every miss is multiplied by e^{2β} relative to every hit. That forces the next stump to focus on what this one got wrong.'
  ];

  function adaView(c) {
    const s = c.state.bAda, data = M().clsData(SEED, c.state.bCls.outlier), y = data.map(d => d.y), N = data.length;
    const run = M().adaBoost(data, s.k + 1), st = run.steps[s.k], after = s.p === 3;
    const terms = n => run.steps.slice(0, n).map(q => ({ stump: q.stump, coef: q.beta }));
    const w = after ? st.wNext : st.w, sel = s.sel === null || s.sel >= N ? argmax(st.w) : s.sel;
    const shown = s.p >= 2 ? s.k + 1 : s.k, missed = i => s.p >= 1 && st.miss[i];
    return {
      ...stepperBindings(c, 'bAda', 'a', adaPhases(st, s.k, N, y), CLS_ROUNDS),
      aRects: s.p === 1 || s.p === 2 ? stumpHalves(st.stump) : modelRects(terms(after ? s.k + 1 : s.k), f => Math.min(Math.abs(f) / 1.5, 1)),
      aShowCut: s.p >= 1, aCut: cutLine(st.stump),
      aDots: data.map((d, i) => ({
        x: f1(px(d.x[0])), y: f1(py(d.x[1])), r: f1(clamp(4.5 * Math.sqrt(w[i] * N), 2.5, 16)), fill: cls(d.y),
        stroke: i === sel || missed(i) ? 'var(--color-text)' : 'var(--color-surface)',
        sw: i === sel ? 3 : missed(i) ? 1.6 : 1.2, dash: i !== sel && missed(i) ? '3 2' : 'none',
        onClick: () => patch(c, 'bAda', { sel: i })
      })),
      aInspect: adaInspect(data[sel], st, sel, s.p, s.k),
      aNote: ADA_NOTES[s.p],
      aHist: run.steps.slice(0, shown).map((q, i) => ({ m: String(i + 1), rule: ruleText(q.stump) + ' → ' + pm(q.stump.a), err: f3(q.err), beta: f2(q.beta) })).slice(-6),
      aHasHist: shown > 0,
      aTrain: pct(M().classError(after ? st.fAfter : st.fBefore, y)),
      aTrainLabel: 'Train error of f' + sub(after ? s.k + 1 : s.k),
      ...outlierBtn(c)
    };
  }

  // ---------- ④ LogitBoost ----------

  const logLoss = (f, y) => M().mean(f.map((v, i) => M().LOSSES.log(y[i] * v)));

  function logitPhases(st, k, y) {
    const m = k + 1, [pLo, pHi] = minMax(st.pi), [wLo, wHi] = minMax(st.w), [zLo, zHi] = minMax(st.z), s = st.stump;
    const clipped = st.zRaw.filter(v => Math.abs(v) > M().Z_MAX).length;
    return [
      { title: 'Turn f into probabilities', formula: 'π_i = 1 / (1 + e^{−2 f_{m−1}(x_i)})',
        body: k === 0 ? 'Round 1: f₀ = 0, so every π_i = 0.50. The model is unsure about every point.'
          : 'π ranges from ' + f2(pLo) + ' to ' + f2(pHi) + '. Log loss so far: ' + f3(logLoss(st.fBefore, y)) + ' bits per point.' },
      { title: 'Newton weights and working responses', formula: 'w_i = π_i(1 − π_i),   z_i = (y*_i − π_i) / w_i',
        body: 'w runs from ' + f3(wLo) + ' to ' + f3(wHi) + ' (most for unsure points, max 0.25). z runs from ' + f2(zLo) + ' to ' + f2(zHi) +
          (clipped ? '; ' + clipped + ' value' + (clipped === 1 ? ' was' : 's were') + ' clipped at ±' + M().Z_MAX : '') + '. Dot size now shows w.' },
      { title: 'Fit a stump by weighted least squares', formula: 'F_m = argmin_F Σᵢ w_i (z_i − F(x_i))²',
        body: 'Best stump: ' + ruleText(s) + ' → ' + sg(s.a) + ', else ' + sg(s.b) + '. Each side predicts the w-weighted mean of z on that side.' },
      { title: 'Take half a step', formula: 'f_m = f_{m−1} + ½ F_m',
        body: 'Log loss ' + f3(logLoss(st.fBefore, y)) + ' → ' + f3(logLoss(st.fAfter, y)) + ' bits per point. Train error of sign(f' + sub(m) + '): ' + pct(M().classError(st.fAfter, y)) + '.' }
    ];
  }

  function logitInspect(d, st, i, p, k) {
    const ys = d.y > 0 ? 1 : 0, clipped = Math.abs(st.zRaw[i]) > M().Z_MAX;
    return upTo(p, [
      'Point #' + (i + 1) + (d.outlier ? ' (the outlier)' : '') + ' at ' + where(d) + ', ỹ = ' + pm(d.y) + ' so y* = ' + ys + '. f' + sub(k) + '(x) = ' + f2(st.fBefore[i]) +
        ', so π = 1/(1 + e^{−2·' + f2(st.fBefore[i]) + '}) = ' + f3(st.pi[i]) + '.',
      'w = ' + f3(st.pi[i]) + ' · ' + f3(1 - st.pi[i]) + ' = ' + f3(st.w[i]) + '; z = (' + ys + ' − ' + f3(st.pi[i]) + ') / ' + f3(st.w[i]) + ' = ' + f2(st.zRaw[i]) +
        (clipped ? ', clipped to ' + f2(st.z[i]) : '') + '.',
      'The stump gives F(x) = ' + sg(st.F[i]) + '.',
      'f = ' + f2(st.fBefore[i]) + ' + ½ · ' + sg(st.F[i]) + ' = ' + f2(st.fAfter[i]) + ', so π moves to ' + f3(st.piAfter[i]) + '.'
    ]);
  }

  const LOGIT_NOTES = [
    'LogitBoost keeps a probability model, p(ỹ = +1 | x) = σ(2f(x)), and minimises log loss instead of exponential loss.',
    'Log loss has no closed-form step, so each round takes one Newton step. That step is a weighted least squares problem with weights w and targets z.',
    'Same stump search as in ②, but every squared error is multiplied by w_i, so confident points (w near 0) barely count.',
    'The ½ comes from the 2 in σ(2f). Each point\'s pull, w_i · z_i = y*_i − π_i, is never more than 1 in size, so a hopeless outlier cannot take over the way it can in AdaBoost.'
  ];

  function logitView(c) {
    const s = c.state.bLog, data = M().clsData(SEED, c.state.bCls.outlier), y = data.map(d => d.y), N = data.length;
    const run = M().logitBoost(data, s.k + 1), st = run.steps[s.k], after = s.p === 3;
    const terms = n => run.steps.slice(0, n).map(q => ({ stump: q.stump, coef: 0.5 }));
    const f = after ? st.fAfter : st.fBefore, sel = s.sel === null || s.sel >= N ? argmax(st.zRaw.map(Math.abs)) : s.sel;
    const wrong = i => (f[i] >= 0 ? 1 : -1) !== y[i] && (after || s.k > 0);
    return {
      ...stepperBindings(c, 'bLog', 'g', logitPhases(st, s.k, y), CLS_ROUNDS),
      gRects: s.p === 2 ? stumpHalves(st.stump) : modelRects(terms(after ? s.k + 1 : s.k), v => Math.abs(2 * M().prob(v) - 1)),
      gShowCut: s.p >= 2, gCut: cutLine(st.stump),
      gDots: data.map((d, i) => ({
        x: f1(px(d.x[0])), y: f1(py(d.x[1])), r: f1(s.p === 0 ? 5 : 2.5 + 8 * Math.sqrt(st.w[i] / 0.25)), fill: cls(d.y),
        stroke: i === sel || wrong(i) ? 'var(--color-text)' : 'var(--color-surface)',
        sw: i === sel ? 3 : wrong(i) ? 1.6 : 1.2, dash: i !== sel && wrong(i) ? '3 2' : 'none',
        onClick: () => patch(c, 'bLog', { sel: i })
      })),
      gInspect: logitInspect(data[sel], st, sel, s.p, s.k),
      gNote: LOGIT_NOTES[s.p],
      gTrain: pct(M().classError(f, y)),
      gLoss: f3(logLoss(f, y)),
      gTrainLabel: 'Train error of f' + sub(after ? s.k + 1 : s.k)
    };
  }

  // ---------- ④ loss comparison: 0-1, exponential and log loss against the margin ỹ f(x) ----------

  const LW = 560, LH = 220, LPL = 36, LPR = 12, LPT = 12, LPB = 26, MLO = -3, MHI = 3, LTOP = 5, LOSS_SAMPLES = 240;
  const mx = m => LPL + ((m - MLO) / (MHI - MLO)) * (LW - LPL - LPR);
  const my = v => LH - LPB - (v / LTOP) * (LH - LPT - LPB);

  function lossPath(fn) {
    const ms = Array.from({ length: LOSS_SAMPLES + 1 }, (_, i) => MLO + (i * (MHI - MLO)) / LOSS_SAMPLES).filter(m => fn(m) <= LTOP);
    return ms.map((m, i) => (i ? 'L ' : 'M ') + f1(mx(m)) + ' ' + f1(my(fn(m)))).join(' ');
  }

  function lossNote(m) {
    if (m < -1) return 'A badly misclassified point (an outlier, or a mislabelled one). Exponential loss charges e^' + f1(-m) + ' and pulls on it ' + f1(M().PULL.exp(m) / M().PULL.log(m)) + '× harder than log loss, which only grows in a straight line out here.';
    if (m <= 0) return 'On or just over the wrong side: all three losses say "fix this", and exp and log loss are still close.';
    if (m < 1) return 'Right side but close to the boundary: 0-1 loss is already 0, yet exp and log loss keep pushing for a bigger margin.';
    return 'Confidently correct: both smooth losses (and their pulls) fade toward 0, so the point barely affects the next learner.';
  }

  function lossView(c) {
    const m = c.state.bLoss.m, L = M().LOSSES, P = M().PULL;
    const dot = (fn, color) => ({ x: f1(mx(m)), y: f1(my(Math.min(fn(m), LTOP))), color, show: fn(m) <= LTOP ? '1' : '0' });
    return {
      xZero: lossPath(L.zeroOne), xExp: lossPath(L.exp), xLog: lossPath(L.log),
      xMarkX: f1(mx(m)), xDots: [dot(L.zeroOne, 'var(--color-neutral-600)'), dot(L.exp, NEG), dot(L.log, POS)],
      xM: m, xMText: f2(m), xOnM: e => patch(c, 'bLoss', { m: num(e) }),
      xValZero: String(L.zeroOne(m)), xValExp: f2(L.exp(m)), xValLog: f2(L.log(m)),
      xPullExp: f2(P.exp(m)), xPullLog: f2(P.log(m)),
      xNote: lossNote(m)
    };
  }

  // ---------- state ----------

  function initState() {
    return {
      bReg: { seed: SEED },
      bLs: { k: 0, p: 0, nu: 1, sel: null },
      bCls: { outlier: false },
      bAda: { k: 0, p: 0, sel: null },
      bLog: { k: 0, p: 0, sel: null },
      bLoss: { m: -2 }
    };
  }

  function render(c) {
    return { ...lsView(c), ...adaView(c), ...logitView(c), ...lossView(c) };
  }

  return { initState, render };
})();
