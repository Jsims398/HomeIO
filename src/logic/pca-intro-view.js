// View logic for Topic 19 ① (PCA by hand, §20.1.1): six 2-D points squashed to one number each,
// one step at a time. Loaded only by pca.html; reading-log.js merges render(c) into t19().
// Math is in pca-math.js; Next/Back controls are Walkthrough.steps in walkthrough.js.
window.PCAIntroView = (function () {
  const M = () => window.PCAMath, W = () => window.Walkthrough;
  const KEY = 'pi';
  const S = 400, PL = 40, PR = 12, PT = 12, PB = 40, LIM = 8, ARROW = 2, LINE_HALF = 7;
  const TICKS = [0, 2, 4, 6, 8];
  const STEP = { CENTER: 1, COV: 2, EIG: 3, ENCODE: 4, DECODE: 5 };
  const DEG = 180 / Math.PI;

  const sx = v => PL + (v / LIM) * (S - PL - PR);
  const sy = v => S - PB - (v / LIM) * (S - PT - PB);
  const PX_PER_UNIT = (S - PL - PR) / LIM;
  const f1 = v => v.toFixed(1), f2 = v => v.toFixed(2);
  const vec = v => '(' + v.map(f2).join(', ') + ')';
  const mat2 = A => '[' + A.map(r => r.map(f2).join(', ')).join(' ; ') + ']';

  let cached = null;
  const model = () => (cached || (cached = M().pca(M().TOY)));
  const pc1Deg = m => Math.round(Math.atan2(m.W[0][1], m.W[0][0]) * DEG);

  // the 2×2 eigen-decomposition done by hand, with Σ = [a, b; b, c]
  function howToW(m) {
    const a = m.S[0][0], b = m.S[0][1], c = m.S[1][1];
    const tr = a + c, det = a * c - b * b, disc = Math.sqrt(tr * tr - 4 * det);
    const l1 = m.values[0], l2 = m.values[1];
    const raw = [b, l1 - a], len = Math.hypot(raw[0], raw[1]);
    return 'How to get w (2×2 by hand), writing Σ = [a, b ; b, c] = ' + mat2(m.S) + ':' +
      '\n1. λ solves det(Σ − λI) = 0, i.e. λ² − (a + c)λ + (ac − b²) = 0:  λ² − ' + f2(tr) + 'λ + ' + f2(det) + ' = 0.' +
      '\n   λ = (' + f2(tr) + ' ± √(' + f2(tr) + '² − 4·' + f2(det) + ')) / 2 = (' + f2(tr) + ' ± ' + f2(disc) + ') / 2  →  λ₁ = ' + f2(l1) + ', λ₂ = ' + f2(l2) + '.' +
      '\n2. w solves (Σ − λ₁I) w = 0. Top row: (' + f2(a) + ' − ' + f2(l1) + ') w_x + ' + f2(b) + ' w_y = 0, so w ∝ (b, λ₁ − a) = (' + f2(raw[0]) + ', ' + f2(raw[1]) + ').' +
      '\n3. Scale to length 1: divide by √(' + f2(raw[0]) + '² + ' + f2(raw[1]) + '²) = ' + f2(len) + '  →  w₁ = ' + vec(raw.map(v => v / len)) + '.' +
      '\n4. Same with λ₂ gives w₂ = (b, λ₂ − a) / length. Its sign is arbitrary: w and −w are the same direction.';
  }

  function phases(m) {
    const z = M().TOY.map(x => M().encode(x, m.mu, m.W, 1)[0]);
    const trace = m.S[0][0] + m.S[1][1];
    return [
      { title: 'Start with the data', formula: 'x_n ∈ ℝ²,  N = 6',
        body: 'Six points, each a pair (x₁, x₂). Goal: describe each one with a single number z_n and lose as little as possible.' },
      { title: 'Center', formula: 'μ = (1/N) Σ x_n = ' + vec(m.mu) + ',   x̃_n = x_n − μ',
        body: 'Subtract the mean from every point. The cross marks μ, the new origin. See the x̃ column in the table.' },
      { title: 'Covariance', formula: 'Σ = (1/N) Σ x̃_n x̃_nᵀ = ' + mat2(m.S),
        body: 'Diagonal: variance of x₁ (' + f2(m.S[0][0]) + ') and of x₂ (' + f2(m.S[1][1]) + '). Off-diagonal ' + f2(m.S[0][1]) +
          ' > 0: they rise together. The ellipse is the shape Σ describes.' },
      { title: 'Eigenvectors of Σ', formula: 'Σ w = λ w:  w₁ = ' + vec(m.W[0]) + ', λ₁ = ' + f2(m.values[0]) + ';  w₂ = ' + vec(m.W[1]) + ', λ₂ = ' + f2(m.values[1]),
        body: howToW(m) + '\n\nMeaning: w₁ is the long axis of the ellipse, w₂ the short one, and λ is the variance along each. Together they keep all the variance: λ₁ + λ₂ = ' +
          f2(m.values[0] + m.values[1]) + ' = trace Σ = ' + f2(trace) + '.' },
      { title: 'Encode', formula: 'z_n = w₁ᵀ x̃_n',
        body: 'Each point\'s shadow on the w₁ line: z = ' + z.map(f2).join(', ') + '. Two numbers per point became one.' },
      { title: 'Decode and measure the loss', formula: 'x̂_n = μ + z_n w₁,   (1/N) Σ ‖x_n − x̂_n‖² = λ₂ = ' + f2(m.values[1]),
        body: 'Map each z back to 2-D: it lands on the line. The grey gaps are what was lost, and their mean square is exactly λ₂, the variance in the direction we dropped.' }
    ];
  }

  const at = (mu, u, t) => [mu[0] + t * u[0], mu[1] + t * u[1]];
  const CROSS = 7;
  const cross = (x, y) => 'M ' + f1(x - CROSS) + ' ' + f1(y - CROSS) + ' L ' + f1(x + CROSS) + ' ' + f1(y + CROSS) +
    ' M ' + f1(x - CROSS) + ' ' + f1(y + CROSS) + ' L ' + f1(x + CROSS) + ' ' + f1(y - CROSS);
  const seg = (a, b) => ({ x1: f1(sx(a[0])), y1: f1(sy(a[1])), x2: f1(sx(b[0])), y2: f1(sy(b[1])) });

  function arrows(m) {
    return m.W.map((w, j) => {
      const tip = at(m.mu, w, ARROW);
      return { ...seg(m.mu, tip), tx: f1(sx(tip[0]) + 6), ty: f1(sy(tip[1]) - 6), label: 'w' + (j ? '₂' : '₁'),
        color: j ? 'var(--color-accent-2-700)' : 'var(--color-accent-700)' };
    });
  }

  function ellipse(m) {
    return {
      cx: f1(sx(m.mu[0])), cy: f1(sy(m.mu[1])),
      rx: f1(2 * Math.sqrt(m.values[0]) * PX_PER_UNIT), ry: f1(2 * Math.sqrt(m.values[1]) * PX_PER_UNIT),
      rot: 'rotate(' + (-pc1Deg(m)) + ' ' + f1(sx(m.mu[0])) + ' ' + f1(sy(m.mu[1])) + ')'
    };
  }

  // the line you can rotate, each point's projection onto it, and the gap left behind
  function projection(m, deg) {
    const st = M().lineStats(m, deg / DEG);
    const hats = st.z.map(z => at(m.mu, st.u, z));
    return {
      st, hats,
      line: seg(at(m.mu, st.u, -LINE_HALF), at(m.mu, st.u, LINE_HALF)),
      proj: hats.map(h => ({ x: f1(sx(h[0])), y: f1(sy(h[1])) })),
      resid: M().TOY.map((x, n) => seg(x, hats[n]))
    };
  }

  function table(m, p, pr) {
    const show = (step, text) => (p >= step ? text : '·');
    return M().TOY.map((x, n) => ({
      n: n + 1, x: vec(x),
      xc: show(STEP.CENTER, vec(m.Xc[n])),
      z: show(STEP.ENCODE, f2(pr.st.z[n])),
      xhat: show(STEP.DECODE, vec(pr.hats[n])),
      err: show(STEP.DECODE, f2((x[0] - pr.hats[n][0]) ** 2 + (x[1] - pr.hats[n][1]) ** 2))
    }));
  }

  function slider(c, m, s, pr) {
    const best = pc1Deg(m), onBest = s.ang === best;
    return {
      piAng: s.ang,
      piOnAng: e => W().patch(c, KEY, { ang: parseInt(e.target.value, 10) }),
      piSnap: () => W().patch(c, KEY, { ang: best }),
      piVar: f2(pr.st.variance), piErr: f2(pr.st.error), piTot: f2(pr.st.variance + pr.st.error),
      piAngNote: onBest ? 'This is w₁ (' + best + '°): the biggest variance and the smallest error possible.'
        : 'Line at ' + s.ang + '°. w₁ is at ' + best + '°. Variance is lower and error higher than they could be.'
    };
  }

  function render(c) {
    const m = model(), s = c.state[KEY], p = s.p, pr = projection(m, s.ang);
    return {
      ...W().steps(c, KEY, 'pi', phases(m)),
      ...slider(c, m, s, pr),
      // tick labels are static <text> in pca.html: the runtime can't put interpolated text inside SVG
      piGridX: TICKS.map(v => ({ x: f1(sx(v)), y1: f1(sy(LIM)), y2: f1(sy(0)) })),
      piGridY: TICKS.map(v => ({ y: f1(sy(v)), x1: f1(sx(0)), x2: f1(sx(LIM)) })),
      piDots: M().TOY.map(x => ({ x: f1(sx(x[0])), y: f1(sy(x[1])) })),
      piShowMean: p >= STEP.CENTER, piMuPath: cross(sx(m.mu[0]), sy(m.mu[1])),
      piMuAxes: [seg([m.mu[0], 0], [m.mu[0], LIM]), seg([0, m.mu[1]], [LIM, m.mu[1]])],
      piShowEllipse: p >= STEP.COV, piEll: ellipse(m),
      piShowEig: p >= STEP.EIG, piArrows: arrows(m),
      piW1Lbl: { x: arrows(m)[0].tx, y: arrows(m)[0].ty }, piW2Lbl: { x: arrows(m)[1].tx, y: arrows(m)[1].ty },
      piShowLine: p >= STEP.ENCODE, piLine: pr.line, piProj: pr.proj,
      piShowResid: p >= STEP.DECODE, piResid: pr.resid,
      piRows: table(m, p, pr).map(r => ({ ...r, n: String(r.n) }))
    };
  }

  function initState() {
    return { [KEY]: { p: 0, ang: pc1Deg(model()) } };
  }

  return { initState, render };
})();
