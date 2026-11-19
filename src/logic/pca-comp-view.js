// View logic for Topic 19 ② (computational issues, §20.1.3): units and standardizing, the N < D
// Gram-matrix trick, and PCA by SVD. Loaded only by pca.html; reading-log.js merges render(c) into t19().
// Math is in pca-math.js; Next/Back controls are Walkthrough.steps in walkthrough.js.
window.PCACompView = (function () {
  const M = () => window.PCAMath, W = () => window.Walkthrough;
  const KEY = 'pc';
  const UNIT_FACTOR = 10, SCALED_COL = 0, ZERO = 1e-9;
  const STEP = { STANDARDIZE: 1, WIDE: 2 };
  const DEG = 180 / Math.PI;
  const C = 110, R = 90;

  const f2 = v => v.toFixed(2);
  const vec = v => '(' + v.map(f2).join(', ') + ')';
  const pad = v => (Math.abs(v) < 0.005 ? 0 : v).toFixed(2).padStart(7);
  const mat = A => A.map(r => r.map(pad).join('')).join('\n');
  const row = v => v.map(pad).join('');
  const deg = w => Math.round(Math.atan2(w[1], w[0]) * DEG);

  let cached = null;
  function model() {
    if (cached) return cached;
    const toy = M().TOY, mm = M().rescaleColumn(toy, SCALED_COL, UNIT_FACTOR);
    const base = M().pca(toy), stdCm = M().standardize(toy), stdMm = M().standardize(mm);
    cached = {
      base, mm: M().pca(mm), stdCm, stdMm,
      wStd: M().pca(stdCm.Z), wStdMm: M().pca(stdMm.Z),
      gram: M().gramTrick(M().WIDE), svd: M().svd(base.Xc)
    };
    return cached;
  }

  function phases(d) {
    const g = d.gram, sv = d.svd, N = d.base.Xc.length;
    const nz = g.eigK.filter(e => e.value > ZERO).map(e => f2(e.value)).join(', ');
    return [
      { title: 'Change the units', formula: 'x₁ → ' + UNIT_FACTOR + '·x₁   (cm → mm)',
        body: 'Same six points as ①, but x₁ is now measured in mm. w₁ swings from ' + deg(d.base.W[0]) + '° to ' + deg(d.mm.W[0]) +
          '° and λ₁ jumps from ' + f2(d.base.values[0]) + ' to ' + f2(d.mm.values[0]) + '. Nothing about the data changed, only the ruler: the feature with the biggest numbers wins.' },
      { title: 'Standardize', formula: 'x_nj → (x_nj − μ_j) / σ_j   ⇒   PCA on the correlation matrix',
        body: 'Divide each column by its standard deviation. In cm (σ = ' + vec(d.stdCm.sd) + ') and in mm (σ = ' + vec(d.stdMm.sd) +
          ') you now get the same w₁ = ' + vec(d.wStd.W[0]) + ' and ' + vec(d.wStdMm.W[0]) + '. Do this whenever features are in different units.' },
      { title: 'More features than examples', formula: 'N = 3, D = 5:   X̃ᵀX̃ is 5×5,   X̃X̃ᵀ is 3×3',
        body: 'Picture faces: N = 400 images, D = 10,304 pixels. Eigen-decomposing the D×D matrix is the slow part. The N×N Gram matrix X̃X̃ᵀ is tiny by comparison.' },
      { title: 'Eigen of the small matrix', formula: 'K = X̃X̃ᵀ,   K u = λ u',
        body: 'K gives λ = ' + g.eigK.map(e => f2(e.value)).join(', ') + '. The big 5×5 matrix gives ' + g.eigBig.map(e => f2(e.value)).join(', ') +
          ': the same non-zero ones (' + nz + '). Centering 3 points leaves only 2 directions, so the rest are 0.' },
      { title: 'Map back to D dimensions', formula: 'v = X̃ᵀ u / √λ',
        body: 'From the 3×3 problem: v₁ = ' + vec(g.mapped[0].v) + '. Straight from the 5×5 problem: ' + vec(g.eigBig[0].vector) + '. Same vector, found from a much smaller matrix.' },
      { title: 'Or skip Σ entirely: SVD', formula: 'X̃ = U S Vᵀ,   λ_j = s_j² / N',
        body: 'Back to the six points of ①: s = ' + vec(sv.s) + ', so s²/' + N + ' = ' + vec(sv.s.map(s => (s * s) / N)) +
          ', the λ from ①. The rows of Vᵀ are w₁ and w₂, and the first column of U S is the scores z from ①. Libraries compute PCA this way: it is more accurate than forming Σ.' }
    ];
  }

  // a compass: each w₁ drawn as an arrow from the centre at its angle; the legend carries the labels
  function compass(d, p) {
    const arrow = (w, color, label) => {
      const a = Math.atan2(w[1], w[0]);
      return { x2: (C + R * Math.cos(a)).toFixed(1), y2: (C - R * Math.sin(a)).toFixed(1), color, label };
    };
    const list = [
      arrow(d.base.W[0], 'var(--color-accent-700)', 'w₁ in cm: ' + deg(d.base.W[0]) + '°'),
      arrow(d.mm.W[0], '#4f6d8a', 'w₁ in mm: ' + deg(d.mm.W[0]) + '°')
    ];
    return p >= STEP.STANDARDIZE ? [...list, arrow(d.wStd.W[0], 'var(--color-accent-2-700)', 'w₁ standardized (cm or mm): ' + deg(d.wStd.W[0]) + '°')] : list;
  }

  function blocks(d, p) {
    const g = d.gram, sv = d.svd;
    const byStep = [
      [{ title: 'Σ in cm', text: mat(d.base.S) }, { title: 'Σ in mm', text: mat(d.mm.S) }],
      [{ title: 'correlation matrix (cm or mm)', text: mat(M().covariance(d.stdCm.Z)) }],
      [{ title: 'X̃  (3 × 5, centered)', text: mat(g.Xc) }, { title: 'X̃ᵀX̃  (5 × 5)', text: mat(g.big) }],
      [{ title: 'K = X̃X̃ᵀ  (3 × 3)', text: mat(g.K) }, { title: 'eigenvalues of K', text: row(g.eigK.map(e => e.value)) },
        { title: 'eigenvalues of X̃ᵀX̃', text: row(g.eigBig.map(e => e.value)) }],
      [{ title: 'u₁ (3-D, from K)', text: row(g.mapped[0].u) }, { title: 'v₁ = X̃ᵀu₁ / √λ₁', text: row(g.mapped[0].v) },
        { title: 'v₁ from the 5 × 5 directly', text: row(g.eigBig[0].vector) }],
      [{ title: 'U  (6 × 2)', text: mat(sv.U) }, { title: 's', text: row(sv.s) },
        { title: 'Vᵀ  (rows = w₁, w₂)', text: mat(sv.V) }, { title: 'U S  (column 1 = z)', text: mat(sv.U.map(r => r.map((u, j) => u * sv.s[j]))) }]
    ];
    return byStep[p];
  }

  function render(c) {
    const d = model(), p = c.state[KEY].p;
    return {
      ...W().steps(c, KEY, 'pc', phases(d)),
      pcShowCompass: p < STEP.WIDE,
      pcArrows: compass(d, p),
      pcBlocks: blocks(d, p)
    };
  }

  function initState() {
    return { [KEY]: { p: 0 } };
  }

  return { initState, render };
})();
