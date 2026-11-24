// View logic for Topic 20 ② (the manifold hypothesis, §20.4.2): 7×7 images of a bar. Random pixels give
// noise; rotating one bar traces a 1-D loop through ℝ⁴⁹. Loaded only by manifold.html; reading-log.js
// merges render(c) into t20(). Math is in manifold-math.js; Next/Back is Walkthrough.steps.
window.ManifoldImagesView = (function () {
  const M = () => window.ManifoldMath, W = () => window.Walkthrough, G = () => window.ManifoldRollView.geom;
  const KEY = 'mh';
  const CELL = 20, THRESHOLD = 0.9, SPEC_SHOWN = 12, NEIGHBOUR_STEP = 2;
  const STEP = { RANDOM: 1, KNOB: 2, LOOP: 3, SPECTRUM: 4 };
  const SC = { S: 240, PAD: 22 };
  const CH = { W: 420, H: 200, PL: 16, PR: 12, PT: 14, PB: 24, FILL: 0.62 };

  const f1 = v => v.toFixed(1), f2 = v => v.toFixed(2), pct = v => Math.round(v * 100) + '%';

  let cached = null;
  function model() {
    if (cached) return cached;
    const rs = M().rotationSet(), total = M().sum(rs.values);
    const cum = rs.values.map((_, i) => M().sum(rs.values.slice(0, i + 1)) / total);
    cached = { ...rs, total, cum, need: cum.findIndex(f => f >= THRESHOLD) + 1, step: 180 / rs.angles.length };
    return cached;
  }

  const size = () => M().IMG.SIZE;
  const cells = img => img.map((v, k) => ({ x: (k % size()) * CELL, y: Math.floor(k / size()) * CELL, o: f2(v) }));
  const asText = img => Array.from({ length: size() }, (_, r) => img.slice(r * size(), (r + 1) * size()).map(v => v.toFixed(1)).join(' ')).join('\n');

  // nearest of the bar images to a given image, and the gap between neighbouring bar images
  const nearestBar = (d, img) => Math.min(...d.X.map(x => M().dist(x, img)));
  const neighbourGap = d => M().dist(d.X[0], d.X[1]);

  function phases(d, idx, noise) {
    const n = d.angles.length, far = M().dist(d.X[idx], d.X[(idx + n / 2) % n]);
    return [
      { title: 'An image is a point', formula: 'x ∈ ℝᴰ,   D = 7 × 7 = 49 pixels',
        body: 'Write the 49 brightness values in a row and the picture becomes one point in 49-dimensional space. A 28×28 MNIST digit is a point in ℝ⁷⁸⁴ the same way.' },
      { title: 'A random point is noise', formula: 'x ~ Uniform[0, 1]⁴⁹',
        body: 'Pick 49 random brightnesses and you get static. Its distance to the nearest bar image is ' + f2(noise) + ', while two bar images ' + d.step + '° apart are only ' +
          f2(neighbourGap(d)) + ' apart. Almost all of ℝ⁴⁹ looks like this: real images fill only a tiny sliver of the space.' },
      { title: 'One knob: rotate the bar', formula: 'x(θ) = bar rotated by θ',
        body: 'Drag θ. All 49 numbers change, but only one knob is turning, so the images trace out a curve: a 1-D manifold inside ℝ⁴⁹. Add left/right and up/down shifts and it becomes a 3-D manifold. ' +
          'That is the manifold hypothesis: natural high-dimensional data sit on or near a low-dimensional manifold, d ≪ D.' },
      { title: 'See the curve', formula: 'PC1, PC2 of the ' + n + ' rotated images',
        body: 'Squash all ' + n + ' images (one every ' + d.step + '°) to 2-D with PCA and they lie on a closed loop: turning 180° brings the bar back to where it started. ' +
          'Neighbouring angles are close (' + f2(neighbourGap(d)) + '); the bar at θ + 90° is ' + f2(far) + ' away. Moving θ walks the highlighted dot round the loop.' },
      { title: 'Curved, so linear needs many directions', formula: 'F_L ≥ ' + pct(THRESHOLD) + '  needs  L = ' + d.need + ',   true d = 1',
        body: 'The eigenvalue spectrum of the same images. PCA needs ' + d.need + ' directions to keep ' + pct(THRESHOLD) + ' of the variance, though one number (θ) describes every image. ' +
          'A linear method counts the flat directions the curve bends through; a manifold method looks for the single coordinate along it.' }
    ];
  }

  // PC1/PC2 scatter scaled into a square, coloured round the loop by angle
  function loop(d, idx) {
    const m = Math.max(...d.scores.flat().map(Math.abs)) * 1.1;
    const to = v => SC.PAD + ((v + m) / (2 * m)) * (SC.S - 2 * SC.PAD);
    const pts = d.scores.map(([a, b], n) => ({ x: f1(to(a)), y: f1(SC.S - to(b)), color: G().color(n / d.angles.length) }));
    return { pts, path: [...pts, pts[0]].map(q => q.x + ',' + q.y).join(' '), cur: pts[idx] };
  }

  function spectrum(d) {
    const n = SPEC_SHOWN, pw = CH.W - CH.PL - CH.PR, ph = CH.H - CH.PT - CH.PB, hi = d.values[0];
    const w = (pw / n) * CH.FILL;
    return d.values.slice(0, n).map((v, i) => {
      const h = (v / hi) * ph;
      return { x: f1(CH.PL + ((i + 0.5) / n) * pw - w / 2), y: f1(CH.PT + ph - h), w: f1(w), h: f1(h),
        color: i < d.need ? 'var(--color-accent)' : 'var(--color-neutral-400)' };
    });
  }

  function stats(d, p, ang, noise, idx) {
    if (p === STEP.RANDOM) return [{ k: 'to nearest bar', v: f2(noise) }, { k: 'bar to next bar', v: f2(neighbourGap(d)) }];
    if (p === STEP.KNOB || p === STEP.LOOP) {
      const next = d.X[(idx + NEIGHBOUR_STEP) % d.angles.length];
      return [{ k: 'θ', v: ang + '°' }, { k: 'PC1, PC2', v: d.scores[idx].map(f2).join(', ') },
        { k: 'distance to θ + ' + NEIGHBOUR_STEP * d.step + '°', v: f2(M().dist(d.X[idx], next)) }];
    }
    if (p === STEP.SPECTRUM) return [{ k: 'top 2 keep', v: pct(d.cum[1]) }, { k: 'L for ' + pct(THRESHOLD), v: String(d.need) }, { k: 'true d', v: '1' }];
    return [{ k: 'D', v: String(size() * size()) }, { k: 'θ', v: ang + '°' }];
  }

  function render(c) {
    const d = model(), s = c.state[KEY], p = s.p;
    const idx = Math.round(s.ang / d.step) % d.angles.length;
    const randImg = M().randomImage(s.seed), noise = nearestBar(d, randImg);
    const showRandom = p === STEP.RANDOM, img = showRandom ? randImg : M().barImage(s.ang), lp = loop(d, idx);
    return {
      ...W().steps(c, KEY, 'mh', phases(d, idx, noise)),
      mhAng: s.ang, mhAngStep: d.step, mhAngMax: 180 - d.step,
      mhOnAng: e => W().patch(c, KEY, { ang: parseInt(e.target.value, 10) }),
      mhReroll: () => W().patch(c, KEY, { seed: s.seed + 1 }),
      mhShowRandom: showRandom, mhShowKnob: p >= STEP.KNOB,
      mhCells: cells(img), mhImgTitle: showRandom ? 'random 49 numbers' : 'bar at θ = ' + s.ang + '°',
      mhVector: asText(img),
      mhShowLoop: p === STEP.LOOP, mhLoopPts: lp.pts, mhLoopPath: lp.path, mhLoopCur: lp.cur,
      mhShowSpec: p === STEP.SPECTRUM, mhSpec: spectrum(d),
      mhStats: stats(d, p, s.ang, noise, idx)
    };
  }

  function initState() {
    return { [KEY]: { p: 0, ang: 30, seed: 7 } };
  }

  return { initState, render };
})();
