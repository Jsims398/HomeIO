// Pure math for Topic 9 (Naive Bayes, §9.3 – 9.4).
// Loaded only by naive-bayes.html; reading-log.js calls it from t9().
window.NBMath = (function () {
  // toy training set: how many emails in each class contained each word
  const TRAIN = {
    nSpam: 8, nHam: 12,
    words: [
      { word: 'free', spam: 6, ham: 2 },
      { word: 'money', spam: 5, ham: 1 },
      { word: 'winner', spam: 3, ham: 0 },
      { word: 'meeting', spam: 0, ham: 7 },
      { word: 'lunch', spam: 1, ham: 5 }
    ]
  };

  // θ = count / total, or (count + 1) / (total + 2) with add-one smoothing
  const rate = (c, n, smooth) => (smooth ? (c + 1) / (n + 2) : c / n);

  function params(smooth) {
    const { nSpam, nHam, words } = TRAIN;
    return {
      prior: nSpam / (nSpam + nHam),
      words: words.map(w => ({ ...w, ts: rate(w.spam, nSpam, smooth), th: rate(w.ham, nHam, smooth) }))
    };
  }

  // states[d] is 'yes' | 'no' | 'unk'. An unknown word is marginalized out:
  // Σ_x p(x | y) = 1, so its factor is just dropped.
  function classify(states, smooth) {
    const P = params(smooth);
    const factors = P.words.map((w, d) => {
      const st = states[d];
      if (st === 'unk') return { ...w, st, fs: 1, fh: 1 };
      const fs = st === 'yes' ? w.ts : 1 - w.ts, fh = st === 'yes' ? w.th : 1 - w.th;
      return { ...w, st, fs, fh };
    });
    const spam = factors.reduce((a, f) => a * f.fs, P.prior);
    const ham = factors.reduce((a, f) => a * f.fh, 1 - P.prior);
    const total = spam + ham;
    return { prior: P.prior, factors, spam, ham, post: total > 0 ? spam / total : NaN };
  }

  // ---- 9.4 race: Naive Bayes vs logistic regression on the same data ----

  function rng(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6d2b79f5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // p(clue = 1 | spam), p(clue = 1 | ham) for the independent clues
  const CLUES = [[0.7, 0.3], [0.65, 0.4], [0.35, 0.6], [0.6, 0.45], [0.4, 0.55], [0.55, 0.4]];
  // the first clue also gets near-copies, which breaks the naive assumption
  const COPIES = 4, COPY_FLIP = 0.05;

  function draw(rand, copies) {
    const y = rand() < 0.5 ? 1 : 0;
    const x = CLUES.map(([s, h]) => (rand() < (y ? s : h) ? 1 : 0));
    for (let c = 0; c < copies; c++) x.push(rand() < COPY_FLIP ? 1 - x[0] : x[0]);
    return { x, y };
  }

  function fitNB(data) {
    const D = data[0].x.length, n1 = data.filter(e => e.y).length, n0 = data.length - n1;
    const c1 = new Array(D).fill(0), c0 = new Array(D).fill(0);
    data.forEach(e => e.x.forEach((v, d) => { if (v) (e.y ? c1 : c0)[d]++; }));
    const t1 = c1.map(c => (c + 1) / (n1 + 2)), t0 = c0.map(c => (c + 1) / (n0 + 2));
    const bias = Math.log((n1 + 1) / (n0 + 1));
    return x => x.reduce((a, v, d) => a + Math.log(v ? t1[d] / t0[d] : (1 - t1[d]) / (1 - t0[d])), bias);
  }

  // plain gradient descent on the mean log loss with a small L2 penalty
  function fitLR(data, iters = 300, lr = 0.5, l2 = 0.01) {
    const D = data[0].x.length, w = new Array(D + 1).fill(0), N = data.length;
    for (let it = 0; it < iters; it++) {
      const g = new Array(D + 1).fill(0);
      data.forEach(({ x, y }) => {
        const z = x.reduce((a, v, d) => a + v * w[d + 1], w[0]);
        const err = 1 / (1 + Math.exp(-z)) - y;
        g[0] += err;
        x.forEach((v, d) => { g[d + 1] += err * v; });
      });
      for (let d = 0; d <= D; d++) w[d] -= lr * (g[d] / N + (d ? l2 * w[d] : 0));
    }
    return x => x.reduce((a, v, d) => a + v * w[d + 1], w[0]);
  }

  const accuracy = (score, test) => test.filter(e => (score(e.x) > 0 ? 1 : 0) === e.y).length / test.length;

  // average test accuracy of each model for every training-set size
  function race(sizes, reps, copies, seed) {
    const rand = rng(seed);
    const test = Array.from({ length: 2000 }, () => draw(rand, copies));
    const nb = sizes.map(() => 0), lr = sizes.map(() => 0);
    for (let r = 0; r < reps; r++) {
      sizes.forEach((n, i) => {
        const train = Array.from({ length: n }, () => draw(rand, copies));
        nb[i] += accuracy(fitNB(train), test) / reps;
        lr[i] += accuracy(fitLR(train), test) / reps;
      });
    }
    return { sizes, nb, lr };
  }

  return { TRAIN, COPIES, params, classify, race, rng };
})();
