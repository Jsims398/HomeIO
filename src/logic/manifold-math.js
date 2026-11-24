// Pure math for Topic 20 (manifold learning, §20.4.1 – 20.4.3). Loaded only by manifold.html, after
// kt-math.js (KTMath.rng) and pca-math.js (PCAMath.pca for the 2-D PCA). The manifold-*-view.js files call it.
window.ManifoldMath = (function () {
  const R = () => window.KTMath, P = () => window.PCAMath;
  const JACOBI_SWEEPS = 50, TINY = 1e-10;
  const sum = arr => arr.reduce((s, v) => s + v, 0);
  const dist = (a, b) => Math.hypot(...a.map((v, i) => v - b[i]));

  // eigen-decomposition of a symmetric matrix by cyclic Jacobi, values sorted biggest first.
  // Works on private copies and touches O(n) entries per rotation, so 70×70 stays fast.
  function eigSym(S) {
    const n = S.length, A = S.map(r => [...r]);
    const V = Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));
    for (let sweep = 0; sweep < JACOBI_SWEEPS; sweep++) {
      let off = 0;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p][q] * A[p][q];
      if (off < TINY) break;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
        if (Math.abs(A[p][q]) > TINY) rotate(A, V, p, q);
      }
    }
    return A.map((r, i) => ({ value: r[i], vector: V.map(row => row[i]) })).sort((a, b) => b.value - a.value);
  }

  // one Jacobi rotation zeroing A[p][q], applied in place to eigSym's private copies A and V
  function rotate(A, V, p, q) {
    const theta = (A[q][q] - A[p][p]) / (2 * A[p][q]);
    const t = (theta >= 0 ? 1 : -1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
    const c = 1 / Math.sqrt(t * t + 1), s = t * c;
    for (let k = 0; k < A.length; k++) {
      const akp = A[k][p], akq = A[k][q];
      A[k][p] = c * akp - s * akq; A[k][q] = s * akp + c * akq;
    }
    for (let k = 0; k < A.length; k++) {
      const apk = A[p][k], aqk = A[q][k];
      A[p][k] = c * apk - s * aqk; A[q][k] = s * apk + c * aqk;
    }
    for (let k = 0; k < V.length; k++) {
      const vkp = V[k][p], vkq = V[k][q];
      V[k][p] = c * vkp - s * vkq; V[k][q] = s * vkp + c * vkq;
    }
  }

  // ---- §20.4.1: a rolled-up curve, the 2-D cousin of the swiss roll ----
  // Archimedean spiral r = B·θ for θ in [TH0, TH1]; points spaced evenly by arc length.
  const SPIRAL = { N: 120, B: 0.5, TH0: 1.5 * Math.PI, TH1: 4.5 * Math.PI, DENSE: 4000 };
  const onSpiral = th => [SPIRAL.B * th * Math.cos(th), SPIRAL.B * th * Math.sin(th)];

  function spiral() {
    const { N, TH0, TH1, DENSE } = SPIRAL;
    const ths = Array.from({ length: DENSE + 1 }, (_, i) => TH0 + ((TH1 - TH0) * i) / DENSE);
    const pts = ths.map(onSpiral);
    const cum = pts.reduce((acc, p, i) => [...acc, i ? acc[i - 1] + dist(p, pts[i - 1]) : 0], []);
    const total = cum[DENSE];
    const picks = Array.from({ length: N }, (_, n) => {
      const target = (total * n) / (N - 1);
      const i = cum.findIndex(c => c >= target - 1e-9);
      return { x: pts[i < 0 ? DENSE : i], t: cum[i < 0 ? DENSE : i], th: ths[i < 0 ? DENSE : i] };
    });
    return { X: picks.map(p => p.x), t: picks.map(p => p.t), th: picks.map(p => p.th), total };
  }

  // unit tangent of the spiral at angle θ (derivative of (Bθ cos θ, Bθ sin θ), normalized)
  function tangent(th) {
    const d = [Math.cos(th) - th * Math.sin(th), Math.sin(th) + th * Math.cos(th)], n = Math.hypot(...d);
    return d.map(v => v / n);
  }

  // Spearman rank correlation between two lists (no ties expected)
  function spearman(a, b) {
    const rank = v => {
      const order = v.map((x, i) => [x, i]).sort((p, q) => p[0] - q[0]).map(([, i]) => i);
      return v.map((_, i) => order.indexOf(i));
    };
    const ra = rank(a), rb = rank(b), n = a.length;
    return 1 - (6 * sum(ra.map((r, i) => (r - rb[i]) ** 2))) / (n * (n * n - 1));
  }

  // ---- §20.4.3: k-NN graph → shortest paths → classical MDS ----
  // symmetric k-nearest-neighbour graph: edges { i, j, w } and an adjacency list of { j, w }
  function knnGraph(X, k) {
    const near = X.map((x, i) => X.map((y, j) => ({ j, w: dist(x, y) })).filter(e => e.j !== i).sort((a, b) => a.w - b.w).slice(0, k));
    const keys = new Set(near.flatMap((es, i) => es.map(e => Math.min(i, e.j) + '-' + Math.max(i, e.j))));
    const edges = [...keys].map(s => s.split('-').map(Number)).map(([i, j]) => ({ i, j, w: dist(X[i], X[j]) }));
    const adj = X.map((_, i) => edges.filter(e => e.i === i || e.j === i).map(e => ({ j: e.i === i ? e.j : e.i, w: e.w })));
    return { edges, adj };
  }

  // Dijkstra from one source (O(N²), fine for N < 100); d and prev are local working arrays
  function dijkstra(adj, src) {
    const n = adj.length, d = Array(n).fill(Infinity), prev = Array(n).fill(-1), done = Array(n).fill(false);
    d[src] = 0;
    for (let it = 0; it < n; it++) {
      let u = -1;
      for (let i = 0; i < n; i++) if (!done[i] && (u < 0 || d[i] < d[u])) u = i;
      if (u < 0 || d[u] === Infinity) break;
      done[u] = true;
      adj[u].forEach(({ j, w }) => { if (d[u] + w < d[j]) { d[j] = d[u] + w; prev[j] = u; } });
    }
    return { d, prev };
  }

  const pathTo = (prev, dst) => (dst < 0 ? [] : [...pathTo(prev, prev[dst]), dst]);

  // classical MDS into 1-D: B = −½ J D² J, z = √λ₁ u₁ (λ₁ = top eigenvalue of B)
  function mds1(D) {
    const n = D.length, D2 = D.map(r => r.map(v => v * v));
    const rowM = D2.map(r => sum(r) / n), all = sum(rowM) / n;
    const B = D2.map((r, i) => r.map((v, j) => -0.5 * (v - rowM[i] - rowM[j] + all)));
    const top = eigSym(B)[0];
    return top.vector.map(u => u * Math.sqrt(Math.max(top.value, 0)));
  }

  // the whole nonparametric pipeline for one k; z is null when the graph falls apart
  function isomap1(X, k) {
    const g = knnGraph(X, k);
    const runs = X.map((_, i) => dijkstra(g.adj, i));
    const D = runs.map(r => r.d);
    const connected = D.every(r => r.every(Number.isFinite));
    return { ...g, D, runs, connected, z: connected ? mds1(D) : null };
  }

  // ---- §20.4.2: images of a rotated bar, a 1-D manifold in pixel space ----
  const IMG = { SIZE: 7, HALF_LEN: 2.8, WIDTH: 0.9, ANGLES: 36 };

  // pixel brightness in [0, 1]: near the bar's centre line and within its length (row 0 = top)
  function barImage(deg) {
    const { SIZE, HALF_LEN, WIDTH } = IMG, mid = (SIZE - 1) / 2, a = (deg * Math.PI) / 180;
    const u = [Math.cos(a), -Math.sin(a)];
    return Array.from({ length: SIZE * SIZE }, (_, k) => {
      const dx = (k % SIZE) - mid, dy = Math.floor(k / SIZE) - mid;
      const along = dx * u[0] + dy * u[1], across = Math.abs(dx * u[1] - dy * u[0]);
      const fade = Math.max(0, 1 - Math.max(0, Math.abs(along) - HALF_LEN));
      return Math.max(0, 1 - across / WIDTH) * fade;
    });
  }

  const randomImage = seed => { const rand = R().rng(seed); return Array.from({ length: IMG.SIZE * IMG.SIZE }, () => rand()); };

  // ANGLES rotated copies, PCA via the N×N Gram matrix: scores on PC1/PC2 and the full spectrum
  function rotationSet() {
    const angles = Array.from({ length: IMG.ANGLES }, (_, i) => (180 * i) / IMG.ANGLES);
    const X = angles.map(barImage), N = X.length;
    const mu = X[0].map((_, j) => sum(X.map(r => r[j])) / N), Xc = X.map(r => r.map((v, j) => v - mu[j]));
    const K = Xc.map(a => Xc.map(b => sum(a.map((v, i) => v * b[i]))));
    const eig = eigSym(K);
    const values = eig.map(e => Math.max(0, e.value) / N);
    const scores = X.map((_, n) => [0, 1].map(j => eig[j].vector[n] * Math.sqrt(Math.max(eig[j].value, 0))));
    return { angles, X, mu, values, scores };
  }

  return {
    sum, dist, eigSym, SPIRAL, spiral, tangent, spearman, knnGraph, dijkstra, pathTo, mds1, isomap1,
    IMG, barImage, randomImage, rotationSet, pca: X => P().pca(X)
  };
})();
