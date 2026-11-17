// Pure math for Topic 18 (co-training, §19.3.3). Loaded only by co-training.html, after kt-math.js
// (it reuses KTMath.rng). ct-view.js calls it.
//
// Every example has two views, x₁ and x₂. Model A only ever sees x₁, model B only sees x₂.
// Within a class the two views are independent, which is the assumption co-training relies on.
window.CTMath = (function () {
  const T = () => window.KTMath;
  const LIM = 10, EDGE = 0.2, PER_CLASS = 30, SD = 1.3, PICKS_PER_CLASS = 2;
  const CLASS_MEANS = [[3, 3], [7, 7]];
  // the only two hand-labelled examples: chosen so A's boundary starts too far right and B's too low
  const SEED_POINTS = [{ x: [4.2, 1.6], y: 0 }, { x: [8.6, 5.0], y: 1 }];

  const gauss = rand => Math.sqrt(-2 * Math.log(rand() || 1e-12)) * Math.cos(2 * Math.PI * rand());
  const clamp = v => Math.max(EDGE, Math.min(LIM - EDGE, v));
  const mean = arr => (arr.length ? arr.reduce((s, v) => s + v, 0) / arr.length : 0);

  // the labelled seeds first (ids 0 and 1), then PER_CLASS unlabelled points of each class
  function dataset(seed) {
    const rand = T().rng(seed);
    const unl = [0, 1].flatMap(y => Array.from({ length: PER_CLASS }, () => ({
      x: CLASS_MEANS[y].map(m => clamp(m + SD * gauss(rand))), y
    })));
    return [...SEED_POINTS, ...unl].map((d, id) => ({ ...d, id }));
  }

  // nearest class mean on one view: the boundary sits halfway between the two class means
  function fit(pool, data, view) {
    const mu = [0, 1].map(c => mean(pool.filter(e => e.label === c).map(e => data[e.id].x[view])));
    const t = (mu[0] + mu[1]) / 2, dir = mu[1] >= mu[0] ? 1 : -1;
    return { mu, t, dir, n: pool.length };
  }

  const predict = (model, v) => ((v - model.t) * model.dir > 0 ? 1 : 0);
  const confidence = (model, v) => Math.abs(v - model.t);

  // the PICKS_PER_CLASS most confident unlabelled points on each side of the boundary
  function pick(model, data, view, unlabelled, by) {
    return [0, 1].flatMap(c => unlabelled
      .filter(id => predict(model, data[id].x[view]) === c)
      .sort((a, b) => confidence(model, data[b].x[view]) - confidence(model, data[a].x[view]))
      .slice(0, PICKS_PER_CLASS)
      .map(id => ({ id, label: c, by })));
  }

  const accuracy = (model, data, view) => mean(data.map(d => (predict(model, d.x[view]) === d.y ? 1 : 0)));

  // co-training (Blum & Mitchell 1998): A labels points for B, B labels points for A, both retrain.
  // Each step records the models before and after so the view can show how far each boundary moved.
  function run(data, rounds) {
    const seeds = SEED_POINTS.map((_, id) => ({ id, label: data[id].y, by: 'seed' }));
    const steps = [];
    let poolA = seeds, poolB = seeds, unl = data.slice(SEED_POINTS.length).map(d => d.id);
    for (let m = 0; m < rounds && unl.length; m++) {
      const A = fit(poolA, data, 0), B = fit(poolB, data, 1);
      const picksA = pick(A, data, 0, unl, 'A');
      const rest = unl.filter(id => !picksA.some(p => p.id === id));
      const picksB = pick(B, data, 1, rest, 'B');
      const nextA = [...poolA, ...picksB], nextB = [...poolB, ...picksA];
      steps.push({ A, B, picksA, picksB, A2: fit(nextA, data, 0), B2: fit(nextB, data, 1) });
      poolA = nextA;
      poolB = nextB;
      unl = rest.filter(id => !picksB.some(p => p.id === id));
    }
    return steps;
  }

  const MAX_ROUNDS = Math.floor((2 * PER_CLASS) / (4 * PICKS_PER_CLASS));

  return { LIM, MAX_ROUNDS, dataset, predict, accuracy, run };
})();
