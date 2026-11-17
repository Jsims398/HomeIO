// View logic for Topic 18 (co-training, §19.3.3): one step-by-step walkthrough of two models, each
// seeing one view of the data, labelling points for each other. Loaded only by co-training.html.
// reading-log.js calls CTView.initState() and CTView.render(component) from t18().
// The math is in ct-math.js; the round/step controls are in walkthrough.js.
window.CTView = (function () {
  const M = () => window.CTMath, W = () => window.Walkthrough;
  const SEED = 18;
  const CLS = ['#4f6d8a', 'var(--color-accent)'];
  const MODEL = { A: 'var(--color-accent-2-700)', B: '#8a5e7a' };
  const UNLABELLED = 'var(--color-neutral-300)';

  const S = 400, PL = 40, PR = 12, PT = 12, PB = 40, LIM = 10;
  const sx = v => PL + (v / LIM) * (S - PL - PR);
  const sy = v => S - PB - (v / LIM) * (S - PT - PB);
  const f1 = v => v.toFixed(1), f2 = v => v.toFixed(2);
  const pct = v => Math.round(v * 100) + '%';
  const TICKS = [0, 2, 4, 6, 8, 10];
  const TICK = 9;

  // the data and every round are fixed, so compute them once on first render
  let cached = null;
  const model = () => {
    if (!cached) {
      const data = M().dataset(SEED);
      cached = { data, steps: M().run(data, M().MAX_ROUNDS) };
    }
    return cached;
  };

  const vline = t => ({ x1: f1(sx(t)), y1: f1(sy(LIM)), x2: f1(sx(t)), y2: f1(sy(0)) });
  const hline = t => ({ x1: f1(sx(0)), y1: f1(sy(t)), x2: f1(sx(LIM)), y2: f1(sy(t)) });
  const list = (data, picks, view) => picks.map(p => f1(data[p.id].x[view])).join(', ');
  const means = m => 'μ₀ = ' + f2(m.mu[0]) + ', μ₁ = ' + f2(m.mu[1]);

  function phases(data, st) {
    return [
      { title: 'A labels points for B', formula: 'A sees only x₁: boundary t_A = ½(μ₀ + μ₁) = ' + f2(st.A.t),
        body: 'A trained on ' + st.A.n + ' points. It labels the 2 unlabelled points farthest from its line on each side, at x₁ = ' + list(data, st.picksA, 0) + '.' },
      { title: 'B labels points for A', formula: 'B sees only x₂: boundary t_B = ½(μ₀ + μ₁) = ' + f2(st.B.t),
        body: 'B trained on ' + st.B.n + ' points. Its 4 most confident picks sit at x₂ = ' + list(data, st.picksB, 1) + '.' },
      { title: 'Swap and retrain', formula: 'A ← A + B\'s labels,   B ← B + A\'s labels',
        body: 'A\'s line: ' + f2(st.A.t) + ' → ' + f2(st.A2.t) + '. B\'s line: ' + f2(st.B.t) + ' → ' + f2(st.B2.t) + '.' }
    ];
  }

  // what each model's new labels look like on the OTHER model's axis: the heart of co-training
  function inspect(data, st, p) {
    const forB = 'A chose its points by x₁ alone, yet on B\'s axis they land at x₂ = ' + list(data, st.picksA, 1) +
      ': spread out, not bunched at the edges. Within a class x₁ says nothing about x₂, so to B these are fresh, typical examples.';
    const forA = 'Likewise, B\'s picks land at x₁ = ' + list(data, st.picksB, 0) + ' on A\'s axis.';
    const moved = 'A\'s class means are now ' + means(st.A2) + ' (were ' + means(st.A) + '). B\'s are ' + means(st.B2) + ' (were ' + means(st.B) + ').';
    return [forB, forA, moved].slice(0, p + 1).join(' ');
  }

  const NOTES = [
    'A is confident far from its own line. The tick marks on the left edge show where those points sit on B\'s axis.',
    'The tick marks along the bottom show where B\'s picks sit on A\'s axis.',
    'Each model moved because of the other one\'s labels. Neither could fix itself this way: it never sees the other view.'
  ];

  // a pick drawn as a tick on the axis of the model that RECEIVES it
  const tickOnB = data => pk => ({ x1: f1(sx(0) - TICK), y1: f1(sy(data[pk.id].x[1])), x2: f1(sx(0)), y2: f1(sy(data[pk.id].x[1])), color: CLS[pk.label] });
  const tickOnA = data => pk => ({ x1: f1(sx(data[pk.id].x[0])), y1: f1(sy(0)), x2: f1(sx(data[pk.id].x[0])), y2: f1(sy(0) + TICK), color: CLS[pk.label] });

  function dots(data, steps, k, p) {
    const st = steps[k];
    const earlier = steps.slice(0, k).flatMap(q => [...q.picksA, ...q.picksB]);
    const current = [...st.picksA, ...(p >= 1 ? st.picksB : [])];
    const byId = new Map([...earlier.map(pk => [pk.id, { pk, now: false }]), ...current.map(pk => [pk.id, { pk, now: true }])]);
    return data.map((d, i) => {
      const at = { x: f1(sx(d.x[0])), y: f1(sy(d.x[1])) };
      if (i < 2) return { ...at, r: 8, fill: CLS[d.y], stroke: 'var(--color-text)', sw: 2.5 };
      const hit = byId.get(i);
      if (!hit) return { ...at, r: 4.5, fill: UNLABELLED, stroke: 'none', sw: 0 };
      return { ...at, r: hit.now ? 7.5 : 5.5, fill: CLS[hit.pk.label], stroke: MODEL[hit.pk.by], sw: hit.now ? 3.5 : 2 };
    });
  }

  function lines(st, p) {
    const done = p === 2;
    const A = done ? st.A2 : st.A, B = done ? st.B2 : st.B;
    return {
      ctLineA: vline(A.t), ctLineB: hline(B.t),
      ctShowOld: done, ctOldA: vline(st.A.t), ctOldB: hline(st.B.t),
      ctTA: f2(A.t), ctTB: f2(B.t)
    };
  }

  function render(c) {
    const { data, steps } = model(), s = c.state.ct, st = steps[s.k], done = s.p === 2;
    return {
      ...W().bindings(c, 'ct', 'ct', phases(data, st), steps.length),
      ...lines(st, s.p),
      ctDots: dots(data, steps, s.k, s.p),
      ctTicksB: st.picksA.map(tickOnB(data)),
      ctTicksA: s.p >= 1 ? st.picksB.map(tickOnA(data)) : [],
      ctGridX: TICKS.map(v => ({ x: f1(sx(v)), y1: f1(sy(LIM)), y2: f1(sy(0)) })),
      ctGridY: TICKS.map(v => ({ y: f1(sy(v)), x1: f1(sx(0)), x2: f1(sx(LIM)) })),
      ctInspect: inspect(data, st, s.p),
      ctNote: NOTES[s.p],
      ctAccA: pct(M().accuracy(done ? st.A2 : st.A, data, 0)),
      ctAccB: pct(M().accuracy(done ? st.B2 : st.B, data, 1))
    };
  }

  function initState() {
    return { ct: { k: 0, p: 0 } };
  }

  return { initState, render };
})();
