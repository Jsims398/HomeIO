// Shared "round m, step p" walkthrough controls for the boosting pages (Topics 16 and 17).
// Each walkthrough keeps { k, p } in one state key: k = rounds already finished, p = step inside round k + 1.
// Loaded by boosting.html and gradient-boosting.html before their *-view.js files.
window.Walkthrough = (function () {
  const SKIP = 5;

  const patch = (c, key, change) => c.setState({ [key]: { ...c.state[key], ...change } });

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

  // bindings every walkthrough needs, under a prefix: xNext, xBack, xSkip, xReset, xLabel, xNextLabel, xCards
  function bindings(c, key, prefix, phases, maxRounds) {
    const ctl = controls(c, key, phases.length, maxRounds);
    return {
      [prefix + 'Next']: ctl.next, [prefix + 'Back']: ctl.back, [prefix + 'Skip']: ctl.skip, [prefix + 'Reset']: ctl.reset,
      [prefix + 'Label']: ctl.label, [prefix + 'NextLabel']: ctl.nextLabel, [prefix + 'Cards']: stepCards(c, key, phases)
    };
  }

  // the inspector text grows one sentence per step reached
  const upTo = (p, sentences) => sentences.slice(0, p + 1).join(' ');

  return { patch, bindings, upTo };
})();
