"""corpus_mix.py — deterministic multi-corpus weighted interleaving for bitdistill.

Build item **B4** of the capability-distillation plan
(`.planning/2026-07-12-capability-distillation-design.md`, §3 core-vs-blend).

The BitDistill trainer (`bitdistill.py::train`) consumes a *token stream* — an
iterator that yields fixed-length ``[seq_len]`` int64 tensors, one per call to
``next(stream)`` (see ``bitdistill_data.cpt_token_stream``). This module produces
a **single** such stream that interleaves several corpora at configurable weight
ratios, so a run can mix tool-use / C# / instruction capability data with a
general anchor (default **tool-use 35 / C# 35 / instruction 20 / general 10**).

Design contract honoured
------------------------
* **Same element contract** — each input corpus yields ``[seq_len]`` int64
  tensors; the mixer forwards them unchanged (it never inspects/reshapes them),
  so any corpus that satisfies the ``cpt_token_stream`` contract plugs in.
* **Deterministic** — the interleave order is a fixed schedule (smooth weighted
  round-robin / error-diffusion by default), or a *seeded* index sequence. No
  ``random`` without a seed, no wall-clock. Two runs with the same inputs and
  seed draw corpora in the identical order.
* **Long-run ratio fidelity** — over N draws each corpus is drawn
  ``round(N * weight)`` times (±1 with the stride schedule).
* **Generic** — takes stream *factories* or iterables/iterators as arguments;
  it knows nothing about tool-use vs C# vs general. The convenience builder
  ``standard_capability_mix`` supplies the plan's default weights and tolerates
  a missing corpus by renormalizing.
* **Cycling / refresh** — an exhausted finite corpus (e.g. a materialised list
  of capability sequences) is transparently restarted; an infinite corpus
  (streaming cpt) simply never exhausts. A corpus given as a bare, non-reusable
  iterator is consumed once and then dropped (weights renormalize over the rest).

Public API
----------
* ``mixed_token_stream(named_streams, weights, *, schedule, seed, ...)``
* ``standard_capability_mix(factories, weights=None, **kw)``
* ``curriculum_stream(...)`` — optional thin 2-phase curriculum wrapper.

CPU-only, no model / GPU / download here — pure iterator plumbing.
"""

from __future__ import annotations

import random
from typing import Callable, Iterable, Iterator, Optional, Union

# A corpus source is anything that can produce ``[seq_len]`` tensors:
#   * a zero-arg factory returning a fresh iterator   (preferred — refreshable)
#   * a re-iterable container (list/tuple of tensors) (refreshable via iter())
#   * a bare iterator/generator                       (single-pass, non-refreshable)
CorpusSource = Union[Callable[[], Iterable], Iterable]


# ---------------------------------------------------------------------------
# Default capability blend (design §3 / DECISIONS LOCKED 2026-07-12)
# ---------------------------------------------------------------------------
STANDARD_WEIGHTS: dict = {
    "tooluse": 35.0,
    "csharp": 35.0,
    "instruction": 20.0,
    "general": 10.0,
}


# ---------------------------------------------------------------------------
# Source normalization — turn any CorpusSource into a forever-yielding
# generator that refreshes finite sources and single-passes bare iterators.
# ---------------------------------------------------------------------------
def _is_iterator(obj) -> bool:
    """True if ``obj`` is an iterator (``iter(obj) is obj``), e.g. a generator."""
    return hasattr(obj, "__next__")


def _refreshable(value: CorpusSource) -> Iterator:
    """Yield elements from ``value`` forever, refreshing finite sources.

    * **factory** (callable, not itself an iterator): call it for a fresh
      iterator each time the previous one is exhausted → infinite cycling.
    * **re-iterable** (list/tuple/…): ``iter(value)`` gives a fresh pass each
      cycle → infinite cycling over the fixed contents.
    * **bare iterator/generator**: consumed exactly once (cannot be restarted),
      then this generator returns → the mixer drops the corpus and renormalizes.

    A source that produces zero elements on a pass returns immediately rather
    than spinning forever, so an empty/missing corpus can't hang the mixer.
    """
    if callable(value) and not _is_iterator(value):
        while True:
            produced = False
            for x in value():
                produced = True
                yield x
            if not produced:
                return
    elif _is_iterator(value):
        # Non-reusable: single pass. (An infinite iterator such as the streaming
        # cpt corpus never exhausts, so it is effectively unbounded anyway.)
        for x in value:
            yield x
        return
    else:
        # Re-iterable container: cycle by re-``iter()``-ing each pass.
        while True:
            produced = False
            for x in value:
                produced = True
                yield x
            if not produced:
                return


# ---------------------------------------------------------------------------
# Core mixer
# ---------------------------------------------------------------------------
def mixed_token_stream(
    named_streams: dict,
    weights: dict,
    *,
    schedule: str = "stride",
    seed: int = 0,
    strict: bool = True,
) -> Iterator:
    """Deterministically interleave named corpora at the given weight ratios.

    Parameters
    ----------
    named_streams:
        ``{name: source}`` where each *source* is a :data:`CorpusSource` yielding
        the fixed-length ``[seq_len]`` int64 tensor contract. Insertion order is
        the tie-break order for the schedule (so results are stable).
    weights:
        ``{name: weight}``. Need not sum to 1 or 100 — they are normalized over
        the corpora that are actually present with a **positive** weight. A name
        in ``weights`` but absent from ``named_streams`` is ignored; a name in
        ``named_streams`` with no (or non-positive) weight is not drawn.
    schedule:
        * ``"stride"`` (default) — smooth weighted round-robin (error diffusion).
          Fully deterministic, no seed needed; each corpus is drawn
          ``round(N*w)`` times (±1) over any prefix of N draws — tighter than
          random and drift-free.
        * ``"random"`` — seeded weighted choice per draw (``random.Random(seed)``).
          Deterministic given ``seed``; proportions converge in the long run with
          normal sampling variance.
    seed:
        Seed for the ``"random"`` schedule (ignored by ``"stride"``).
    strict:
        If True, raise when ``named_streams`` is empty or no corpus has a
        positive weight. If False, an empty active set yields nothing.

    Yields
    ------
    ``torch.Tensor`` of shape ``[seq_len]`` — whatever the chosen corpus yields,
    forwarded unchanged.

    Notes
    -----
    A corpus that exhausts (a bare, non-refreshable iterator that ends) is
    removed and the remaining weights are renormalized on the fly, so the mix
    degrades gracefully to the survivors. When every corpus has exhausted, the
    stream ends.
    """
    if schedule not in ("stride", "random"):
        raise ValueError(f"schedule must be 'stride' or 'random', got {schedule!r}")

    names = list(named_streams.keys())
    w = {n: float(weights.get(n, 0.0)) for n in names}
    active = [n for n in names if w[n] > 0.0]

    if not active:
        if strict:
            raise ValueError(
                "mixed_token_stream: no corpus has a positive weight "
                f"(streams={names}, weights={weights})"
            )
        return

    iters = {n: _refreshable(named_streams[n]) for n in active}
    rng = random.Random(seed) if schedule == "random" else None

    def _normalized(subset: list) -> dict:
        s = sum(w[n] for n in subset)
        return {n: w[n] / s for n in subset}

    norm = _normalized(active)
    deficit = {n: 0.0 for n in active}  # error-diffusion accumulators (stride)

    def _pick() -> str:
        if schedule == "random":
            r = rng.random()
            acc = 0.0
            for n in active:
                acc += norm[n]
                if r < acc:
                    return n
            return active[-1]  # float-rounding guard
        # stride: smooth weighted round-robin.
        for n in active:
            deficit[n] += norm[n]
        # max() returns the FIRST maximal element → ties break by insertion order.
        pick = max(active, key=lambda n: deficit[n])
        deficit[pick] -= 1.0
        return pick

    while active:
        pick = _pick()
        try:
            yield next(iters[pick])
        except StopIteration:
            # Corpus exhausted (only possible for a bare, non-refreshable
            # iterator). Drop it and renormalize over the survivors.
            active.remove(pick)
            iters.pop(pick, None)
            deficit.pop(pick, None)
            if not active:
                return
            norm = _normalized(active)


# ---------------------------------------------------------------------------
# Convenience builder — the plan's standard capability blend
# ---------------------------------------------------------------------------
def standard_capability_mix(
    factories: dict,
    weights: Optional[dict] = None,
    *,
    schedule: str = "stride",
    seed: int = 0,
) -> Iterator:
    """Build the plan's default capability mix from a dict of corpus sources.

    Parameters
    ----------
    factories:
        ``{name: source}`` for any subset of the canonical corpora
        ``"tooluse"``, ``"csharp"``, ``"instruction"``, ``"general"`` (and/or
        extra custom names). Each *source* is a :data:`CorpusSource`. Only the
        corpora actually supplied are mixed — a missing one is simply omitted
        and the remaining weights renormalize (so a tool-use + instruction +
        general run works before the C# training stream exists).
    weights:
        Override the default ratios. Defaults to
        **tool-use 35 / C# 35 / instruction 20 / general 10**. Keys absent from
        ``factories`` are ignored; extra keys present in both are honoured.

    Returns
    -------
    A ``mixed_token_stream`` iterator over the supplied corpora.
    """
    base = dict(STANDARD_WEIGHTS)
    if weights:
        base.update({k: float(v) for k, v in weights.items()})
    # Restrict to corpora that were actually provided.
    present = {n: base.get(n, 0.0) for n in factories.keys()}
    if not any(v > 0.0 for v in present.values()):
        raise ValueError(
            "standard_capability_mix: none of the provided corpora "
            f"{list(factories.keys())} has a positive weight (weights={base})"
        )
    return mixed_token_stream(
        factories, present, schedule=schedule, seed=seed
    )


# ---------------------------------------------------------------------------
# Optional: 2-phase curriculum wrapper (design §3 secondary lever)
# ---------------------------------------------------------------------------
def curriculum_stream(
    factories: dict,
    *,
    seq_len: int,
    total_tokens: float,
    phase_a_fraction: float = 0.15,
    phase_a_weights: Optional[dict] = None,
    phase_b_weights: Optional[dict] = None,
    schedule: str = "stride",
    seed: int = 0,
) -> Iterator:
    """Thin 2-phase curriculum over the mixer (design §3).

    **Phase A** (first ``phase_a_fraction`` of ``total_tokens``): a
    stabilisation blend — by default general + instruction only — to settle the
    freshly-ternarized student before capability-heavy training.
    **Phase B** (remainder): the full capability-heavy blend (default
    :data:`STANDARD_WEIGHTS`).

    The switch is counted in *emitted sequences*: each yielded tensor is
    ``seq_len`` tokens, and the trainer accounts tokens as ``batch.numel()`` —
    so ``phase_a_seqs = round(total_tokens * phase_a_fraction / seq_len)``.
    Only corpora present in ``factories`` participate in each phase (weights
    renormalize), so an absent corpus never blocks a phase.

    This is a *secondary* lever — start with plain ``standard_capability_mix``
    and reach for the curriculum only if a target plateaus.
    """
    if not (0.0 <= phase_a_fraction < 1.0):
        raise ValueError(f"phase_a_fraction must be in [0, 1), got {phase_a_fraction}")

    if phase_a_weights is None:
        # Default stabilisation blend: general + broad instruction.
        phase_a_weights = {"general": 50.0, "instruction": 50.0}

    phase_a_seqs = int(round(float(total_tokens) * phase_a_fraction / float(seq_len)))

    if phase_a_seqs > 0:
        # Phase A is an EXPLICIT blend: only the corpora named in phase_a_weights
        # participate (restrict factories so capability corpora don't leak in via
        # standard defaults). Use mixed_token_stream directly, not the standard
        # builder, which would merge onto the full default weights.
        a_factories = {n: factories[n] for n in phase_a_weights if n in factories}
        if not a_factories:
            raise ValueError(
                "curriculum_stream: phase_a_weights names no corpus present in "
                f"factories (phase_a={list(phase_a_weights)}, have={list(factories)})"
            )
        phase_a = mixed_token_stream(
            a_factories, phase_a_weights, schedule=schedule, seed=seed
        )
        emitted = 0
        for x in phase_a:
            yield x
            emitted += 1
            if emitted >= phase_a_seqs:
                break

    # Phase B: capability-heavy. Defaults to STANDARD_WEIGHTS over all present
    # corpora; a custom phase_b_weights is merged onto those defaults.
    # Distinct seed so the two phases don't share an identical draw order.
    phase_b = standard_capability_mix(
        factories, phase_b_weights, schedule=schedule, seed=seed + 1
    )
    for x in phase_b:
        yield x


# ---------------------------------------------------------------------------
# Self-test (CPU only, no torch model, no download)
# ---------------------------------------------------------------------------
def _selftest() -> None:
    """Validate ratio fidelity + determinism with tagged dummy iterators.

    Each dummy corpus yields a distinct constant int so the realized draw
    proportions can be counted directly.
    """
    def tagged(tag: int):
        # Infinite, refreshable factory yielding a 1-elem [seq_len]-style vector.
        def factory():
            while True:
                yield [tag]
        return factory

    factories = {
        "tooluse": tagged(0),
        "csharp": tagged(1),
        "instruction": tagged(2),
        "general": tagged(3),
    }
    weights = STANDARD_WEIGHTS
    N = 1000

    def draw(seed_stream):
        counts = {k: 0 for k in factories}
        order = list(factories.keys())
        for i, v in enumerate(seed_stream):
            counts[order[v[0]]] += 1
            if i + 1 >= N:
                break
        return counts

    s1 = mixed_token_stream(factories, weights, schedule="stride")
    c1 = draw(s1)
    total_w = sum(weights.values())
    print(f"[corpus_mix:selftest] N={N} stride draws:")
    ok = True
    for k in factories:
        want = N * weights[k] / total_w
        got = c1[k]
        drift = abs(got - want)
        flag = "OK" if drift <= 1.5 else "BAD"
        if flag == "BAD":
            ok = False
        print(f"  {k:12s} want~{want:6.1f}  got {got:5d}  drift {drift:4.1f}  [{flag}]")

    # Determinism: two independent stride streams draw identically.
    a = list(x[0] for x in _take(mixed_token_stream(factories, weights), N))
    b = list(x[0] for x in _take(mixed_token_stream(factories, weights), N))
    det = a == b
    print(f"[corpus_mix:selftest] determinism (stride, 2 runs identical): {det}")

    # Missing-corpus tolerance: drop C#, weights renormalize.
    part = {k: factories[k] for k in ("tooluse", "instruction", "general")}
    cs = standard_capability_mix(part)
    cpart = {k: 0 for k in part}
    order = list(factories.keys())
    for i, v in enumerate(cs):
        cpart[order[v[0]]] += 1
        if i + 1 >= N:
            break
    sub_w = sum(STANDARD_WEIGHTS[k] for k in part)
    print(f"[corpus_mix:selftest] N={N} standard mix w/ C# ABSENT (renormalized):")
    for k in part:
        want = N * STANDARD_WEIGHTS[k] / sub_w
        print(f"  {k:12s} want~{want:6.1f}  got {cpart[k]:5d}")

    assert ok and det, "corpus_mix self-test FAILED"
    print("[corpus_mix:selftest] PASSED")


def _take(it: Iterator, n: int) -> list:
    out = []
    for i, x in enumerate(it):
        out.append(x)
        if i + 1 >= n:
            break
    return out


if __name__ == "__main__":
    _selftest()
