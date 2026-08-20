# Speculative decoding

Decoding one token requires one pass over the whole model. At batch size 1 that
pass moves far more weight through memory than it does arithmetic, so the
hardware spends most of its time waiting on bandwidth. Running the same pass over
*several* candidate tokens costs almost the same wall-clock as running it over
one.

Speculative decoding spends that headroom. A cheap draft model proposes `k`
tokens; the expensive target model checks all `k` in a single pass. Every
proposal the target agrees with is a token produced without a target call of its
own.

## The output is not an approximation

The natural suspicion is that this trades quality for speed. It does not, and
that is the whole reason the technique is interesting rather than merely fast.

**Greedy.** A proposal is kept only when it equals the target's own argmax.
Identical by construction.

**Sampling.** A proposal `x` drawn from the draft's `q` is kept with probability
`min(1, p(x)/q(x))`. On rejection the replacement is drawn from the normalised
residual `max(0, p - q)`. Those two branches compose to exactly `p`.

So the draft's quality affects **speed only**. A poor draft is rejected more
often and saves less; it cannot change the distribution the tokens come from.

Both claims are tested rather than asserted. Greedy equality is checked token for
token against an ordinary decoding loop. The sampling claim is checked
statistically: 3,000 single-token generations against a deliberately skewed draft
that forces the rejection branch to carry about 70% of the traffic, comparing the
empirical distribution to the target's own softmax. The test asserts on the mass
of the *drafted* token specifically, because that is where a broken rule leaks —
substituting a plain target draw for the residual draw nearly doubles that one
entry (0.068 to 0.118) while moving total variation only from 0.023 to 0.055, and
an aggregate bound loose enough not to flake would let it through.

## Progress cannot stall

On rejection the target's own token for that position is emitted, so every target
call yields at least one token no matter how badly the draft performs. When all
`k` proposals are accepted the trailing logits supply a free bonus token, giving
`k + 1` tokens from one call.

## Measured

The quantity that matters is **tokens per target call** — target calls dominate
cost, so that ratio is the speedup ceiling, and it is 1.0 without speculation.

The acceptance rate is an *input* here, not a measurement. Randomly initialised
models agree with each other at a rate that reflects their shared degeneracy
rather than any real draft/target relationship, and reporting that as an
acceptance rate would be meaningless. A synthetic draft agrees with a chosen
probability instead, which gives a curve over the parameter a real deployment
would have to measure — and one that can be checked against theory.

For i.i.d. acceptance `a` and lookahead `k`, the accepted count is geometric
capped at `k` and every block emits one further token, so

    E[tokens per target call] = (1 - a^(k+1)) / (1 - a)

    python benchmarks/benchmark_speculative.py

| acceptance | k=1  | k=2  | k=4  | k=8  |
| ---------: | :--- | :--- | :--- | :--- |
|        0.3 | 1.31 | 1.40 | 1.45 | 1.48 |
|        0.5 | 1.50 | 1.75 | 1.94 | 1.99 |
|        0.7 | 1.73 | 2.18 | 2.79 | 3.06 |
|        0.9 | 1.91 | 2.67 | 4.23 | 6.00 |

Measured, against closed-form 1.30 / 1.39 / 1.43 / 1.43, 1.50 / 1.75 / 1.94 /
2.00, 1.70 / 2.19 / 2.77 / 3.20, 1.90 / 2.71 / 4.10 / 6.13. The agreement is
evidence the implementation is correct — it is not itself a performance claim.

These are the default run's figures — the command above, with no flags. Unlike
the closed form beside them they are Monte-Carlo estimates: acceptance is drawn
per token, so they move with `--tokens` and would not reproduce digit for digit
from a shorter run. The estimate tightens as the sample grows; the closed form
is what it converges toward.

### Reading the table

Returns saturate fast at low acceptance. At `a = 0.5` the ceiling is already 1.94
by `k = 4` and can never exceed 2.0, so a larger lookahead only buys more wasted
draft work. Lookahead is worth raising only when the draft is genuinely good:
the `a = 0.9` row is still climbing at `k = 8`.

### This is a ceiling, not a speedup

Tokens per target call ignores what the draft costs. The realisable gain is

    ceiling / (1 + k * draft_cost / target_cost)

so a draft costing a tenth of the target at `k = 4` keeps roughly 5/7 of the
ceiling. Both cost terms are hardware-dependent and **none of them are measured
here** — no timing in this document comes from a GPU.

### On the acceptance-rate metric

`SpeculativeStats.acceptance_rate` is accepted over proposals *made*, not the
per-token probability that the draft agrees. A block stops at its first
rejection, but the draft has already generated the whole lookahead, and those
unused proposals still count — they cost draft time. So the figure falls as `k`
rises even for an equally good draft: a true 0.9 agreement reads 0.92 at `k = 1`
and 0.62 at `k = 8`. It answers "how much draft work was wasted", not "how good
is the draft". Compare lookaheads with `tokens_per_target_call`.

## What is still missing

* **Batch size 1 only.** Batched speculation needs per-row acceptance lengths,
  which makes the KV cache ragged in a way the contiguous cache cannot express —
  the same limitation described in
  [continuous-batching.md](continuous-batching.md#what-is-still-missing).
* **No GPU timing.** The ceiling is hardware-independent; converting it to a
  speedup is not, and that conversion has not been measured.
* **No self-speculation.** Drafting with the target's own early layers, or with
  n-gram lookup, would remove the second model. Neither is implemented.
