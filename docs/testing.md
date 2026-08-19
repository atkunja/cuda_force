# Testing

## What runs where

| Suite | Count | Needs | Runs on the dev host? |
| --- | --- | --- | --- |
| C++ (`tests/cpp`) | 87 cases, 4,409 assertions | C++20 compiler | yes |
| C++ under TSan / ASan / UBSan | same 87 | clang | yes, all three clean |
| Python (`tests/python`) | 251 | Python + PyTorch | yes |
| CUDA (`tests/cuda`) | 30+ cases | NVIDIA GPU | **no** |
| CUDA structural checks | 22 files | nothing | yes |

```bash
./scripts/test.sh          # everything this machine can run, with skips reported
```

`test.sh` prints `[SKIP]` with a reason for anything it cannot run. A suite that
silently skips is worse than one that fails, because it looks like a pass.

## What the C++ tests assert

Concurrency bugs do not reproduce on demand, so the tests assert *properties*
rather than sequences:

* capacity is never exceeded, observed by an independent watcher thread;
* every produced item is consumed exactly once, across 6 producers and 5
  consumers;
* blocked producers and consumers are both released by shutdown;
* accepted work survives shutdown — a closed-but-non-empty queue still returns
  items;
* a bounded queue caps depth under a producer faster than the pool;
* a throwing task does not reduce pool capacity;
* the batch deadline is anchored to the oldest request, verified by trickling
  arrivals faster than the deadline and requiring batches to still close.

### Assertions never run on worker threads

Catch2's `REQUIRE` family is not thread-safe: the macros manipulate per-run
state including an output redirect that asserts if two threads activate it at
once. This surfaced here as a sporadic `SIGABRT` under AddressSanitizer, whose
timing depended on the scheduler and looked unrelated to the code under test.

Concurrency tests now route worker-thread checks through
[`thread_assert.hpp`](../tests/cpp/thread_assert.hpp) — an atomic failure
counter — and assert once on the main thread after joining.

## What the Python tests assert

Beyond the obvious shape and value checks, several tests exist to pin down
properties that would otherwise regress silently:

| Property | Test |
| --- | --- |
| RMSNorm survives fp16 magnitudes that overflow when squared | `test_rmsnorm_survives_float16_magnitudes_that_would_overflow` |
| Softmax is finite for ±1000 logits | `test_softmax_is_stable_for_large_logits` |
| Int8 round-trip error stays under `scale / 2` | `test_quantization_round_trip_stays_within_the_theoretical_bound` |
| An outlier degrades only its own block | `test_an_outlier_only_degrades_its_own_block` |
| LoRA with B=0 is exactly the base model | `test_b_starts_at_zero_so_the_adapted_model_matches_the_base` |
| Merging a LoRA layer is exact, not approximate | `test_merging_is_exact` |
| Gradient accumulation equals one large step | `test_gradient_accumulation_matches_a_single_large_step` |
| Packed labels do not alias inputs | `test_labels_do_not_alias_the_inputs` |
| Outstanding futures are settled at shutdown | `test_outstanding_futures_are_failed_at_shutdown` |

The last one found a real bug: a runner returning the wrong row count raised
outside the engine's guard, so no future was completed and every caller blocked
until its timeout.

## Tolerances

CUDA tests compare against a **double-precision** host reference — the
mathematical answer, not another float32 accumulation with its own error — and
scale the tolerance with the term count, since floating-point addition is not
associative and a tree reduction genuinely differs from a sequential sum.

Where the input is exactly representable (a sum of ones below 2^24), the test
demands exact equality: any deviation there is a bug, not accumulated error.

## Determinism

Every test seeds explicitly. An unseeded generator makes a marginal tolerance
failure appear intermittently, which is far harder to diagnose than a consistent
one — and floating-point kernels sit close enough to their tolerances for this
to matter.

`tests/python/conftest.py` seeds torch before every test via an autouse fixture,
and skips `cuda`-marked tests with a stated reason when no GPU is present.
