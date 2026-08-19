# Testing

## What runs where

| Suite | Count | Needs | Runs on the dev host? |
| --- | --- | --- | --- |
| C++ (`tests/cpp`) | 164 cases, 49,876 assertions | C++20 compiler | yes |
| C++ under TSan / ASan / UBSan | same 164 | clang | yes, all three clean |
| Python (`tests/python`) | 450 | Python + PyTorch | yes |
| CUDA (`tests/cuda`) | 69 cases | NVIDIA GPU | **no** |
| CUDA structural checks | 29 files | nothing | yes |

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

### CUDA logic that is tested without CUDA

Not everything under `cuda/` needs a GPU to be checked. Launch geometry —
grid sizes, block sizes, warp counts — is pure arithmetic that happens to
produce CUDA launch parameters, and getting it wrong is a real bug class: a
kernel that silently processes only part of its input.

It therefore lives in [`cpp/include/cudaforge/launch_config.hpp`](../cpp/include/cudaforge/launch_config.hpp),
which has no CUDA include, and `cuda_utils.cuh` includes it. That single move
puts the following under test on a machine with no toolkit:

* `ceil_div` covers every element and never over-launches by a whole block;
* `block_size_for_row` is always a power of two, at least one warp, and capped
  at 1024;
* `grid_size_for_stride_loop` never returns zero, never launches idle blocks for
  a small input, and is capped by occupancy rather than by input size for a
  large one.

The remaining hardware dependency is one device query for the SM count, which is
passed in as a parameter.

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

## Cross-implementation conformance

The batcher exists twice — once in C++, once in Python — and the documentation
claims they implement the same policy. Each was tested only against its own
expectations, which is exactly how two implementations drift apart while both
suites stay green.

`tests/python/test_batcher_conformance.py` runs the *same* scenario through
both and compares. The C++ half is a standalone harness,
[`batcher_scenario.cpp`](../tests/cpp/batcher_scenario.cpp), that emits JSON;
the Python test drives it and its own batcher over identical parameters.

Exact equality is deliberately not asserted, and would be wrong to assert: the
two are separately scheduled, so batch-by-batch sizes legitimately differ. What
must agree is the *policy*:

| Property | Scenario |
| --- | --- |
| No request is lost | saturated, trickle, single producer, batch-of-one |
| No batch exceeds the size limit | all four |
| A saturated queue closes mostly on size, not on the deadline | four producers, no gap |
| A batch size of one never aggregates | the degenerate configuration |
| A saturated queue aggregates more than one request | eight producers, no gap |

Skipped with a stated reason when the harness has not been built.

### Shared constants and defaults

Several values are defined twice, once per language, because neither can import
the other's. A divergence would not fail to compile — it would produce a
numerical mismatch or a behaviour difference that the documentation then
describes wrongly for one of the two runtimes.

`tests/python/test_constant_parity.py` reads the C++ headers and compares:

| Shared | Consequence of drift |
| --- | --- |
| `kQuantBlockSize` | different block scales, so different dequantised values |
| `kWarpSize`, `kDefaultBlockSize` | the warp primitives assume 32 at compile time |
| Every batching default | `docs/concurrency.md` describes one policy for both |
| Every metrics field name | a dashboard would need to know which runtime produced a snapshot |

Textual comparison, deliberately: the alternative is a build-time code
generator, which is more machinery than a handful of constants justifies.

## Tolerances

CUDA tests compare against a **double-precision** host reference — the
mathematical answer, not another float32 accumulation with its own error — and
scale the tolerance with the term count, since floating-point addition is not
associative and a tree reduction genuinely differs from a sequential sum.

Where the input is exactly representable (a sum of ones below 2^24), the test
demands exact equality: any deviation there is a bug, not accumulated error.

## Coverage

```bash
pytest tests/python -q --cov --cov-report=term-missing
```

**91% of statements, 88% of branches.** There is deliberately no `fail_under`
threshold: a number that must not drop invites tests written to raise it rather
than to catch anything.

What remains uncovered is uncovered for a stated reason:

| Uncovered | Why |
| --- | --- |
| `train.build_model` | downloads model weights; exercised end to end in CI instead |
| `dataset.load_texts` with a named dataset | downloads a dataset |
| CUDA branches in `ops.py` | need a GPU; the reference branches beside them are covered |
| `TransformersRunner`'s loading path | downloads weights — its *logic* is covered through injection |

The two refactors that made the difference were injection points: `train()` and
`TransformersRunner` both accept a model and a loader, so the parts most likely
to be subtly wrong — accumulation, the unscale-then-clip ordering, left padding,
per-row truncation — are tested without a network round trip. Code that can only
be run by downloading something is code that does not get tested.

## Isolation

Tests that touch process-wide state restore it. `cli.serve` configures the
server by writing `CUDAFORGE_*` environment variables — the right interface for
a CLI, and the wrong thing to leave behind in a test run. An autouse fixture in
`conftest.py` snapshots and restores them.

This was not hypothetical: without it, a CLI test's variables overrode a later
server test's config file, and the failure appeared only when the whole suite
ran rather than when the test did. That is the worst shape a test failure can
take, because the obvious next step — running the failing test alone — makes it
disappear.

## Determinism

Every test seeds explicitly. An unseeded generator makes a marginal tolerance
failure appear intermittently, which is far harder to diagnose than a consistent
one — and floating-point kernels sit close enough to their tolerances for this
to matter.

`tests/python/conftest.py` seeds torch before every test via an autouse fixture,
and skips `cuda`-marked tests with a stated reason when no GPU is present.
