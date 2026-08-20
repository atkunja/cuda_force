# API Reference

The public surface, and the contract each part guarantees. Everything here is
importable from the package root.

```python
import cudaforge
```

## Operators

Each dispatches to the compiled CUDA kernel when one is available, and to a
reference PyTorch implementation otherwise. Results agree either way; only the
implementation differs.

### `rmsnorm(x, weight, eps=1e-6) -> Tensor`

Root-mean-square normalisation over the last dimension.

| Argument | Shape | Notes |
| --- | --- | --- |
| `x` | `[..., hidden]` | made contiguous if it is not |
| `weight` | `[hidden]` | learned gain |
| `eps` | scalar | added **inside** the square root |

Raises `ValueError` if `weight` is not 1-D or does not match `x`'s last
dimension. Float16 inputs are promoted to float32 before squaring — at
magnitude 256 the square overflows float16 and poisons the whole row.

### `softmax(x) -> Tensor`

Numerically stable softmax over the last dimension. The row maximum is
subtracted first, which is an exact identity and is what keeps logits above ~88
from overflowing.

### `lora_linear(x, weight, lora_a, lora_b, scale=1.0) -> Tensor`

`x @ weight + scale * ((x @ lora_a) @ lora_b)`.

| Argument | Shape |
| --- | --- |
| `x` | `[batch, in_features]` |
| `weight` | `[in_features, out_features]` — frozen base |
| `lora_a` | `[in_features, rank]` |
| `lora_b` | `[rank, out_features]` |

`scale` is usually `alpha / rank`. Raises `ValueError` on any shape mismatch,
naming which operand disagrees.

### `silu(x) -> Tensor`

SiLU, also called swish: `x * sigmoid(x)`. Smooth and non-monotonic, and its
gradient does not vanish for negative inputs — the property that displaced ReLU
in the transformer feed-forward block.

### `gelu(x) -> Tensor`

GELU, **tanh approximation**. That form specifically, because GPT-2 and BERT
were trained against this curve; the exact erf form changes outputs by more than
the numerical difference suggests.

### `swiglu(gate, up) -> Tensor`

`silu(gate) * up`, the LLaMA-family feed-forward activation. Both operands must
match in shape and dtype.

Fusing the activation with the multiply takes it from three passes over memory
to one read of each input and one write — the theoretical minimum, which is why
it is a kernel rather than two framework calls.

### `fused_residual_rmsnorm(x, residual, weight, eps=1e-6) -> (Tensor, Tensor)`

Residual add followed by RMSNorm, in one pass. Returns
`(normalised, residual_out)`.

Both outputs are needed: the first feeds the next sublayer, the second is what
the *following* residual connection adds to. Returning only the first would
force the caller to recompute the sum — and is an easy mistake that silently
changes the model rather than failing.

This is the highest-frequency fusion in inference, because every transformer
block is `x = x + sublayer(norm(x))` twice over.

### `sum_reduce(x) -> Tensor`

Sum of every element, as a 0-D tensor. Returns zero for an empty input.

### `quantize_int8(x) -> (Tensor, Tensor)`

Block-wise symmetric INT8 quantisation. Returns `(quantised, scales)` where
`quantised` matches `x`'s shape and `scales` has one entry per
`QUANT_BLOCK_SIZE` (64) elements.

### `dequantize_int8(quantised, scales) -> Tensor`

Inverse of the above. Lossy by construction, with round-trip error bounded by
half a quantisation step — `scales.max() / 2`. Raises `ValueError` if the scale
count does not match the element count.

### `kernel_supports(dtype) -> bool`

Whether a CUDA kernel exists for a storage dtype: float32, float16 and
bfloat16. Anything else takes the reference path, which handles every dtype
ATen does.

Worth checking before benchmarking — an unsupported dtype produces a timing that
says nothing about the kernel. `KERNEL_DTYPES` is the underlying set, and
`CUDA_KERNELS_AVAILABLE` is the boolean for whether any of them can actually
run here: it requires both a CUDA-compiled extension *and* a visible device.

BF16 needs compute capability 8.0 or later; below that it is emulated and would
be slower than the FP16 path it replaces.

### `backend_report() -> BackendReport`

Which implementation path is active, and why.

```python
report = cudaforge.backend_report()
report.using_custom_kernels  # bool — the only field that matters for timings
report.extension_loaded  # the compiled module imported
report.cuda_compiled  # it was built with CUDA support
report.cuda_device_available  # a GPU is visible
```

Call this before drawing any conclusion from a measurement. A silent fallback
to the reference path looks exactly like a very slow custom kernel.

## Engine

### `InferenceEngine(config=None, runner=None, metrics=None)`

Concurrent, dynamically batched inference. Thread-safe: `submit` may be called
from any number of threads.

```python
with cudaforge.InferenceEngine(config=cudaforge.EngineConfig()) as engine:
    response = engine.generate("Explain CUDA warps.")
```

| Method | Returns | Notes |
| --- | --- | --- |
| `submit(prompt, generation=None, block_when_full=True, deadline_seconds=None)` | `Future[Response]` | returns immediately |
| `generate(prompt, generation=None, timeout=60.0)` | `Response` | blocking wrapper |
| `generate_many(prompts, generation=None, timeout=120.0)` | `list[Response]` | submits all, then collects |
| `snapshot()` | `MetricsSnapshot` | counters and percentiles |
| `shutdown(timeout=30.0)` | `None` | drains, then settles every outstanding future |

`generate_many` submits every prompt before waiting on any. That is the point:
it gives the batcher several requests to aggregate. Submitting and waiting one
at a time produces batches of one.

`block_when_full=False` sheds load instead of applying backpressure — raises
`EngineClosedError` rather than queueing. Which is correct depends on the
caller: a batch client wants backpressure, an HTTP frontend generally wants to
return 503.

`deadline_seconds` drops the request if it is still waiting past that point,
completing its future with a `RequestExpired` error. Under load this stops the
runtime spending capacity on work nobody is waiting for.

### `Response`

| Field | Meaning |
| --- | --- |
| `request_id` | assigned at ingress |
| `text` | generated text; unspecified when `error` is set |
| `prompt_tokens`, `generated_tokens` | token counts |
| `queue_time`, `inference_time` | seconds; reported separately because they point at different problems |
| `total_latency` | their sum |
| `batch_size` | how many requests ran together |
| `error` | `None` on success |
| `ok` | `error is None` |

## Batching

### `DynamicBatcher(handler, max_batch_size=16, max_wait_seconds=0.005, queue_capacity=1024, metrics=None, on_expired=None)`

Aggregates concurrent requests on a background thread. A batch closes on
whichever comes first: `max_batch_size` requests, or the **oldest** request
having waited `max_wait_seconds`. The deadline is anchored, never extended — so
no request waits longer than `max_wait_seconds` plus its batch's service time.

| Method | Behaviour when full |
| --- | --- |
| `submit(request, timeout=None)` | blocks |
| `try_submit(request)` | returns `False` |

### `ContinuousBatcher(runner, on_complete, max_batch_size=16, queue_capacity=1024, ...)`

Iteration-level scheduling: rows freed by finished sequences are refilled at the
next decode step instead of at the end of the batch. Takes a `StepwiseRunner`
rather than a `ModelRunner`, because the scheduler has to see between steps.

| Method | Behaviour |
| --- | --- |
| `submit(request, timeout=None)` | blocks while the queue is full |
| `try_submit(request)` | rejects instead of blocking |
| `stats()` | a `ContinuousStats` snapshot |
| `shutdown(timeout=30.0)` | stops accepting, drains what is queued, joins |

Shutdown drains rather than abandoning: work already accepted is finished, the
same guarantee `DynamicBatcher` makes.

### `ContinuousStats`

`decode_steps`, `occupied_rows`, `available_rows`, `admissions`, `completions`,
`expired`, `max_observed_batch`, and `utilisation` — occupied over available,
the fraction of the batch that held a live sequence. That is the number static
batching loses and this recovers.

### `StepwiseRunner`, `SequenceState`, `EchoStepwiseRunner`, `TransformersStepwiseRunner`

The protocol continuous batching needs: `prefill`, `decode_step` and `evict`.
`decode_step` advances **every** active sequence by exactly one token, because a
decode step is one forward pass whose cost barely moves with batch width —
advancing sequences individually would forfeit the amortisation batching exists
for.

`SequenceState` carries the tokens generated so far and distinguishes
`stopped_early` (the model emitted end-of-sequence) from exhausting the token
budget. `EchoStepwiseRunner` is the deterministic implementation, with optional
per-step cost and forced early stops so scheduling can be tested without a model.

`TransformersStepwiseRunner(config, model=None, tokenizer=None)` implements the
protocol against a real causal model, owning the KV cache rather than delegating
to `generate`. `prefill` appends rows to that cache, `evict` removes them, and
ragged lengths are left-padded so every row's newest token sits where a decode
step reads. Position ids are derived from the mask, not from the index —
otherwise a padded row is told its padding is real history. `prefills` and
`steps` count the calls; `active_rows` and `cache_length` expose the cache shape.

Sampling is vectorised across rows, so each sequence keeps its own
`temperature`, `top_k` and `top_p` — a batch mixes requests, and one row being
greedy must not pin the others.

See [continuous-batching.md](continuous-batching.md).

### `SpeculativeDecoder` and `SpeculativeStats`

`SpeculativeDecoder(target, draft, lookahead=4)` runs a cheap draft model ahead
of an expensive target one. The draft proposes `lookahead` tokens; the target
checks all of them in a single forward pass, because decoding at batch size 1 is
bandwidth-bound and a pass over several candidate tokens costs about what a pass
over one costs.

`generate(input_ids, generation, seed=None)` returns the tokens and a
`SpeculativeStats`. Batch size 1 only — batched speculation makes the KV cache
ragged, since rows accept different numbers of tokens.

The output is **not an approximation**. Greedy keeps a proposal only when it
equals the target's own argmax. Sampling keeps it with probability
`min(1, p(x)/q(x))` and otherwise draws from the normalised residual
`max(0, p - q)`, which composes back to exactly `p`. The draft's quality
therefore affects speed alone: a bad draft is rejected more often and saves
less, but cannot change the distribution. Both properties are asserted in
`tests/python/test_speculative.py`, the second by comparing an empirical
distribution against the target's own.

Progress is guaranteed: a rejected position emits the target's own token, so
every target call yields at least one token. When every proposal is accepted the
trailing logits yield a free bonus token, for `lookahead + 1` tokens from one
call.

`SpeculativeStats` reports `tokens_per_target_call` — the speed signal, exactly
1.0 without speculation — and `acceptance_rate`, which is accepted over
proposals *made*. The latter is not the per-token agreement probability: a block
stops at its first rejection while the draft has already produced the whole
lookahead, so it falls as `lookahead` rises even for an equally good draft. Use
it for "how much draft work was wasted", not for comparing lookaheads.

`expected_tokens_per_call(acceptance, lookahead)` gives the closed form
`(1 - a^(k+1)) / (1 - a)`, so a lookahead can be chosen without running
anything. Returns saturate toward `1 / (1 - a)`: at `a = 0.5` a lookahead of 4
already captures 97% of what an infinite one would.

### `Request` and `Batch`

`Request` carries the prompt, its `GenerationConfig`, timestamps, and an
optional `deadline`. `Batch` carries the requests, the `BatchTrigger` that
closed it, and the formation time.

`BatchTrigger` is `MAX_SIZE`, `TIMEOUT` or `SHUTDOWN`. Worth recording: a
batcher that only ever closes on `TIMEOUT` is starved, one that only ever closes
on `MAX_SIZE` is saturated, and batch size alone cannot distinguish them.

## Configuration

### `EngineConfig`

| Field | Default | Effect |
| --- | --- | --- |
| `model_name` | `sshleifer/tiny-gpt2` | |
| `device`, `dtype` | `auto` | CUDA → MPS → CPU; bf16 where supported |
| `max_batch_size` | 16 | throughput lever |
| `max_wait_us` | 5000 | tail-latency lever |
| `queue_capacity` | 1024 | must be at least `max_batch_size` |
| `worker_threads` | 4 | executor width |
| `cuda_streams` | 4 | copy/compute overlap |
| `max_prompt_chars` | 8192 | 0 disables |
| `warmup_iterations` | 3 | keeps first-call costs out of measurements |

Validated in `__post_init__`, so an unworkable configuration fails at
construction rather than twenty minutes into a benchmark.

### `GenerationConfig`

`max_new_tokens`, `temperature`, `top_p`, `top_k`, `seed`. `temperature == 0`
means greedy. Every field is range-checked.

## Runners

`ModelRunner` is a runtime-checkable protocol with `warmup`, `generate` and
`description`. Two implementations ship:

* `EchoRunner(per_token_seconds=0.0, fixed_overhead=0.0)` — deterministic, no
  model. The cost parameters simulate real inference's fixed-plus-variable
  shape, which is what makes batching observable in a test.
* `TransformersRunner(config)` — a causal LM from transformers, with left
  padding (a causal model continues from the last position, so right padding
  would ask it to continue from padding).

Each result is a `GenerationResult` carrying `text`, `prompt_tokens` and
`generated_tokens`.

Implementations must return results **in the same order** they were given, and
one per prompt. The engine pairs them positionally and fails the whole batch
with a clear error if the count disagrees.

## Metrics

`LatencyHistogram` holds the samples — a recency window with exact percentiles,
sized by measurement rather than generosity, since two are recorded per request.
`MetricsRegistry` collects; `MetricsSnapshot` is a point-in-time view;
`render_prometheus(snapshot)` formats it for a scraper. Field names match the
C++ registry, so a dashboard need not know which runtime produced a snapshot —
a parity test enforces that.
