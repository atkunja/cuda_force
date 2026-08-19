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
says nothing about the kernel. `KERNEL_DTYPES` is the underlying set.

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

Implementations must return results **in the same order** they were given, and
one per prompt. The engine pairs them positionally and fails the whole batch
with a clear error if the count disagrees.

## Metrics

`MetricsRegistry` collects; `MetricsSnapshot` is a point-in-time view;
`render_prometheus(snapshot)` formats it for a scraper. Field names match the
C++ registry, so a dashboard need not know which runtime produced a snapshot —
a parity test enforces that.
