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

### `backend_report() -> BackendReport`

Which implementation path is active, and why.

```python
report = cudaforge.backend_report()
report.using_custom_kernels   # bool — the only field that matters for timings
report.extension_loaded       # the compiled module imported
report.cuda_compiled          # it was built with CUDA support
report.cuda_device_available  # a GPU is visible
```

Call this before drawing any conclusion from a measurement. A silent fallback
to the reference path looks exactly like a very slow custom kernel.
