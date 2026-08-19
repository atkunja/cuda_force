# Fine-Tuning

## Scope

The training pipeline uses PyTorch, Transformers and PEFT. Rewriting those would
add volume without adding signal — the custom systems engineering in this
project lives in the CUDA kernels and the concurrency runtime, not in a
reimplementation of a transformer.

What *is* written out rather than delegated:

* the training loop, so gradient accumulation, loss scaling and scheduler timing
  are explicit rather than hidden inside `Trainer`;
* `LoRALinear`, a from-scratch adapter layer used as a reference and as the
  oracle for the CUDA kernel — not as a replacement for PEFT;
* dataset packing, evaluation and checkpointing.

## Why LoRA

Full fine-tuning of a 7B model in fp16 needs roughly:

| Component | Size |
| --- | --- |
| Weights | 14 GB |
| Gradients | 14 GB |
| Adam state (two fp32 moments) | 56 GB |
| Activations | workload-dependent |

The optimiser state, not the weights, is what does not fit. LoRA freezes the
base weights and learns `W + (alpha/r)·B·A`, so gradients and optimiser state
exist only for the adapters:

| Component | Size with r=16 adapters |
| --- | --- |
| Weights (frozen) | 14 GB |
| Gradients | ~0.1 GB |
| Adam state | ~0.4 GB |

Those adapter figures follow from the parameter count, which is roughly
`2 · r · (d_in + d_out)` per adapted projection. The observed share on the tiny
model used in the examples is printed at startup — 64 of 102,778 parameters,
0.062% — and `describe_parameters()` reports it for any model.

## Initialisation is not arbitrary

```python
nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
```

`B` starts at exactly zero, so `B·A` is zero and the adapted model is
**numerically identical** to the base model at step 0. Training therefore starts
from the pretrained model rather than from a perturbed one, which is what makes
LoRA stable at learning rates an order of magnitude above full fine-tuning.

Initialising both to zero would leave the product's gradient zero as well, and
the adapter would never train. `tests/python/test_lora.py` asserts both halves.

## alpha is not a second learning rate

The effective adapter scale is `alpha / r`. Raising `r` without raising `alpha`
quietly *shrinks* the adapter's influence — a real and easy-to-miss consequence,
asserted directly in
`test_raising_rank_without_alpha_shrinks_the_adapter_scale`.

The `alpha = 2r` convention in the shipped configs holds the scale at 2.0 as
rank changes, so rank can be tuned without re-tuning the learning rate.

## Merging is exact

```python
merged_weight = base.weight + scaling * (lora_b @ lora_a)
```

Not an approximation. Once merged, the adapter costs nothing at inference —
which is the property that distinguishes LoRA from adapter methods that add
layers. `test_merging_is_exact` checks that the merged `nn.Linear` and the
adapted module produce identical outputs.

## Data packing

Examples are concatenated with EOS separators and cut into fixed-length blocks:

```
[doc1 tokens] EOS [doc2 tokens] EOS [doc3 tokens] EOS ...
└──── block 0 ────┘└──── block 1 ────┘└─ remainder, dropped ─┘
```

No padding is used, because padding is compute the GPU performs and then
discards. The trailing remainder is dropped rather than padded: a single short
block contributes little and complicates every downstream shape assumption.

The EOS between documents is what teaches the model where a document ends.
Without it, packing silently trains the model to continue from one document
straight into an unrelated one.

`labels` is a *clone* of `input_ids`, not an alias. The trainer may mask label
positions in place, and shared storage would corrupt the inputs — a bug that
would show up as mysteriously poor convergence rather than as an error.
`test_labels_do_not_alias_the_inputs` pins this down.

The shift between input and target is applied inside the model's loss, not in
the dataset. Doing it in both places would shift twice and train the model to
skip a token.

## The training loop

### Gradient accumulation

`k` micro-batches are run and their gradients summed before one optimiser step,
giving the convergence behaviour of a `k`-times larger batch at the memory cost
of the small one. The loss is divided by `k` so the effective gradient magnitude
is independent of how the batch was split.

`test_gradient_accumulation_matches_a_single_large_step` verifies this
numerically: four accumulated quarter-batches produce the same gradient as one
whole batch, to within float tolerance. Without that division, the effective
learning rate would silently depend on the accumulation setting.

The scheduler steps per **optimiser** step, not per micro-batch. Stepping per
micro-batch would advance the schedule `k` times too fast.

### Mixed precision

```python
use_fp16 = mixed_precision and cuda and not torch.cuda.is_bf16_supported()
scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
```

Loss scaling is a **float16 concern only**. Gradients in fp16 underflow to zero
below about 6e-8; scaling the loss up before the backward pass moves them into
range, and the scaler unscales before the optimiser step. bfloat16 has float32's
exponent range and needs none of this, which is why bf16 is preferred wherever
the hardware supports it.

Order matters around clipping:

```python
scaler.unscale_(optimizer)                        # 1. remove the scale factor
torch.nn.utils.clip_grad_norm_(trainable, 1.0)    # 2. then clip
scaler.step(optimizer)
```

Clipping a still-scaled gradient would clip at the wrong threshold, off by
exactly the scale factor — which drifts during training, so the effective
clipping threshold would drift too.

### Only adapters reach the optimiser

```python
trainable = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(trainable, ...)
```

Passing the frozen parameters would allocate Adam state for them — two fp32
tensors per parameter — and give away most of LoRA's memory advantage. The loop
raises if the list is empty, because silently training a model with no adapters
attached produces a run that looks successful and updates nothing.
