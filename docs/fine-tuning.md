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
