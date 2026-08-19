"""Operator tests.

Everything here runs on CPU. The value is that the reference implementations
*are* the fallback path, so testing them tests what actually executes on a
machine without a GPU — and they are also the oracle the CUDA kernels are
checked against in ``tests/cuda``.
"""

from __future__ import annotations

import pytest
import torch

from cudaforge import ops


@pytest.mark.parametrize("shape", [(1, 1), (1, 64), (3, 17), (8, 128), (2, 4096), (33, 1023)])
def test_rmsnorm_matches_the_explicit_formula(shape):
    x = torch.randn(*shape)
    weight = torch.randn(shape[-1])
    eps = 1e-6

    expected = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight
    torch.testing.assert_close(ops.rmsnorm(x, weight, eps), expected, rtol=1e-5, atol=1e-6)


def test_rmsnorm_leaves_unit_input_unchanged_with_unit_weight():
    # Every element is 1, so the RMS is 1 and the output is the weight.
    x = torch.ones(4, 32)
    weight = torch.ones(32)
    torch.testing.assert_close(ops.rmsnorm(x, weight, 0.0), torch.ones(4, 32))


def test_rmsnorm_scale_invariance():
    # RMSNorm divides by the RMS, so scaling the whole row leaves the result
    # unchanged in the limit of eps -> 0. This is the property that makes it a
    # normalisation rather than a scaling.
    x = torch.randn(4, 128)
    weight = torch.ones(128)
    base = ops.rmsnorm(x, weight, 1e-12)
    scaled = ops.rmsnorm(x * 10.0, weight, 1e-12)
    torch.testing.assert_close(base, scaled, rtol=1e-4, atol=1e-4)


def test_rmsnorm_rejects_a_mismatched_weight():
    with pytest.raises(ValueError, match="does not match"):
        ops.rmsnorm(torch.randn(2, 8), torch.randn(16))


def test_rmsnorm_rejects_a_non_vector_weight():
    with pytest.raises(ValueError, match="1-D"):
        ops.rmsnorm(torch.randn(2, 8), torch.randn(2, 8))


def test_rmsnorm_handles_non_contiguous_input():
    # Transposing produces a non-contiguous view; the kernels index rows
    # arithmetically, so the wrapper must materialise a contiguous copy.
    x = torch.randn(64, 4).t()
    assert not x.is_contiguous()
    weight = torch.randn(64)
    expected = ops.rmsnorm(x.contiguous(), weight)
    torch.testing.assert_close(ops.rmsnorm(x, weight), expected)


def test_rmsnorm_survives_float16_magnitudes_that_would_overflow():
    # 300^2 = 90000 exceeds float16's maximum of 65504. Squaring in float16
    # would produce inf and poison the row; the implementation promotes first.
    x = torch.full((2, 64), 300.0, dtype=torch.float16)
    weight = torch.ones(64, dtype=torch.float16)
    result = ops.rmsnorm(x, weight)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result.float(), torch.ones(2, 64), rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("shape", [(1, 1), (1, 32), (4, 17), (8, 256), (3, 1025)])
def test_softmax_matches_torch(shape):
    x = torch.randn(*shape)
    torch.testing.assert_close(ops.softmax(x), torch.softmax(x, dim=-1))


def test_softmax_rows_sum_to_one():
    result = ops.softmax(torch.randn(16, 128))
    torch.testing.assert_close(result.sum(dim=-1), torch.ones(16), rtol=1e-5, atol=1e-6)


def test_softmax_is_stable_for_large_logits():
    # exp(1000) overflows float32. Subtracting the row maximum is what keeps
    # this finite, and attention logits genuinely reach these magnitudes.
    x = torch.tensor([[1000.0, 1000.0, 1000.0], [-1000.0, 0.0, 1000.0]])
    result = ops.softmax(x)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result[0], torch.full((3,), 1 / 3), rtol=1e-5, atol=1e-6)
    assert result[1, 2] == pytest.approx(1.0)


def test_softmax_is_shift_invariant():
    x = torch.randn(4, 64)
    torch.testing.assert_close(ops.softmax(x), ops.softmax(x + 12.5), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("batch", "in_features", "out_features", "rank"),
    [(1, 8, 4, 2), (4, 64, 32, 8), (7, 129, 65, 3), (16, 256, 256, 16)],
)
def test_lora_linear_matches_the_definition(batch, in_features, out_features, rank):
    x = torch.randn(batch, in_features)
    weight = torch.randn(in_features, out_features)
    lora_a = torch.randn(in_features, rank)
    lora_b = torch.randn(rank, out_features)
    scale = 0.25

    expected = x @ weight + scale * ((x @ lora_a) @ lora_b)
    torch.testing.assert_close(
        ops.lora_linear(x, weight, lora_a, lora_b, scale), expected, rtol=1e-4, atol=1e-4
    )


def test_lora_with_zero_b_reduces_to_the_frozen_layer():
    # This is how LoRA is initialised: B starts at zero so the adapted model is
    # numerically identical to the base model at step 0.
    x = torch.randn(4, 32)
    weight = torch.randn(32, 16)
    lora_a = torch.randn(32, 4)
    lora_b = torch.zeros(4, 16)
    torch.testing.assert_close(
        ops.lora_linear(x, weight, lora_a, lora_b, 1.0), x @ weight, rtol=1e-5, atol=1e-5
    )


def test_lora_scale_is_linear_in_the_adapter_path():
    x = torch.randn(4, 32)
    weight = torch.zeros(32, 16)  # isolate the adapter contribution
    lora_a = torch.randn(32, 4)
    lora_b = torch.randn(4, 16)

    at_one = ops.lora_linear(x, weight, lora_a, lora_b, 1.0)
    at_two = ops.lora_linear(x, weight, lora_a, lora_b, 2.0)
    torch.testing.assert_close(at_two, 2 * at_one, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize(
    ("shapes", "message"),
    [
        # x's feature count disagrees with the weight's input dimension.
        ({"x": (4, 31), "w": (32, 16), "a": (32, 4), "b": (4, 16)}, "input features"),
        # lora_a's input dimension disagrees with the weight's.
        ({"x": (4, 32), "w": (32, 16), "a": (31, 4), "b": (4, 16)}, "lora_a"),
        # lora_b's rank disagrees with lora_a's.
        ({"x": (4, 32), "w": (32, 16), "a": (32, 4), "b": (5, 16)}, "lora_b"),
        # lora_b's output dimension disagrees with the weight's.
        ({"x": (4, 32), "w": (32, 16), "a": (32, 4), "b": (4, 15)}, "lora_b"),
    ],
)
def test_lora_rejects_mismatched_shapes(shapes, message):
    with pytest.raises(ValueError, match=message):
        ops.lora_linear(
            torch.randn(*shapes["x"]),
            torch.randn(*shapes["w"]),
            torch.randn(*shapes["a"]),
            torch.randn(*shapes["b"]),
        )


@pytest.mark.parametrize("size", [1, 63, 64, 65, 1000, 4096])
def test_sum_reduce_matches_torch(size):
    x = torch.randn(size)
    torch.testing.assert_close(ops.sum_reduce(x), x.sum(), rtol=1e-4, atol=1e-4)


def test_sum_of_empty_is_zero():
    assert ops.sum_reduce(torch.empty(0)).item() == 0.0


@pytest.mark.parametrize("size", [1, 63, 64, 65, 128, 1000])
def test_quantization_round_trip_stays_within_the_theoretical_bound(size):
    # Symmetric absmax rounding cannot err by more than half a step, and the
    # step is the block's scale. Exceeding this means a bug, not merely loss.
    x = torch.randn(size) * 5.0
    quantised, scales = ops.quantize_int8(x)
    restored = ops.dequantize_int8(quantised, scales)
    assert (x - restored).abs().max().item() <= scales.max().item() / 2 + 1e-6


def test_quantized_values_stay_in_the_symmetric_int8_range():
    quantised, _ = ops.quantize_int8(torch.randn(1000) * 100)
    assert quantised.min().item() >= -127
    assert quantised.max().item() <= 127


def test_quantization_preserves_shape():
    x = torch.randn(7, 33)
    quantised, scales = ops.quantize_int8(x)
    assert quantised.shape == x.shape
    assert scales.numel() == (x.numel() + ops.QUANT_BLOCK_SIZE - 1) // ops.QUANT_BLOCK_SIZE


def test_all_zero_input_round_trips_exactly():
    # The scale would be zero; the implementation substitutes 1 so the block
    # maps to zero and back without dividing by zero.
    x = torch.zeros(128)
    quantised, scales = ops.quantize_int8(x)
    assert (scales > 0).all()
    torch.testing.assert_close(ops.dequantize_int8(quantised, scales), x)


def test_an_outlier_only_degrades_its_own_block():
    # Block-wise scaling exists so one large value does not stretch the range of
    # unrelated elements. The outlier itself is always exact — it defines the
    # absmax and maps to 127 — so the comparison has to be between an ordinary
    # element sharing the outlier's block and the same value in a clean block.
    x = torch.full((256,), 0.5)
    x[0] = 1000.0
    quantised, scales = ops.quantize_int8(x)
    restored = ops.dequantize_int8(quantised, scales)

    same_block_error = (x[1] - restored[1]).abs()  # block 0, with the outlier
    clean_block_error = (x[200] - restored[200]).abs()  # block 3, no outlier

    assert same_block_error > clean_block_error
    assert clean_block_error < 0.01
    # The outlier defines its block's scale, so it survives the round trip.
    torch.testing.assert_close(restored[0], x[0])


def test_dequantize_rejects_a_wrong_scale_count():
    quantised, _ = ops.quantize_int8(torch.randn(128))
    with pytest.raises(ValueError, match="block scales"):
        ops.dequantize_int8(quantised, torch.ones(1))


def test_quantization_error_helper_agrees_with_a_manual_round_trip():
    x = torch.randn(512)
    quantised, scales = ops.quantize_int8(x)
    manual = (x - ops.dequantize_int8(quantised, scales)).abs().max()
    torch.testing.assert_close(ops.quantization_error(x), manual)


def test_backend_report_is_self_consistent():
    report = ops.backend_report()
    assert isinstance(report.message, str)
    assert report.using_custom_kernels == (
        report.extension_loaded and report.cuda_compiled and report.cuda_device_available
    )
    if not torch.cuda.is_available():
        assert not report.using_custom_kernels
    assert "cudaforge" in str(report)


def test_cuda_availability_flag_agrees_with_torch():
    if not torch.cuda.is_available():
        assert not ops.CUDA_KERNELS_AVAILABLE


# --- activations -----------------------------------------------------------


@pytest.mark.parametrize("shape", [(1,), (64,), (4, 17), (8, 4096), (3, 1023)])
def test_silu_matches_torch(shape):
    x = torch.randn(*shape)
    torch.testing.assert_close(ops.silu(x), torch.nn.functional.silu(x))


def test_silu_matches_its_definition():
    x = torch.randn(256)
    torch.testing.assert_close(ops.silu(x), x * torch.sigmoid(x), rtol=1e-5, atol=1e-6)


def test_silu_is_zero_at_zero():
    assert ops.silu(torch.zeros(8)).abs().max() == 0.0


def test_silu_does_not_vanish_for_negative_inputs():
    # The property that displaced ReLU: the gradient is non-zero on the negative
    # side, so units do not become permanently inactive.
    x = torch.tensor([-4.0, -2.0, -1.0], requires_grad=True)
    ops.silu(x).sum().backward()
    assert (x.grad.abs() > 0).all()


@pytest.mark.parametrize("shape", [(1,), (64,), (4, 17), (8, 4096)])
def test_gelu_matches_the_tanh_approximation(shape):
    # The tanh form, not the exact erf form: GPT-2 and BERT were trained against
    # this curve, and substituting the exact version is not a free change.
    x = torch.randn(*shape)
    torch.testing.assert_close(
        ops.gelu(x), torch.nn.functional.gelu(x, approximate="tanh"), rtol=1e-5, atol=1e-6
    )


def test_gelu_differs_from_the_exact_form_by_a_small_but_real_amount():
    x = torch.linspace(-3, 3, 512)
    difference = (ops.gelu(x) - torch.nn.functional.gelu(x, approximate="none")).abs().max()
    assert 0 < difference < 1e-2


@pytest.mark.parametrize("shape", [(1,), (64,), (4, 17), (8, 4096), (2, 3, 128)])
def test_swiglu_matches_its_definition(shape):
    gate = torch.randn(*shape)
    up = torch.randn(*shape)
    torch.testing.assert_close(
        ops.swiglu(gate, up), torch.nn.functional.silu(gate) * up, rtol=1e-5, atol=1e-6
    )


def test_swiglu_with_a_unit_up_projection_is_silu():
    gate = torch.randn(128)
    torch.testing.assert_close(ops.swiglu(gate, torch.ones(128)), ops.silu(gate))


def test_swiglu_is_linear_in_the_up_projection():
    gate = torch.randn(128)
    up = torch.randn(128)
    torch.testing.assert_close(
        ops.swiglu(gate, 2 * up), 2 * ops.swiglu(gate, up), rtol=1e-5, atol=1e-6
    )


def test_swiglu_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        ops.swiglu(torch.randn(4, 8), torch.randn(4, 16))


def test_swiglu_rejects_mismatched_dtypes():
    with pytest.raises(ValueError, match="same dtype"):
        ops.swiglu(torch.randn(8), torch.randn(8, dtype=torch.float64))


def test_swiglu_promotes_half_inputs_for_the_sigmoid():
    # exp() of a moderately negative input underflows a 10-bit mantissa long
    # before it underflows float32, which would flatten the negative tail.
    gate = torch.linspace(-20, 20, 256, dtype=torch.float16)
    up = torch.ones(256, dtype=torch.float16)

    result = ops.swiglu(gate, up)
    expected = (torch.nn.functional.silu(gate.float()) * up.float()).half()

    assert result.dtype == torch.float16
    assert torch.isfinite(result).all()
    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-3)


def test_activations_handle_non_contiguous_input():
    x = torch.randn(64, 4).t()
    assert not x.is_contiguous()
    torch.testing.assert_close(ops.silu(x), torch.nn.functional.silu(x.contiguous()))


def test_activations_of_empty_tensors_are_empty():
    empty = torch.empty(0)
    assert ops.silu(empty).numel() == 0
    assert ops.gelu(empty).numel() == 0
    assert ops.swiglu(empty, empty).numel() == 0


# --- fused residual + rmsnorm ---------------------------------------------


@pytest.mark.parametrize("shape", [(1, 1), (1, 64), (4, 17), (8, 4096), (33, 1023)])
def test_fused_residual_rmsnorm_matches_the_unfused_sequence(shape):
    x = torch.randn(*shape)
    residual = torch.randn(*shape)
    weight = torch.randn(shape[-1])

    normalised, summed = ops.fused_residual_rmsnorm(x, residual, weight)

    expected_sum = x + residual
    expected_norm = ops.rmsnorm(expected_sum, weight)

    torch.testing.assert_close(summed, expected_sum)
    torch.testing.assert_close(normalised, expected_norm, rtol=1e-5, atol=1e-6)


def test_the_residual_output_is_the_plain_sum():
    # It is what the *following* residual connection adds to, so it must be the
    # un-normalised sum — returning the normalised value would silently change
    # the model.
    x = torch.randn(4, 32)
    residual = torch.randn(4, 32)
    _, summed = ops.fused_residual_rmsnorm(x, residual, torch.ones(32))
    torch.testing.assert_close(summed, x + residual)


def test_a_zero_residual_reduces_to_plain_rmsnorm():
    x = torch.randn(4, 64)
    weight = torch.randn(64)
    normalised, _ = ops.fused_residual_rmsnorm(x, torch.zeros(4, 64), weight)
    torch.testing.assert_close(normalised, ops.rmsnorm(x, weight), rtol=1e-5, atol=1e-6)


def test_the_operands_are_symmetric():
    # Addition commutes, so swapping them must not change the result.
    x = torch.randn(4, 32)
    residual = torch.randn(4, 32)
    weight = torch.randn(32)

    first, _ = ops.fused_residual_rmsnorm(x, residual, weight)
    second, _ = ops.fused_residual_rmsnorm(residual, x, weight)
    torch.testing.assert_close(first, second, rtol=1e-5, atol=1e-6)


def test_fused_residual_rmsnorm_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        ops.fused_residual_rmsnorm(torch.randn(4, 8), torch.randn(4, 16), torch.ones(8))


def test_fused_residual_rmsnorm_rejects_a_bad_weight():
    with pytest.raises(ValueError, match="weight"):
        ops.fused_residual_rmsnorm(torch.randn(4, 8), torch.randn(4, 8), torch.ones(16))


def test_two_fused_blocks_compose_like_a_transformer_stack():
    # The pattern the kernel exists for: each block's residual output feeds the
    # next block's residual input.
    torch.manual_seed(0)
    hidden = torch.randn(4, 64)
    residual = torch.zeros(4, 64)
    weight = torch.ones(64)

    fused_state = (hidden, residual)
    unfused_residual = residual.clone()
    unfused_hidden = hidden.clone()

    for _ in range(3):
        normalised, carried = ops.fused_residual_rmsnorm(*fused_state[:2], weight)
        fused_state = (normalised, carried)

        summed = unfused_hidden + unfused_residual
        unfused_hidden = ops.rmsnorm(summed, weight)
        unfused_residual = summed

    torch.testing.assert_close(fused_state[0], unfused_hidden, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(fused_state[1], unfused_residual, rtol=1e-4, atol=1e-5)


# --- kernel dtype coverage -------------------------------------------------


def test_the_dtype_the_config_prefers_has_a_kernel():
    # EngineConfig.resolve_dtype prefers bfloat16 on Ampere and later. A
    # configured dtype with no kernel behind it would silently take the fallback
    # and look like a slow kernel.
    from cudaforge.config import EngineConfig

    for dtype in (torch.float32, torch.float16, torch.bfloat16):
        assert ops.kernel_supports(dtype), dtype

    resolved = EngineConfig(device="cuda", dtype="auto").resolve_dtype()
    assert ops.kernel_supports(resolved)


def test_an_unsupported_dtype_is_reported_as_such():
    assert not ops.kernel_supports(torch.float64)
    assert not ops.kernel_supports(torch.int8)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_the_reference_path_handles_every_kernel_dtype(dtype):
    # The fallback must cover at least what the kernels cover, or a CPU-only
    # host cannot exercise the same configurations.
    x = torch.randn(4, 64, dtype=dtype)
    weight = torch.ones(64, dtype=dtype)

    result = ops.rmsnorm(x, weight)
    assert result.dtype == dtype
    assert torch.isfinite(result).all()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_swiglu_preserves_reduced_precision_dtypes(dtype):
    gate = torch.randn(128, dtype=dtype)
    up = torch.randn(128, dtype=dtype)

    result = ops.swiglu(gate, up)
    assert result.dtype == dtype
    assert torch.isfinite(result).all()


def test_bfloat16_holds_magnitudes_that_overflow_float16():
    # The property bfloat16 exists for: float32's exponent range. 300 squared is
    # 90,000, which is infinite in float16 and unremarkable in bfloat16.
    value = torch.tensor([300.0], dtype=torch.bfloat16)
    assert torch.isfinite(value.float() ** 2).all()

    as_half = torch.tensor([300.0], dtype=torch.float16)
    assert torch.isinf(as_half * as_half).all()

    # The reference path promotes for exactly this reason, so both survive.
    for dtype in (torch.float16, torch.bfloat16):
        x = torch.full((2, 64), 300.0, dtype=dtype)
        result = ops.rmsnorm(x, torch.ones(64, dtype=dtype))
        assert torch.isfinite(result).all(), dtype
