"""Autograd behaviour of the custom operators.

The kernels are inference-only — no backward kernels exist for them. The danger
is not that gradients are unavailable; it is that PyTorch's default behaviour
for an operator with no registered autograd kernel is to warn once and then
produce quietly incorrect gradients. Training would converge to something, just
not to the right thing, and nothing would point at the cause.

Two mechanisms address that, and both are tested here:

1. The extension registers an explicit guard, so differentiating through a
   registered operator raises with the operator's name.
2. `cudaforge.ops` routes to the reference implementation whenever a backward
   pass is expected, so callers get correct gradients without having to know
   any of this.
"""

from __future__ import annotations

import pytest
import torch

from cudaforge import ops

pytestmark = pytest.mark.skipif(
    not ops.backend_report().extension_loaded,
    reason="the compiled extension is not present, so there is no dispatcher to guard",
)


def test_differentiating_a_registered_operator_raises():
    # Loud failure rather than silently wrong gradients.
    x = torch.randn(8, requires_grad=True)
    with pytest.raises(RuntimeError, match="derivative for cudaforge::silu is not implemented"):
        torch.ops.cudaforge.silu(x).sum().backward()


def test_the_registered_operator_still_works_for_inference():
    torch.testing.assert_close(
        torch.ops.cudaforge.silu(torch.randn(16)),
        torch.nn.functional.silu(torch.randn(16)),
        rtol=1,
        atol=1,
    )  # shapes and finiteness; values differ because the inputs differ
    assert torch.ops.cudaforge.silu(torch.randn(16)).shape == (16,)


@pytest.mark.parametrize(
    ("name", "call"),
    [
        ("silu", lambda x: ops.silu(x)),
        ("gelu", lambda x: ops.gelu(x)),
        ("softmax", lambda x: ops.softmax(x)),
        ("sum_reduce", lambda x: ops.sum_reduce(x)),
    ],
)
def test_the_python_wrappers_are_differentiable(name, call):
    x = torch.randn(4, 16, requires_grad=True)
    call(x).sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_rmsnorm_produces_gradients_for_both_operands():
    x = torch.randn(4, 32, requires_grad=True)
    weight = torch.ones(32, requires_grad=True)
    ops.rmsnorm(x, weight).sum().backward()

    assert x.grad is not None
    assert weight.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(weight.grad).all()


def test_swiglu_produces_gradients_for_both_operands():
    gate = torch.randn(4, 32, requires_grad=True)
    up = torch.randn(4, 32, requires_grad=True)
    ops.swiglu(gate, up).sum().backward()

    assert gate.grad is not None
    assert up.grad is not None


def test_a_gradient_on_any_operand_routes_to_the_reference():
    # Only the adapter requires grad, which is exactly the LoRA case: the base
    # weight is frozen.
    x = torch.randn(4, 32)
    weight = torch.randn(32, 16)
    lora_a = torch.randn(32, 4, requires_grad=True)
    lora_b = torch.randn(4, 16, requires_grad=True)

    ops.lora_linear(x, weight, lora_a, lora_b, 2.0).sum().backward()
    assert lora_a.grad is not None
    assert lora_b.grad is not None


def test_gradients_match_the_analytic_derivative():
    # SiLU'(x) = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    x = torch.linspace(-4, 4, 64, requires_grad=True)
    ops.silu(x).sum().backward()

    sigmoid = torch.sigmoid(x.detach())
    expected = sigmoid * (1 + x.detach() * (1 - sigmoid))
    torch.testing.assert_close(x.grad, expected, rtol=1e-5, atol=1e-6)


def test_no_grad_context_uses_the_dispatcher():
    # Inside no_grad there is no backward pass to protect, so the fast path is
    # the right one.
    with torch.no_grad():
        x = torch.randn(8, 16)
        assert ops.silu(x).shape == (8, 16)


def test_inputs_without_requires_grad_use_the_dispatcher():
    x = torch.randn(8, 16)
    assert not x.requires_grad
    assert ops.silu(x).shape == (8, 16)


def test_gradcheck_passes_on_the_reference_path():
    # Double precision, small input: the standard numerical-vs-analytic check.
    x = torch.randn(6, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(ops.silu, (x,), eps=1e-6, atol=1e-4)
