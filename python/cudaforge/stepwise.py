"""Step-wise generation, the interface continuous batching needs.

`ModelRunner.generate` is one-shot: hand it prompts, get finished text. That is
enough for static batching, where a batch runs until its longest member is done,
and it is the reason static batching wastes so much.

Consider a batch of four asking for 8, 8, 8 and 256 tokens. The one-shot
interface has no way to release the three short ones early, so they occupy their
rows for 248 steps after they finished. Utilisation is the shaded fraction:

    static      [====][====][====][==========================]
                                  ^ three rows idle from here

    continuous  [====][====][====][===]
                [new ][new ][new ][new ]   ← admitted into the freed rows

Iteration-level scheduling needs generation broken into steps, so the scheduler
can look between them. That is what this protocol adds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from cudaforge.config import GenerationConfig


@dataclass
class SequenceState:
    """One in-flight sequence, as the scheduler sees it."""

    sequence_id: int
    prompt: str
    generation: GenerationConfig
    tokens: list[str] = field(default_factory=list)
    #: Set when the model emits its end-of-sequence marker, as distinct from
    #: hitting the token budget. Both finish a sequence; only the first means
    #: the model considered itself done.
    stopped_early: bool = False

    @property
    def generated(self) -> int:
        return len(self.tokens)

    @property
    def finished(self) -> bool:
        return self.stopped_early or self.generated >= self.generation.max_new_tokens

    @property
    def text(self) -> str:
        return "".join(self.tokens)

    @property
    def remaining(self) -> int:
        """Tokens still owed. Zero once finished."""
        return max(0, self.generation.max_new_tokens - self.generated)


@runtime_checkable
class StepwiseRunner(Protocol):
    """A model that can be advanced one token at a time, for a whole batch.

    The contract that makes continuous batching possible:

    * `prefill` admits sequences and processes their prompts.
    * `decode_step` advances **every active sequence by exactly one token**, so
      the scheduler regains control after each one.
    * `evict` drops a sequence mid-generation, releasing whatever it held.

    One token for the whole batch, not per sequence: a decode step is a single
    forward pass over the batch, and the cost is dominated by reading the
    weights once. Advancing sequences individually would forfeit exactly the
    amortisation batching exists for.
    """

    def prefill(self, states: list[SequenceState]) -> None: ...

    def decode_step(self, states: list[SequenceState]) -> None: ...

    def evict(self, sequence_id: int) -> None: ...

    @property
    def description(self) -> str: ...


class EchoStepwiseRunner:
    """Deterministic step-wise runner with no model dependency.

    Exists for the same reason `EchoRunner` does: the scheduling logic is what
    is worth testing, and it is model-independent. Output is derived from the
    prompt, so assertions are stable across runs and machines.

    `stop_after` makes a sequence emit its end-of-sequence marker before its
    budget is exhausted, which is the case continuous batching is *for* — a
    short sequence sharing a batch with a long one.
    """

    def __init__(
        self,
        per_step_seconds: float = 0.0,
        prefill_seconds: float = 0.0,
        stop_after: dict[str, int] | None = None,
    ) -> None:
        self._per_step = per_step_seconds
        self._prefill = prefill_seconds
        self._stop_after = stop_after or {}
        self.active: set[int] = set()
        self.steps = 0
        self.prefills = 0
        #: Batch width at each decode step, which is what utilisation is
        #: computed from.
        self.step_widths: list[int] = []

    def prefill(self, states: list[SequenceState]) -> None:
        import time

        if not states:
            return
        self.prefills += 1
        if self._prefill:
            time.sleep(self._prefill)
        for state in states:
            self.active.add(state.sequence_id)

    def decode_step(self, states: list[SequenceState]) -> None:
        import time

        active = [state for state in states if not state.finished]
        if not active:
            return

        self.steps += 1
        self.step_widths.append(len(active))
        # One sleep for the whole batch, not one per sequence: a decode step is
        # a single forward pass, and its cost barely moves with batch width.
        if self._per_step:
            time.sleep(self._per_step)

        for state in active:
            limit = self._stop_after.get(state.prompt)
            if limit is not None and state.generated >= limit:
                state.stopped_early = True
                continue
            state.tokens.append(f"{state.prompt[:2]}{state.generated}")

    def evict(self, sequence_id: int) -> None:
        self.active.discard(sequence_id)

    @property
    def description(self) -> str:
        return f"EchoStepwiseRunner(per_step={self._per_step}s)"
