# Contributing

## Before you start

```bash
./scripts/setup.sh && source .venv/bin/activate
./scripts/test.sh
```

Then install the hooks, either way round — they cover the same ground:

```bash
pre-commit install                      # if you use pre-commit
git config core.hooksPath .githooks     # if you would rather not
```

Both run formatting, linting and the CUDA and documentation checks on staged
files only, so they finish in under a second. Slow suites belong in CI, because
a hook that takes 30 seconds gets bypassed with `--no-verify`.

`test.sh` prints `[SKIP]` with a reason for anything your machine cannot run —
most commonly the CUDA suites. That is expected and is not a failure.

## The rules that are not negotiable

**No fabricated measurements.** If you did not run it, do not write the number.
Say it was not measured. Every performance claim in this repository is traceable
to a harness and a machine; one invented figure would make the rest worthless.

**Every CUDA call is checked.** `CUDAFORGE_CHECK` around anything returning a
status, `CUDAFORGE_CHECK_LAUNCH` after every launch. An ignored status surfaces
thousands of lines later as an unrelated illegal access.

**No `cudaDeviceSynchronize`.** It is a barrier across every stream and collapses
the pipeline the scheduler exists to create. Synchronise a stream, or wait on an
event. `scripts/check_cuda_sources.py` will reject it.

**Portable code stays portable.** Nothing in `cpp/` may include a CUDA header.
That split is what lets the concurrency runtime be tested on any machine.

**New kernels ship with a reference.** A PyTorch or host implementation that
defines the semantics, plus tests over non-power-of-two, very small and large
shapes. The reference is also the CPU fallback, so it stays exercised.

## Comments

Comment the *why*, never the *what*. `// increment the counter` above
`++counter;` is noise. `// Relaxed: nothing is published through this counter,
so no reader establishes happens-before from it` is the kind of thing worth
writing down, because it cannot be recovered from reading the code.

If a line is subtle enough to need a comment, the comment should say what breaks
without it.

## Tests

Assert properties, not sequences — concurrency bugs do not reproduce on demand.
Look at how `tests/cpp/test_concurrent_queue.cpp` verifies that capacity is
never exceeded (an independent watcher thread) rather than checking a particular
interleaving.

**Never call Catch2 assertion macros from a worker thread.** They are not
thread-safe and produce a scheduler-dependent abort that looks unrelated to your
change. Use `tests/cpp/thread_assert.hpp` and assert on the main thread after
joining.

Seed every generator. An unseeded input makes a marginal tolerance failure
intermittent, which is far harder to diagnose than a consistent one.

## Commits

Conventional prefixes: `feat`, `fix`, `test`, `docs`, `bench`, `build`, `style`,
`chore`, `perf`. Scope in parentheses where it helps: `feat(cuda)`, `fix(python)`.

Keep them small and self-contained. A commit that changes one thing can be
reverted; one that changes six cannot.

## Dependencies must be declared, not merely installed

A dependency that happens to be in your virtualenv but is not in `pyproject.toml`
passes locally and fails in CI. This has happened here: `pyyaml`, `httpx` and
`httpx2` were all present on the development machine and missing from the
extras that needed them.

Before adding an import, check it is declared in the right extra — and if you
are unsure, verify against a clean environment rather than yours:

```bash
python3 -m venv /tmp/check
/tmp/check/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
/tmp/check/bin/pip install -e ".[dev,serve]"
/tmp/check/bin/pytest tests/python -q
```

Note the bare `pytest`, not `python -m pytest`. The latter puts the current
directory on `sys.path` and hides imports that only work from a checkout.

## Before opening a pull request

```bash
./scripts/lint.sh        # ruff, mypy, clang-format, CUDA structural checks
./scripts/test.sh        # everything runnable here
```

If you touched concurrency, also:

```bash
./scripts/build.sh --sanitizer thread && ./build-thread/tests/cpp/cudaforge_tests
```

If you have a GPU and touched anything under `cuda/`, say so in the PR along
with the output of `./build-cuda/tests/cuda/cudaforge_cuda_tests`. That is
information the maintainer's machine cannot produce.
