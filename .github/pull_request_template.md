## What this changes

<!-- One or two sentences. The why matters more than the what. -->

## Checks

- [ ] `./scripts/lint.sh` is clean
- [ ] `./scripts/test.sh` passes, with skips explained
- [ ] Touched concurrency? ran `./scripts/build.sh --sanitizer thread` and the suite
- [ ] Touched `cuda/`? `scripts/check_cuda_sources.py` is clean
- [ ] New kernel? ships with a reference implementation and parity tests

## Measurements

<!--
If this claims a performance improvement, paste the harness output — before and
after, on the same machine.

If you did not measure it, say so. That is an acceptable answer; an invented
number is not.
-->

Ran on: <!-- CPU/GPU, or "not measured" -->

## GPU verification

<!--
The maintainer's machine has no NVIDIA GPU, so anything under cuda/ cannot be
verified locally. If you have hardware, the output of

    ./build-cuda/tests/cuda/cudaforge_cuda_tests

is genuinely useful information to include.
-->

- [ ] Ran the CUDA test suite on a GPU (output below)
- [ ] No GPU available
