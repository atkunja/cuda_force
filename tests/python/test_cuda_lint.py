"""Tests for the CUDA structural checker.

These matter more than they might appear. The checker is the only automated
scrutiny the `.cu` sources get on a machine without a CUDA toolkit, and a linter
that silently stops detecting anything looks exactly like a clean codebase.

Each rule is tested in both directions: it fires on the bad shape, and it stays
quiet on the good one.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

# The checker lives in scripts/, which is not a package; the sys.path insert
# above is what makes this import work.
import check_cuda_sources as checker

REPO_ROOT = Path(__file__).resolve().parents[2]


def findings_for(tmp_path: Path, source: str, name: str = "kernel.cu") -> list[str]:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return [finding.rule for finding in checker.check_file(path)]


# --- unchecked launches ----------------------------------------------------


def test_a_launch_without_a_check_is_flagged(tmp_path):
    assert "unchecked-launch" in findings_for(
        tmp_path,
        """
        void launch(cudaStream_t stream) {
            my_kernel<<<1, 256, 0, stream>>>(nullptr);
        }
        """,
    )


def test_a_checked_launch_is_accepted(tmp_path):
    assert "unchecked-launch" not in findings_for(
        tmp_path,
        """
        void launch(cudaStream_t stream) {
            my_kernel<<<1, 256, 0, stream>>>(nullptr);
            CUDAFORGE_CHECK_LAUNCH(stream);
        }
        """,
    )


def test_a_check_after_a_switch_is_accepted(tmp_path):
    # The real launchers put the check after a switch, far more than a few lines
    # from the launch. A fixed-size window would produce a false positive here,
    # which is what motivated scoping the rule to the enclosing function.
    assert "unchecked-launch" not in findings_for(
        tmp_path,
        """
        void launch(cudaStream_t stream, int variant) {
            switch (variant) {
                case 0:
                    kernel_a<<<1, 256, 0, stream>>>();
                    break;
                case 1: {
                    const int grid = compute_grid();
                    const std::size_t shared = compute_shared();
                    kernel_b<<<grid, 256, shared, stream>>>();
                    break;
                }
                default:
                    break;
            }

            CUDAFORGE_CHECK_LAUNCH(stream);
        }
        """,
    )


# --- device-wide synchronisation -------------------------------------------


def test_device_synchronize_is_flagged(tmp_path):
    assert "device-sync" in findings_for(
        tmp_path,
        """
        void wait() {
            CUDAFORGE_CHECK(cudaDeviceSynchronize());
        }
        """,
    )


def test_stream_synchronize_is_accepted(tmp_path):
    assert "device-sync" not in findings_for(
        tmp_path,
        """
        void wait(cudaStream_t stream) {
            CUDAFORGE_CHECK(cudaStreamSynchronize(stream));
        }
        """,
    )


# --- discarded statuses ----------------------------------------------------


def test_a_discarded_status_is_flagged(tmp_path):
    assert "unchecked-status" in findings_for(
        tmp_path,
        """
        void allocate(void** p) {
            cudaMalloc(p, 1024);
        }
        """,
    )


def test_a_status_split_across_lines_is_accepted(tmp_path):
    # The formatter routinely wraps these. Matching per physical line would
    # report the second line as unchecked.
    assert "unchecked-status" not in findings_for(
        tmp_path,
        """
        void copy(void* dst, const void* src, std::size_t bytes, cudaStream_t stream) {
            CUDAFORGE_CHECK(
                cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream));
        }
        """,
    )


@pytest.mark.parametrize(
    "body",
    [
        "const cudaError_t status = cudaMalloc(&p, 1024);",
        "if (cudaMalloc(&p, 1024) != cudaSuccess) { return nullptr; }",
        "static_cast<void>(cudaFree(p));",
    ],
)
def test_every_accepted_status_form_is_recognised(tmp_path, body):
    assert "unchecked-status" not in findings_for(tmp_path, f"void f() {{ void* p; {body} }}")


# --- warp primitives -------------------------------------------------------


def test_a_maskless_shuffle_is_flagged(tmp_path):
    # Undefined on Volta and later, where lanes can be at different instructions.
    assert "maskless-shuffle" in findings_for(
        tmp_path, "__device__ float f(float v) { return __shfl_down(v, 16); }"
    )


def test_a_masked_shuffle_is_accepted(tmp_path):
    assert "maskless-shuffle" not in findings_for(
        tmp_path,
        "__device__ float f(float v) { return __shfl_down_sync(0xffffffffU, v, 16); }",
    )


# --- barriers --------------------------------------------------------------


def test_a_conditional_barrier_is_flagged(tmp_path):
    # Deadlocks when threads in the block diverge.
    assert "divergent-barrier" in findings_for(
        tmp_path,
        "__global__ void k() { if (threadIdx.x < 16) __syncthreads(); }",
    )


def test_an_unconditional_barrier_is_accepted(tmp_path):
    assert "divergent-barrier" not in findings_for(
        tmp_path,
        """
        __global__ void k() {
            if (threadIdx.x < 16) { tile[threadIdx.x] = 0.0F; }
            __syncthreads();
        }
        """,
    )


# --- comment handling ------------------------------------------------------


def test_violations_inside_comments_are_ignored(tmp_path):
    # The real sources discuss cudaDeviceSynchronize at length in comments
    # explaining why it is not used. Flagging prose would make the rule useless.
    assert (
        findings_for(
            tmp_path,
            """
        // Never call cudaDeviceSynchronize() here.
        /* cudaMalloc(&p, 1024); would be unchecked. */
        void f() {}
        """,
        )
        == []
    )


def test_line_numbers_survive_block_comments(tmp_path):
    path = tmp_path / "kernel.cu"
    path.write_text(
        "/* line one\n   line two\n   line three */\nvoid f() { cudaMalloc(&p, 1); }\n",
        encoding="utf-8",
    )
    findings = checker.check_file(path)
    assert len(findings) == 1
    assert findings[0].line == 4


# --- the real sources ------------------------------------------------------


def test_the_repository_sources_pass(tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/check_cuda_sources.py", "cuda", "tests/cuda"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout


def test_the_exit_code_reflects_findings(tmp_path):
    bad = tmp_path / "bad.cu"
    bad.write_text("void f(cudaStream_t s) { k<<<1, 1, 0, s>>>(); }\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "check_cuda_sources.py"), str(bad)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "unchecked-launch" in result.stdout


# --- block reduction barriers ----------------------------------------------


def test_a_reduction_returning_from_shared_memory_is_flagged(tmp_path):
    # Returning straight from shared memory lets a caller that reuses the array
    # overwrite it from one warp while another has not yet read it. The symptom
    # is a silently wrong row, not a crash.
    assert "unbarriered-reduction-return" in findings_for(
        tmp_path,
        """
        template <typename T>
        __device__ __forceinline__ T block_reduce_sum(T value, T* shared) {
            __syncthreads();
            return shared[0];
        }
        """,
        name="reduce.cuh",
    )


def test_the_barriered_form_is_accepted(tmp_path):
    assert "unbarriered-reduction-return" not in findings_for(
        tmp_path,
        """
        template <typename T>
        __device__ __forceinline__ T block_reduce_max(T value, T* shared, T identity) {
            __syncthreads();
            const T result = shared[0];
            __syncthreads();
            return result;
        }
        """,
        name="reduce.cuh",
    )


def test_calling_a_reduction_is_not_flagged(tmp_path):
    # The rule guards the primitive's definition. Flagging call sites would fire
    # on the correct softmax, which legitimately reduces twice over one array.
    assert "unbarriered-reduction-return" not in findings_for(
        tmp_path,
        """
        __global__ void softmax(const float* in, float* out, int cols) {
            __shared__ float scratch[32];
            const float row_max = block_reduce_max(local_max, scratch, -FLT_MAX);
            const float row_sum = block_reduce_sum(local_sum, scratch);
            out[0] = row_max + row_sum;
        }
        """,
    )


# --- index arithmetic ------------------------------------------------------


def test_narrow_index_arithmetic_is_flagged(tmp_path):
    # The product is computed in 32-bit; near INT_MAX elements it overflows and
    # the kernel reads the wrong element rather than faulting.
    assert "narrow-index-arithmetic" in findings_for(
        tmp_path,
        "__global__ void k(int n) { const int i = blockIdx.x * blockDim.x + threadIdx.x; }",
    )


def test_widened_index_arithmetic_is_accepted(tmp_path):
    assert "narrow-index-arithmetic" not in findings_for(
        tmp_path,
        """
        __global__ void k(int n) {
            const std::size_t i =
                blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
        }
        """,
    )


def test_the_rule_is_not_fooled_by_a_grid_stride(tmp_path):
    # A grid-stride loop multiplies blockDim by gridDim, not by blockIdx; that
    # is a different expression and must not be flagged.
    assert "narrow-index-arithmetic" not in findings_for(
        tmp_path,
        """
        __global__ void k(std::size_t n) {
            const auto stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
        }
        """,
    )
