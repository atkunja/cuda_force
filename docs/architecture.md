# Architecture

## Layering

```mermaid
flowchart TB
    subgraph py["python/cudaforge — user-facing"]
        OPS["ops.py<br/>dispatch: extension or reference"]
        ENG["engine.py<br/>submit → batch → execute → future"]
        SCH["scheduler.py<br/>DynamicBatcher"]
        MET["metrics.py"]
    end

    subgraph cpp["cpp/ — portable C++20, no CUDA"]
        CQ["ConcurrentQueue"]
        TP["ThreadPool"]
        DB["DynamicBatcher"]
        MP["MemoryPool&lt;Backend&gt;"]
        CM["Metrics"]
    end

    subgraph cu["cuda/ — requires NVIDIA"]
        K["Kernels<br/>reduction · softmax · rmsnorm · lora · quant"]
        GS["GpuScheduler<br/>streams · events"]
        DBK["DeviceAllocatorBackend"]
    end

    BIND["cpp/src/bindings.cpp<br/>TORCH_LIBRARY"]

    OPS --> BIND
    ENG --> SCH
    ENG --> MET
    BIND -.CUDA build only.-> K
    GS --> K
    MP -.template parameter.-> DBK
    DB --> CQ
    DB --> CM
    TP --> CQ
```

## The load-bearing split

Everything in `cpp/` builds and is tested on any host with a C++20 compiler.
Everything under `cuda/` is guarded behind `CUDAFORGE_ENABLE_CUDA` and is never
a build prerequisite for the portable targets.

This is not tidiness. It is what makes the project developable and testable on a
machine with no NVIDIA hardware — which is the common case, and was the case
here. The concurrency runtime, which is where most of the subtle correctness
lives, gets full test and sanitizer coverage regardless of what GPU is present.

The same principle appears three more times:

| Boundary | Portable side | Hardware side |
| --- | --- | --- |
| `MemoryPool<Backend>` | caching, size classes, accounting, thread safety | two-line `cudaMalloc` backend |
| `cudaforge.ops` | reference implementations, dispatch, validation | compiled kernels |
| `InferenceEngine` / `ModelRunner` | queueing, batching, metrics, lifecycle | tokenisation and generation |

In each case the interesting logic sits on the portable side of the line and the
hardware-dependent part is small enough to review by eye.
