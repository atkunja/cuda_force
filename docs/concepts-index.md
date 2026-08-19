# Concepts Index

Every claim this repository makes, mapped to the code that backs it. If a
concept is listed here, there is a file and a line you can open.

Ordered by the question someone would actually ask.

## Concurrency

| Question | Code | Discussion |
| --- | --- | --- |
| Why is a mutex needed here at all? | [`concurrent_queue.hpp`](../cpp/include/cudaforge/concurrent_queue.hpp) — class comment | [concurrency.md](concurrency.md#why-a-mutex-not-atomics) |
| Why not atomics? | same — the composite `closed_` + `buffer_` invariant | same |
| Why condition variables over polling? | `push` / `pop` predicate waits | [concurrency.md](concurrency.md#why-condition-variables-not-polling) |
| What is a spurious wakeup and how is it handled? | every `wait` uses the predicate overload | same |
| Why `unique_lock` and not `lock_guard`? | `pop` — `wait` must release the mutex | same |
| What happens on shutdown? | `shutdown()`, and the drain rule in `pop` | [concurrency.md](concurrency.md#shutdown) |
| How does backpressure work? | `push` blocks; `try_push` returns `Full` | [concurrency.md](concurrency.md#backpressure) |
| Where are atomics used, and why relaxed? | [`thread_pool.hpp`](../cpp/include/cudaforge/thread_pool.hpp), [`metrics.hpp`](../cpp/include/cudaforge/metrics.hpp) | [architecture.md](architecture.md#threading-model-summary) |
| How do you know there are no races? | TSan-clean across 114 cases | [testing.md](testing.md) |
| What if a task throws? | `ThreadPool::run` catch block | [architecture.md](architecture.md#failure-isolation) |
| How do futures get their values? | `submit_with_result`, `packaged_task` behind a `shared_ptr` | — |

## Batching and serving

| Question | Code | Discussion |
| --- | --- | --- |
| How are requests dynamically batched? | [`dynamic_batcher.cpp`](../cpp/src/dynamic_batcher.cpp) `collect_batch` | [concurrency.md](concurrency.md#dynamic-batching) |
| Why does batching help at all? | — | [concurrency.md](concurrency.md#dynamic-batching) — weight movement, not arithmetic |
| What stops a steady stream from starving a request? | the deadline is anchored to the first request | [concurrency.md](concurrency.md#the-deadline-is-anchored-not-sliding) |
| Why is batch formation single-threaded? | `DynamicBatcher` owns one thread | same |
| Why does the batcher not execute batches? | `_dispatch` hands off to the pool | [architecture.md](architecture.md#request-lifecycle) |
| Throughput versus latency — which knob? | `max_batch_size` vs `max_wait_us` | [config.hpp](../cpp/include/cudaforge/config.hpp) class comment |
| How do you tell a starved batcher from a saturated one? | `batches_closed_by_size` / `_by_timeout` | [concurrency.md](concurrency.md#reading-the-trigger-counters) |
| What is p99 and how is it computed? | [`latency_histogram.hpp`](../cpp/include/cudaforge/latency_histogram.hpp) | [testing.md](testing.md) |
| Why bucket latencies instead of storing them? | O(1) memory under sustained load | class comment |
| How is load shed? | `try_submit`, and the server's `block_when_full=False` | [concurrency.md](concurrency.md#backpressure) |
| What happens to a request nobody is waiting for? | `Request.expired`, `_drop_if_expired` | [concurrency.md](concurrency.md#deadline-aware-admission) |
| Why check the deadline twice? | `DynamicBatcher._collect` and `InferenceEngine._execute` | [concurrency.md](concurrency.md#checked-twice-and-why) |
| Rejected vs expired vs failed? | three separate counters | [concurrency.md](concurrency.md#three-counters-three-different-problems) |

## CUDA execution

| Question | Code | Discussion |
| --- | --- | --- |
| How do CUDA streams overlap work? | [`gpu_scheduler.cuh`](../cuda/include/cudaforge/gpu_scheduler.cuh) class comment | [gpu-execution.md](gpu-execution.md#streams-are-the-ordering-primitive) |
| What must be true for overlap to actually happen? | — | [gpu-execution.md](gpu-execution.md#three-conditions-all-required) |
| Why `cudaStreamNonBlocking`? | `CudaStream` constructor | [gpu-execution.md](gpu-execution.md#non-blocking-streams) |
| When is synchronisation required? | `stream_wait_event`, `GpuScheduler::chain` | [gpu-execution.md](gpu-execution.md#cross-stream-dependencies) |
| Why never `cudaDeviceSynchronize`? | rejected by `check_cuda_sources.py` | same |
| Timing vs ordering events? | `CudaEvent::Purpose` | [gpu-execution.md](gpu-execution.md#events) |
| How do you time GPU work correctly? | `CudaEvent::elapsed_ms` | [gpu-execution.md](gpu-execution.md#timing-gpu-work) |
| Why pinned host memory? | [`PinnedBuffer`](../cuda/include/cudaforge/cuda_raii.cuh) | [memory-management.md](memory-management.md#pinned-host-memory) |
| How are streams assigned? | `GpuScheduler::acquire`, round-robin | [gpu-execution.md](gpu-execution.md#stream-assignment) |
| How many streams, and why? | — | [gpu-execution.md](gpu-execution.md#how-many-streams) |
| What does RAII buy here? | `CudaStream`, `CudaEvent`, `DeviceBuffer` | class comments |
| How are CUDA errors handled? | [`cuda_error.cuh`](../cuda/include/cudaforge/cuda_error.cuh) | [cuda-kernels.md](cuda-kernels.md#error-handling) |
| Which errors are recoverable? | `CudaError::is_sticky` | same |
| Why two checks after a launch? | `CUDAFORGE_CHECK_LAUNCH` | same |

## Kernels

| Question | Code | Discussion |
| --- | --- | --- |
| What is a warp, and why does it matter? | [`cuda_utils.cuh`](../cuda/include/cudaforge/cuda_utils.cuh) `warp_reduce_sum` | [gpu-execution.md](gpu-execution.md#warp-execution) |
| Why does the shuffle need a mask? | same — mask parameter comment | same |
| How does a parallel reduction work? | [`reduction.cu`](../cuda/src/reduction.cu), three variants | [cuda-kernels.md](cuda-kernels.md#kernel-a--reduction) |
| Why is one atomic per element slow? | `reduce_sum_naive` | same |
| What is memory coalescing? | `row_sum`, `rmsnorm_naive` | [cuda-kernels.md](cuda-kernels.md#what-actually-costs-time) |
| Global vs shared memory? | `softmax_shared`, `matmul_tiled` | same |
| What is a bank conflict? | `tile_b[kTile][kTile + 1]` | [cuda-kernels.md](cuda-kernels.md#the-tiled-matmul-underneath) |
| Why is softmax numerically unstable naively? | `softmax_naive` max subtraction | [cuda-kernels.md](cuda-kernels.md#numerical-stability-is-not-optional) |
| What is online softmax? | `softmax_online` | [cuda-kernels.md](cuda-kernels.md#online-softmax) |
| What is RMSNorm, and how does it differ from LayerNorm? | [`rmsnorm.cuh`](../cuda/include/cudaforge/rmsnorm.cuh) | [cuda-kernels.md](cuda-kernels.md#kernel-c--rmsnorm) |
| Why vectorise with `float4`? | `rmsnorm_vectorised` | [cuda-kernels.md](cuda-kernels.md#vectorisation) |
| When is `float4` unsafe? | `vectorisable()` alignment check | same |
| What is kernel fusion, and when does it pay? | `lora_fused` | [cuda-kernels.md](cuda-kernels.md#why-fusion-pays-here) |
| Why FP32 accumulators for FP16 data? | `rmsnorm_half`, `softmax_half` | [cuda-kernels.md](cuda-kernels.md#fp16-overflow) |
| What is occupancy, and is more always better? | `kDefaultBlockSize` comment | [gpu-execution.md](gpu-execution.md#occupancy-is-a-means-not-an-end) |
| How is grid size chosen? | `reduction_grid_size` | [gpu-execution.md](gpu-execution.md#grid-size) |
| What is SwiGLU and why fuse it? | [`activations.cu`](../cuda/src/activations.cu) | [cuda-kernels.md](cuda-kernels.md#swiglu-and-why-it-is-a-kernel-at-all) |
| Why tanh GELU rather than erf? | `gelu_tanh` | [cuda-kernels.md](cuda-kernels.md#gelu-tanh-not-erf) |
| Can you backprop through these kernels? | the Autograd registration in [`bindings.cpp`](../cpp/src/bindings.cpp) | [cuda-kernels.md](cuda-kernels.md#autograd) |

## Memory

| Question | Code | Discussion |
| --- | --- | --- |
| Why does `cudaMalloc` hurt? | [`gpu_scheduler.cuh`](../cuda/include/cudaforge/gpu_scheduler.cuh) `DeviceAllocatorBackend` | [memory-management.md](memory-management.md#the-problem) |
| How does the pool work? | [`memory_pool.hpp`](../cpp/include/cudaforge/memory_pool.hpp) | [memory-management.md](memory-management.md#the-pool) |
| Why power-of-two size classes? | `round_up` | [memory-management.md](memory-management.md#size-classes) |
| What does the pool *not* do? | — | [memory-management.md](memory-management.md#what-this-is-not) |
| How does it compare to `cudaMallocAsync`? | — | same |
| Why is it templated on a backend? | `AllocatorBackend` concept | [memory-management.md](memory-management.md#testing-without-a-gpu) |
| What metrics matter? | `PoolStats` | [memory-management.md](memory-management.md#metrics-worth-watching) |

## Machine learning

| Question | Code | Discussion |
| --- | --- | --- |
| What is LoRA? | [`lora.py`](../training/lora.py) `LoRALinear` | [fine-tuning.md](fine-tuning.md#why-lora) |
| Why does it save memory? | — | [fine-tuning.md](fine-tuning.md#why-lora) — optimiser state, not weights |
| Why must B start at zero? | `LoRALinear.__init__` | [fine-tuning.md](fine-tuning.md#initialisation-is-not-arbitrary) |
| Why not both at zero? | same | same |
| What does alpha do? | `scaling` property | [fine-tuning.md](fine-tuning.md#alpha-is-not-a-second-learning-rate) |
| Can adapters be merged? | `merged_weight`, `merge` | [fine-tuning.md](fine-tuning.md#merging-is-exact) |
| What is QLoRA, and is it implemented here? | `build_model` `load_in_4bit` | [fine-tuning.md](fine-tuning.md#qlora) |
| Is the custom INT8 kernel used for QLoRA? | **no** | same — scope stated explicitly |
| How does quantisation work? | [`quantization.cu`](../cuda/src/quantization.cu) | [cuda-kernels.md](cuda-kernels.md#kernel-e--quantise--dequantise) |
| Why block-wise scales? | `kQuantBlockSize` comment | same |
| What is the error bound? | tested at `scale / 2` | same |
| What is gradient accumulation? | `train()` accumulation branch | [fine-tuning.md](fine-tuning.md#gradient-accumulation) |
| Why divide the loss by k? | same | same |
| What is loss scaling for? | `GradScaler`, fp16-only | [fine-tuning.md](fine-tuning.md#mixed-precision) |
| Why unscale before clipping? | `scaler.unscale_` then `clip_grad_norm_` | same |
| bf16 vs fp16? | `EngineConfig.resolve_dtype` | [fine-tuning.md](fine-tuning.md#mixed-precision) |
| What is gradient checkpointing? | `gradient_checkpointing_enable` | [fine-tuning.md](fine-tuning.md#qlora) |
| Why left padding for batched generation? | `TransformersRunner.__init__` | class comment |
| How is data packed for causal LM? | [`dataset.py`](../training/dataset.py) | [fine-tuning.md](fine-tuning.md#data-packing) |
| What is DDP, and what does AllReduce do? | [`distributed_train.py`](../examples/distributed_train.py) | module docstring |
| Why AllReduce over a parameter server? | same | same |
| How does communication overlap computation? | same — gradient buckets | same |
| Why is LoRA cheap under DDP? | same | same |

## Performance and measurement

| Question | Code | Discussion |
| --- | --- | --- |
| How do you know any of this is fast? | — | [performance.md](performance.md) — most of it is marked unmeasured |
| What *has* been measured? | — | [performance.md](performance.md#measured-on-the-development-host) |
| How do you benchmark correctly? | [`bench_common.hpp`](../benchmarks/bench_common.hpp) | [benchmarking.md](benchmarking.md#method) |
| Why warm up? | `kWarmupRuns` | same |
| Why the median, not the mean? | `percentile` | same |
| How is dead-code elimination prevented? | `bench::keep` | same |
| What is effective bandwidth for? | `effective_bandwidth` | same |
| How would you profile this? | [`profile.sh`](../scripts/profile.sh) | [profiling.md](profiling.md) |
| What does Nsight Systems tell you? | — | [profiling.md](profiling.md#nsight-systems-is-the-gpu-busy) |
| What does Nsight Compute tell you? | — | [profiling.md](profiling.md#nsight-compute-why-is-this-kernel-slow) |
| How do you detect false sharing? | — | [profiling.md](profiling.md#profiling-the-cpu-side) |
| How are metrics exposed to a scraper? | [`exposition.py`](../python/cudaforge/exposition.py) | counter vs gauge typing matters for `rate()` |
| Liveness vs readiness? | `/health` vs `/ready` in [`server.py`](../inference/server.py) | [troubleshooting.md](troubleshooting.md#the-orchestrator-keeps-restarting-the-container) |
