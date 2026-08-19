# Concurrency Architecture

How requests get from many client threads onto a GPU, and why each piece is
shaped the way it is.

## The pipeline

```mermaid
flowchart TD
    C1[Client thread 1] --> Q
    C2[Client thread 2] --> Q
    CN[Client thread N] --> Q

    Q["ConcurrentQueue&lt;Request&gt;<br/>bounded · mutex + condvar"] --> B

    B["DynamicBatcher<br/>single thread"] --> P

    P["ThreadPool / executor<br/>W workers"] --> S

    S["GpuScheduler<br/>K CUDA streams"] --> G1
    S --> G2
    S --> GK

    G1[Stream 0] --> R[Responses]
    G2[Stream 1] --> R
    GK[Stream K-1] --> R

    R --> F[Per-request futures]
```

Each stage exists to decouple a rate mismatch:

| Stage | Decouples | Failure mode without it |
| --- | --- | --- |
| Bounded queue | arrival rate from service rate | unbounded latency, then OOM |
| Batcher | request granularity from GPU granularity | one kernel launch per request |
| Worker pool | batch formation from batch execution | formation stalls during execution |
| Stream scheduler | copies from compute | GPU idle during every transfer |
