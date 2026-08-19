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

## Why a mutex, not atomics

The queue protects a composite invariant, not a single word. A producer must
observe "not closed AND not full" and then insert, with no window in between
where another thread could close the queue or take the last slot.

Atomics cannot express a check-then-act over two pieces of state. A lock-free
formulation would need a CAS over a value encoding both, plus a solution for
the ABA problem, and would still need somewhere for blocked producers to wait.
For a queue crossed once per request — not once per kernel — the mutex is not
on any hot path worth optimising.

The measurable claim is in `benchmarks/benchmark_queue.cpp`, which sweeps
producer and consumer counts and shows where the single lock stops scaling. If
that point sits below the target concurrency, the fix is sharding the queue,
not removing the lock.

## Why condition variables, not polling

A consumer waiting on an empty queue has nothing useful to do. The options are:

| Approach | Cost |
| --- | --- |
| Spin | burns a core; on an oversubscribed machine, steals cycles from the producer it waits for |
| Sleep-and-retry | adds latency quantised to the sleep interval; wrong at both ends |
| Condition variable | thread parks in the kernel; producer's notify wakes it at the state change |

The condition variable is the only one that is both cheap while idle and prompt
when work arrives.

### Spurious wakeups

`wait` may return without a matching notify. Every wait in
[`concurrent_queue.hpp`](../cpp/include/cudaforge/concurrent_queue.hpp) uses the
predicate overload, which re-checks the condition and loops. There is no bare
`wait(lock)` in the file, and there must not be — one would let a consumer
proceed to pop from an empty deque.

### Lock ownership

`std::unique_lock` rather than `std::lock_guard`, because
`condition_variable::wait` must release and reacquire the mutex. Notifications
are issued after the lock is released, so a woken thread does not immediately
block on a mutex the notifier still holds.

## Backpressure

The queue is bounded. That is a design decision with consequences worth being
explicit about.

An unbounded queue never rejects anything, which sounds better than it is. Under
sustained overload it converts a throughput problem into two worse problems:
memory grows without limit, and queue delay grows without limit — so every
request eventually times out, including ones that could have been served. The
system fails late and totally instead of early and partially.

A bounded queue forces a choice at the ingress, and CudaForge exposes both:

| Call | Behaviour when full | Use when |
| --- | --- | --- |
| `push` / `submit` | blocks the caller | the caller can slow down — batch clients, load generators |
| `try_push` / `try_submit` | returns `Full` immediately | the caller cannot slow down — an HTTP frontend that should return 503 |

The HTTP server uses the second. A client holding a connection open behind a
full queue turns a throughput problem into a connection-exhaustion problem;
returning 503 lets the client retry or fail fast.

## Shutdown

Shutdown must not lose accepted work and must not hang.

```
shutdown()
  ├─ set closed_               producers begin failing
  ├─ notify_all on both CVs    every blocked thread wakes and re-checks
  └─ join                      workers exit as their loops return Closed
```

The rule that makes this safe: **a closed queue with items still in it returns
`Ok`, not `Closed`.** Consumers keep draining; only an empty *and* closed queue
reports `Closed`. Work already accepted is therefore always executed.

`shutdown()` is idempotent everywhere it appears, and every destructor calls it.
The engine additionally fails any future still outstanding after the drain —
otherwise a caller blocked in `result()` would hang until its timeout with no
explanation.

## Dynamic batching

GPU inference is dominated by weight movement, not arithmetic: running one
request through a transformer layer reads the same weights as running sixteen.
Batching amortises that read, so throughput rises close to linearly with batch
size until the kernel becomes compute-bound.

The cost is latency. A request arriving into an empty queue would be fastest
executed immediately; batching makes it wait. A batch therefore closes on
whichever comes first:

1. it holds `max_batch_size` requests, or
2. the **oldest** request in it has waited `max_wait`.

### The deadline is anchored, not sliding

```
t=0    request A arrives, deadline set to t=5ms
t=1ms  request B arrives, deadline unchanged
t=2ms  request C arrives, deadline unchanged
t=5ms  batch [A, B, C] executes
```

If the deadline were reset on each arrival, a steady stream would postpone
execution indefinitely and no request would have a bounded wait — the classic
Nagle-style starvation bug. Anchored to the oldest request, nothing waits longer
than `max_wait` plus the batch's own service time.

`tests/python/test_scheduler.py::test_the_deadline_is_anchored_to_the_oldest_request`
and the matching C++ case assert exactly this, by trickling requests faster than
the deadline and requiring that batches still close.

### Batch formation is single-threaded

Formation is an inherently serial decision. Two threads racing to claim requests
from one queue would need a lock that serialises them anyway, and would make the
deadline non-deterministic. Parallelism belongs downstream: the batcher hands
each formed batch to a worker pool and immediately returns to forming the next.

If the batcher executed batches itself, no batch could form while one ran, and
arrival-time batching would collapse into strict serialisation — every batch
would be a full one assembled from the backlog that accumulated during the
previous execution, regardless of the configured wait.

### Reading the trigger counters

`batches_closed_by_size` and `batches_closed_by_timeout` are reported separately
because average batch size alone cannot distinguish two very different states:

| Observation | Meaning | Action |
| --- | --- | --- |
| Nearly all timeout closures, small batches | arrival rate too low to fill a batch | lower `max_wait`; it is pure added latency |
| Nearly all size closures, queue depth rising | saturated | raise `max_batch_size`, or add capacity |
| Mixed, batch size near the limit | well matched | leave it alone |
