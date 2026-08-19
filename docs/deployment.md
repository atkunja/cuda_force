# Deployment

Running the inference server as a long-lived service. Everything here concerns
the serving path; training is a batch job and is covered in
[fine-tuning.md](fine-tuning.md).

> The container has not been built or run for this repository — the development
> host is Apple Silicon, where `nvidia/cuda` images are the wrong architecture
> and no GPU can be passed through. See [environment.md](environment.md).

## Container

```bash
docker build -t cudaforge:latest .
docker run --gpus all -p 8000:8000 cudaforge:latest \
  python3 -m uvicorn inference.server:app --host 0.0.0.0 --port 8000
```

The host needs an NVIDIA driver and the NVIDIA Container Toolkit. Without the
toolkit the container starts but sees no GPU, and the server degrades to the
reference implementations — check `/health` rather than assuming.

The image is multi-stage: the builder carries the full CUDA toolkit, the
runtime carries only the CUDA runtime libraries. It runs as a non-root user,
because a container writing checkpoints into a mounted volume as root leaves
files the host user cannot delete.

## Probes

The two endpoints answer different questions, and wiring them the wrong way
round causes an outage rather than preventing one.

| Endpoint | Question | A failure means |
| --- | --- | --- |
| `/health` | is this process alive? | restart it |
| `/ready` | should it receive a request right now? | take it out of rotation |

`/ready` returns 503 once the queue is above 90% of capacity. Pointing a
*liveness* probe at it makes an orchestrator restart an instance that is merely
busy — discarding the queued work it was draining and pushing that load onto
its peers, which then also become busy. That failure cascades.

```yaml
livenessProbe:
  httpGet: { path: /health, port: 8000 }
  initialDelaySeconds: 60      # model loading and warmup
  periodSeconds: 30
  failureThreshold: 3

readinessProbe:
  httpGet: { path: /ready, port: 8000 }
  initialDelaySeconds: 30
  periodSeconds: 5             # short: this is the rotation signal
  failureThreshold: 1
```

`initialDelaySeconds` on liveness must exceed model load plus warmup, or the
orchestrator kills the pod during startup and retries forever.

## Graceful shutdown

The server's lifespan handler calls `engine.shutdown()`, which stops accepting
work, drains the batcher, waits for in-flight batches, and settles every
outstanding future. Requests already accepted are completed rather than dropped.

`terminationGracePeriodSeconds` must exceed the longest generation the engine
will accept, or the container is killed mid-batch and those requests are lost:

```yaml
terminationGracePeriodSeconds: 120
```

## Sizing

Two knobs dominate, and they trade against each other:

| Symptom | Read | Change |
| --- | --- | --- |
| `timeout_closure_fraction` near 1.0, small batches | arrivals never fill a batch | lower `max_wait_us` — it is pure added latency |
| `average_batch_size` at the limit, `queue_depth` rising | saturated | raise `max_batch_size`, or scale out |
| `requests_rejected` climbing | queue full | raise `queue_capacity` only if latency allows; otherwise this is load shedding working |
| `requests_expired` climbing | queue deeper than clients will wait | shed earlier, or add capacity |

`average_batch_size` saturating at the *client* count rather than at
`max_batch_size` is normal: N clients blocking on responses can have at most N
requests in flight.

## Scaling out

Horizontal, on `/ready`. Each instance owns one model replica and one queue,
and there is no shared state between them, so a load balancer with least-outstanding-requests
is enough — round-robin sends work to instances that have already reported
themselves unready.

Scaling on CPU utilisation is misleading here: the process is mostly waiting on
the GPU. `cudaforge_queue_depth` and `cudaforge_latency_p99_ms` are the signals
that reflect what is actually happening.

## Monitoring

```
GET /metrics             JSON, for a human
GET /metrics/prometheus  text exposition, for a scraper
```

Alerts worth having, in rough order of value:

| Alert | Condition | Why |
| --- | --- | --- |
| Saturation | `cudaforge_queue_depth` near capacity for minutes | rejections are next |
| Tail latency | `cudaforge_latency_p99_ms` above the SLO | the number users feel |
| Shedding | `rate(cudaforge_requests_rejected_total[5m]) > 0` | capacity is short |
| Expiry | `rate(cudaforge_requests_expired_total[5m]) > 0` | queue deeper than clients will wait |
| Failures | `rate(cudaforge_requests_failed_total[5m]) > 0` | a bug or a bad request |
| Wrong path | `custom_cuda_kernels` false on `/health` | silently running the reference implementations |

The last one is easy to overlook and expensive: a deployment that fell back to
the reference operators works correctly and is simply slow, which looks like a
capacity problem and gets solved by adding instances.

## Configuration

Three shipped configs, as starting points rather than answers:

| Config | `max_batch_size` | `max_wait_us` | For |
| --- | --- | --- | --- |
| `latency.yaml` | 4 | 1,000 | interactive traffic |
| `balanced.yaml` | 16 | 5,000 | before you have measured |
| `throughput.yaml` | 64 | 20,000 | batch and offline traffic |

```bash
cudaforge-serve --config inference/configs/balanced.yaml
```

Individual flags override the file, so a config can be adjusted without editing
it. Run `benchmarks/benchmark_batching.py` against your model before settling on
values — the right ones depend on the service time, which depends on the model.

Environment variables read at startup:

| Variable | Default | Effect |
| --- | --- | --- |
| `CUDAFORGE_MODEL` | `sshleifer/tiny-gpt2` | model to load |
| `CUDAFORGE_MAX_BATCH` | 16 | `max_batch_size` |
| `CUDAFORGE_MAX_WAIT_US` | 5000 | `max_wait_us` |
| `CUDAFORGE_ECHO_RUNNER` | unset | use the deterministic runner; useful for load-testing the runtime without a model |

If the model fails to load, the server logs a warning and starts with the
deterministic runner rather than crashing — a startup crash gives no signal
about which part failed. `/health` reports the active model, so a degraded start
is visible rather than silent.
