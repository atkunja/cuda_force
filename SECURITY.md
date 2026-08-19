# Security

## Scope

CudaForge is a research and portfolio project. It has not been audited, and it
should not be exposed to untrusted traffic without a layer in front of it.

What that means concretely:

| Concern | Status |
| --- | --- |
| Authentication | **none** — every endpoint is unauthenticated |
| Rate limiting | **none** beyond the queue bound, which sheds load rather than limiting a caller |
| Input size | prompt length is capped by `max_prompt_chars` and by the schema |
| Model outputs | not filtered — the model returns whatever it generates |
| TLS | terminate it upstream; the server speaks plain HTTP |

Put it behind an authenticating reverse proxy. The bounded queue protects the
*runtime* from overload; it does not protect it from a determined caller.

## What the project does defend against

These are deliberate, and tested:

* **Unbounded memory growth.** Every queue is bounded. Overload becomes
  rejection, not unbounded latency followed by exhaustion.
* **Malformed requests.** Validated at the HTTP boundary and rejected with 422
  before they reach the engine or occupy a queue slot.
* **A crash taking the service down.** A failing model, a failing batch or a
  failing task is isolated and counted; the runtime keeps serving.
* **Requests nobody is waiting for.** Optional deadlines stop the runtime
  spending capacity on work that has already timed out.

## What it does not

* **Prompt injection or jailbreaks.** Out of scope — this is a runtime, not a
  model or a policy layer.
* **Multi-tenancy.** There is no per-caller isolation, quota or accounting. A
  single caller can fill the queue.
* **Weight integrity.** Models are downloaded from Hugging Face by name and are
  trusted. Verify checksums yourself if that matters.
* **Side-channel isolation between requests in a batch.** Requests share a batch
  and therefore share timing.

## Dependencies

Runtime dependencies are torch and numpy. Everything needed only for training
or serving lives in an optional extra, so the base install pulls in less.

`.github/dependabot.yml` groups updates weekly. Major torch versions are
deliberately held: they can change dispatcher behaviour and the extension ABI,
which needs a deliberate migration rather than a green checkmark.

## Reporting

Open a GitHub issue. Given the scope above, there is no private disclosure
process and no expectation of one — please do not treat this as a project with
a security response commitment it does not have.
