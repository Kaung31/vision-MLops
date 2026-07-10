# CLAUDE.md — Standing instructions for this repo

You are helping build `traffic-vision-platform`, a production-grade vision MLOps
system. The full specification is in `docs/vision-mlops-platform-project-guide.md`.
**At the start of every session, read: (a) the guide's Section 0 (Laws) and
Section 2 (frozen stack), and (b) the full section for the phase we are currently
in — including its exit criteria and Forbidden list.** Read other sections when the
work touches them. This file is the short, always-on version of the rules.

## How we work

- **Follow the guide's phases in order.** Do not start a phase until the previous
  phase's exit criteria are met and committed. If a phase isn't done, say so —
  do not skip ahead "to save time."
- **One phase at a time.** At the start of a session, tell me which phase we are
  in and what its exit criteria are. Do not touch files outside the current
  phase's scope without asking.
- **Propose before you build.** For any non-trivial file, outline the approach in
  a sentence or two and wait for my go-ahead. No large unrequested code dumps.
- **When unsure, ask — do not guess.** Especially about dataset facts, licenses,
  or an API you're not certain is current. Wrong assumptions here cost weeks.

## The Laws (non-negotiable — from the guide)

1. **Stack is frozen.** Use only what the guide's Section 2 lists. No substituting
   a tool because it's easier or more familiar. If something seems impossible with
   the chosen tool, flag it — don't silently swap.
2. **No metric before the eval harness exists (Phase 2).** Do not train, quote, or
   trust any mAP number until the harness is built and frozen.
3. **Splits are by camera location and clip — NEVER by frame.** Frame-level splits
   are data leakage. The splitter has tests; nothing bypasses it.
4. **Frozen eval sets and the production-holdout are never trained on.** Enforce
   with a code guard in the data loader, not memory.
5. **No images in git.** Only code, configs, DVC pointers, checksums, download
   scripts. Datasets are non-commercial licensed — never redistribute frames.
6. **Every training run is reproducible or it didn't happen.** Config + dataset
   version hash + git commit + seed, all logged to MLflow. No notebooks for real
   runs — scripts only.
7. **Honest docs.** Drift is scripted, the "labeler" is me with held-back ground
   truth, Kafka is single-node. Never let the README overclaim.
8. **Vertical slices.** The system runs end-to-end at its current depth after every
   phase. Never leave two half-built layers at once.

## MLflow specifics

- Use **model version aliases** (`@champion`, `@challenger`) — NOT registry
  "stages" (deprecated since MLflow 2.9). Serving loads `models:/<name>@champion`.
  Rollback = reassign the alias to the previous version.

## Hardware routing (READ THIS — the dev machine is Apple Silicon, no CUDA)

Every task runs on the correct machine. Never quote a number from the wrong one.

- **This Mac (M-series) = platform machine.** All non-GPU work: repo, DVC, data
  hygiene, splitters, eval-harness *code*, MLflow server, and the Docker serving
  stack (Kafka, Prometheus, Grafana, consumers — all fine on ARM).
  - Training here is **MPS smoke-tests only** (`device="mps"`, 2–3 epochs to prove
    the code runs). Never a real training run. Never in Docker (macOS containers
    can't reach the Apple GPU).
  - Local Triton is CPU-only and **known to be flaky on Apple Silicon** (can abort
    on first request). Treat it as an optional dev convenience. If it won't run,
    develop the consumer against an ONNX Runtime stub exposing the same gRPC
    contract, and note it in an ADR.
- **Colab Pro = training machine.** ALL real training, tuning, learning-curve, and
  full eval-suite runs. Invoke via the **Colab CLI** (`colab exec <script>` — the
  exact argument-passing syntax is NOT assumed here; check `colab exec --help`
  before writing any workflow that calls it, and pin the verified invocation in an
  ADR) so runs stay plain scripts (Law 6). Every run must
  checkpoint-and-resume — Colab sessions die and premium GPUs aren't guaranteed.
  Reserve A100 for final full-size runs; default to T4.
- **Rented Linux GPU box = measurement machine.** ALL measured serving numbers:
  Triton dynamic-batching study, SLO validation, load tests, GPU chaos tests. Deploy
  the same docker-compose stack there, run the benchmark, tear down. TensorRT
  conversion happens here (hardware-specific). SLOs are declared against THIS
  machine class and labeled as such.

If a task needs a GPU and we're on the Mac, stop and route it to Colab or the
rented box — do not fall back to running it slowly/emulated and reporting the
result.

## CI / runner

- Mac is a **self-hosted GitHub Actions runner for CPU jobs only** (lint, mypy,
  pytest, promotion-contract logic, report generation).
- The training step in `train.yml` calls the Colab CLI — it does not assume a
  local GPU.
- Self-hosted runner must never run workflows from forks.

## Definition of done for any unit of work

Code is done when: it has tests (leakage/guard tests where the guide requires
them), passes ruff + mypy + pytest, is reproducible, and the relevant ADR or
experiment write-up exists in `docs/`. "It runs" is not "it's done."
