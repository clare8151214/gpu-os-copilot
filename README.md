# GPU-OS Copilot

Find out what might be keeping your GPU waiting.

GPU-OS Copilot is a command-line tool for developers investigating CPU/GPU performance
bottlenecks in CUDA workloads. It turns execution traces into timing summaries and candidate
explanations, helping you distinguish CPU submission delays, heavy data transfers and dense
GPU computation.

Start with built-in examples, analyze your own traces, or connect a Meta model to explore the
evidence through tools. A separate benchmark command lets you compare CPU placement settings
for the bundled CUDA workload.

## Install and try it

Use Python 3.11 or newer. From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c requirements-dev.lock -e .

gpu-os-copilot doctor
gpu-os-copilot demo-fixture --scenario cpu_starvation
```

The demo works without a GPU or API key. It prints the observation window, GPU idle fraction,
kernel and transfer durations, candidate explanations and any missing evidence.

`doctor` reports the capabilities available on your machine. Missing GPU or profiling tools
still allow you to use the offline examples and analyze existing trace files.

## Explore the examples

Choose a scenario to see how different workloads affect the diagnosis:

| Scenario | What it illustrates |
| --- | --- |
| `cpu_starvation` | CPU runnable intervals overlapping gaps in GPU activity |
| `transfer_bound` | Data transfers occupying most of the observation window |
| `compute_bound` | Dense GPU computation with little idle time |

```bash
gpu-os-copilot demo-fixture --scenario transfer_bound
gpu-os-copilot demo-fixture --scenario compute_bound
```

All built-in examples use **simulated data**. Their results describe the example, not your GPU.

For recorded measurements from the bundled CUDA workloads, use the samples in this checkout:

```bash
python scripts/replay_demo.py
gpu-os-copilot diagnose --trace examples/measured-traces/cpu_contention.json --mode rules
```

The replay shows saved timings and real Meta tool choices. It makes no new GPU or API requests.
These samples describe one recorded environment; use your own captures to investigate your
application. Profiled trace metrics and unprofiled timing trials are shown separately.

To explore the agent workflow offline:

```bash
gpu-os-copilot demo-fixture --scenario cpu_starvation --mode mock
```

Mock mode uses a scripted agent to inspect the summary and timeline. In the CPU example, it
also requests an affinity experiment plan. It makes no model API requests and runs no GPU
benchmark; its final decision is `inconclusive` because no measured experiment was performed.

Save a structured report with `--json`:

```bash
mkdir -p artifacts
gpu-os-copilot demo-fixture --scenario cpu_starvation --json > artifacts/report.json
```

## Analyze your own trace

Accepted inputs are GPU-OS Copilot canonical JSON traces and Nsight Systems SQLite exports.
A JSON report produced by `--json` is an analysis result, not a trace input.

For a canonical JSON trace:

```bash
gpu-os-copilot diagnose --trace path/to/trace.json --mode rules
```

For a SQLite export, set `WINDOW_START_NS` and `WINDOW_END_NS` to the actual measurement
boundaries in the trace, in nanoseconds. Exclude warm-up and shutdown; the bundled benchmark
marks its measured region with the `copilot_measure` NVTX range.

```bash
gpu-os-copilot diagnose --trace path/to/trace.sqlite \
  --start-ns "$WINDOW_START_NS" --end-ns "$WINDOW_END_NS" \
  --device-id 0 --mode rules
```

Add `--capture-complete` only when the capture contains all relevant workload GPU activity
within that window. Otherwise, GPU idle fraction is reported as unavailable. Use `--simulated`
when importing a synthetic SQLite file.

SQLite import reads supported CUDA runtime, kernel and memory-copy events. CPU scheduling
events and memory-copy directions are not decoded from SQLite; missing information is shown
in the report. CUDA launch delays alone do not establish CPU scheduling delays.

To prepare a trace collection plan for the bundled workload:

```bash
mkdir -p artifacts/traces
gpu-os-copilot trace --scenario cpu_contention --out artifacts/traces/baseline
```

Collection requires Nsight Systems on `PATH` and a CUDA-enabled PyTorch installation.
After reviewing the plan, add `--confirm` to run the bounded workload, export SQLite, and
produce a normalized trace with its measurement window and workload settings attached:

```bash
gpu-os-copilot trace --scenario cpu_contention --out artifacts/traces/baseline --confirm
gpu-os-copilot diagnose --trace artifacts/traces/baseline.capture/trace.json --mode rules
```

Capture defaults to software CUDA/NVTX tracing (`--cuda-trace-mode cuda-sw`); use `cuda` if
your installed Nsight version does not support that option. CPU context-switch collection is
optional via `--cpu-context-switches`; scheduling-delay analysis remains unavailable for
SQLite inputs. Existing capture paths are never overwritten: choose a new output prefix.
Use `--scenario transfer_bound` or `--scenario compute_bound` to capture the other workloads.

## Connect a Meta model

Install the optional API dependency:

```bash
python -m pip install -c requirements-dev.lock -e '.[meta]'
```

Provide your Meta API key through `MODEL_API_KEY`. For example, in Bash, enter it without
displaying it or putting the key itself in a shell command:

```bash
read -rsp "Meta API key: " MODEL_API_KEY
export MODEL_API_KEY

gpu-os-copilot demo-fixture --scenario cpu_starvation \
  --mode meta --allow-paid-api --max-steps 5
```

This sends simulated example evidence to a real Meta model. Meta mode sends trace summaries
and requested timeline excerpts to the Meta API and may incur API charges; `--allow-paid-api`
enables those requests. Using it with a fixture does not run GPU experiments.

The application defaults to `muse-spark-1.3`. Set `META_MODEL` to a model enabled for your account
if needed. Optional environment settings are listed in [.env.example](.env.example); `.env`
files are not loaded automatically.

Use the same `--mode meta --allow-paid-api` options with `diagnose` to analyze your own trace.
Agent runs save tool calls and decisions as JSONL files under `artifacts/agent/`.

To let the agent run a bounded experiment on a trace captured by this tool:

```bash
gpu-os-copilot diagnose --trace artifacts/traces/baseline.capture/trace.json \
  --mode meta --allow-paid-api --confirm
```

The agent can inspect evidence, choose a relevant experiment, and interpret repeated results.
Live experiments require matching bundled-workload metadata; an arbitrary imported trace is
not enough to authorize a comparison for a different workload. Negative controls may lead
the agent to skip an unnecessary experiment.

If a response is cut short, `AGENT_RESPONSE_TRUNCATED` stops the run without executing partial
tool calls. Adjust `META_MAX_COMPLETION_TOKENS` or `META_REASONING_EFFORT` before retrying;
reasoning and visible output share the completion-token budget.

## Compare CPU placement settings

The `affinity_ab` experiment compares two CPU placements for the bundled `launch_test` CUDA
benchmark under controlled CPU contention. It measures synchronized execution time and
throughput across repeated trials. It targets newly launched benchmark processes, not an
existing training or inference job.

Preview the experiment:

```bash
gpu-os-copilot experiment --kind affinity_ab --repeats 5
```

To execute it, use Linux with at least two available CPU cores, an NVIDIA GPU and a compatible
CUDA-enabled PyTorch installation. Check that `doctor` reports `cuda_available: true`, then run:

```bash
gpu-os-copilot experiment --kind affinity_ab --repeats 5 --timeout 60 --confirm
```

`--confirm` starts the benchmark. Without it, the command returns a dry-run plan. The baseline
uses the first available CPU alongside a controlled contender; treatment moves the benchmark
to the second CPU, or to the CPUs supplied with `--cpus`. Choose 2–5 repetitions per condition.
Results include individual trials and aggregate timing/throughput comparisons, saved under
`artifacts/experiments/`.

Add `--scenario transfer_bound` or `--scenario compute_bound` to compare placement for a
control workload. Each condition uses the same scenario, seed and operation count; only
the launcher CPU placement changes.

These timings describe the bundled workload. They do not establish the cause of a slowdown
in a different application or measure GPU idle intervals after the intervention.

## Read the results

- **Measurements:** timing summaries computed from the supplied trace and observation window.
  GPU idle fraction refers to captured activity for that workload and device.
- **Candidate explanations:** interpretations of the evidence, with confidence labels and
  missing-data notes. A candidate is not a confirmed cause.
- **Agent decisions:** tool-driven conclusions. `inconclusive` means the available evidence
  does not support accepting or rejecting the hypothesis.

Use `--json` with `demo-fixture` or `diagnose` for machine-readable output. Add `--help` to
any command to see its options:

```bash
gpu-os-copilot --help
gpu-os-copilot diagnose --help
gpu-os-copilot experiment --help
```
