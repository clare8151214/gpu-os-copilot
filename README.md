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
The replay also shows a later manual repeat, so changes between batches remain visible.

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

SQLite import reads supported CUDA runtime, kernel, memory-copy and memory-set events.
GPU busy time uses their interval union across streams. Nonempty unsupported activity tables
make the trace partial, even with `--capture-complete`, so missing activity cannot appear as idle.
New exports may contain `memset` events; upgrade older readers before analyzing those exports.
CPU scheduling
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

For a Linux host that supports scheduler events through tracefs, an optional helper can collect
the bundled workload's threads alongside CUDA, including the training autograd launcher:

```bash
gpu-os-copilot trace --out artifacts/traces/scheduled --sudo-scheduler
gpu-os-copilot trace --out artifacts/traces/scheduled --sudo-scheduler --confirm
```

Review the plan first. `--sudo-scheduler` explicitly permits a short-lived root helper through
noninteractive sudo; the profiler and workload still run as your normal user. The helper uses
a private tracefs instance with thread filters and removes it after collection. At most 32
threads of the workload are enrolled after warmup; their names are temporarily replaced with
unique identifiers until the workload exits. The helper verifies ownership, process membership,
start times and kernel IDs before and after collection, using `/sys/kernel/debug/sched/debug`
for PID namespaces. Each observed CUDA API thread needs its own switch and wakeup evidence;
late or unverified threads remain explicitly uncovered.
It does not change system-wide tracing settings, sysctls or directory permissions. Unsupported
events or permissions cause capture failure with retained logs. This mode cannot be combined
with `--cpu-context-switches`.

Runnable intervals are accepted only when the launch PID/TID matches, no events were lost,
the capture covers the measurement window and five clock markers validate the timestamp
offset and clock-rate conversion. The report records alignment uncertainty and the assumed
linear clock relation between samples. Sleep intervals are not counted as runnable
delay; measured overlap still does not establish a sole cause of GPU inactivity.

After capturing baseline and treatment with the same workload and instrumentation, compare
their aligned scheduling delay, overlap and GPU idle fraction from this checkout:

```bash
python scripts/compare_scheduler.py \
  --baseline artifacts/traces/baseline.capture/trace.json \
  --treatment artifacts/traces/treatment.capture/trace.json \
  --out artifacts/traces/scheduler-comparison.json
```

The comparison requires complete, aligned evidence and verifies that only launcher placement
changed. It reports two profiled observations; use the separate repeated experiment command
to evaluate throughput without profiler overhead.

For repeated CPU scheduling comparisons, create a JSON manifest alongside the recorded
captures. Include every planned pair, using `null` for a capture that was not obtained:

```json
{
  "pairs": [
    {
      "batch": "batch-1",
      "baseline": "baseline.capture/trace.json",
      "treatment": "treatment.capture/trace.json"
    }
  ]
}
```

Paths must stay within the manifest directory. The reader accepts up to 12 pairs, 32 MiB per
trace and 128 MiB in total. Both arms require matching workload/software settings, controlled
placement, complete observed CUDA-thread coverage, zero trace loss and reproducible five-point
CPU clock calibration. Missing or invalid pairs remain visible and withhold aggregate effects.

```bash
gpu-os-copilot diagnose --trace artifacts/traces/baseline.capture/trace.json \
  --scheduler-study artifacts/traces/study.json --mode rules --json
```

This reads saved files only. It reports CPU wait differences, paired variation and profiled
throughput separately from GPU timing. A consistent CPU-delay direction requires at least five
valid pairs across three batches, with every difference exceeding its conditional clock-error
bound. These descriptive checks do not establish statistical significance or GPU idle causation.
With the Meta options below, the same flag offers a read-only scheduling tool; its numerical
results become available for model citations only after the model reads them. They do not
count as unprofiled throughput trials or authorize new captures.

Scheduling metrics distinguish `unread`, `available` and `unavailable`. The report includes
verified pair counts and counts of pairs with observed GPU/API timestamp contradictions;
an incomplete study withholds the latter aggregate. These counts describe recorded ordering,
not an independently calibrated GPU clock.

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
The JSON report separates the model's `proposed_decision` from the verified `decision`.
A deterministic critic recomputes consumed benchmark batches and records its reasons in
`verification`; it does not make additional model calls. Cited events must have been provided
to the model and must overlap the observation window on the selected device.
The `investigation` section lists ranked candidate explanations with supporting and
contradicting metric/event references, missing telemetry and testable predictions. Numerical
facts come from the trace; model prose remains an interpretation. Insufficient evidence can
produce an explicit unknown diagnosis.

Final narrative numbers are inserted from consumed evidence references, such as
`{{metric:cpu_runnable_delay_ms}}` or `{{percent:gpu_idle_fraction}}`. Count ratios use
`{{ratio:profiled_gpu_api_contradictory_pair_count:profiled_scheduling_pair_count}}`.
The controller checks availability, units and count scope, and asks for a correction when a
reply copies literal numbers or references unavailable data. Corrections use the existing
response budget; an unresolved numeric error fails the run instead of publishing that conclusion.
The `numeric_grounding` report records the source values and rendered references. The original
proposal remains in `proposed_decision` and the audit. Reference checks do not verify the meaning
of prose, spelled-out quantities or causal interpretations; `no_references` is not a factual
accuracy score. Benchmark statistics remain separately available in `verification`.

To let the agent run a bounded experiment on a trace captured by this tool:

```bash
gpu-os-copilot diagnose --trace artifacts/traces/baseline.capture/trace.json \
  --mode meta --allow-paid-api --confirm
```

The agent can inspect evidence, choose a relevant experiment, and interpret repeated results.
Live experiments require matching bundled-workload metadata; an arbitrary imported trace is
not enough to authorize a comparison for a different workload. Negative controls may lead
the agent to skip an unnecessary experiment.
Before live execution, the agent proposes a plan with possible outcomes, expected information
gain and bounded cost. The controller checks the plan against available CPUs, workload and
environment, remaining budget and existing evidence. A changed or unapproved plan cannot run.
Recorded comparison reads require no new GPU execution. Reports distinguish plans from
consumed actions in `planning`, including abstention and any unexecuted pending action.
Stopping an intervention does not require discarding the ranked symptom hypotheses.
When skipping an offered scheduling study, the model can provide `deferred_evidence` with
its reason. The `evidence_access` report distinguishes that choice from a silently unread
study. If capacity remains, the controller requests a read or explanation; a hard budget
stop retains an unresolved status without extra calls. A recorded reason is still a model
judgment and does not prove that skipping the evidence was a good choice.

Accepting an affinity throughput improvement requires at least two distinct, compatible
batches with five trials per condition, an improvement of at least 5% in every batch,
nonoverlapping observed timing ranges, and a smallest effect at least half the largest.
Consistent regressions use the same conservative checks in the opposite direction. Small,
conflicting, incomplete or unread comparisons keep the result `inconclusive`. These are
descriptive checks, not statistical significance, equivalence, or proof of a causal bottleneck.

From a source checkout, reassess a saved proposal without API calls or new experiments:

```bash
python scripts/critic_replay.py --trace artifacts/traces/baseline.capture/trace.json \
  --proposal artifacts/agent-report.json \
  --batch artifacts/experiments/batch-1/result.json \
  --batch artifacts/experiments/batch-2/result.json --out artifacts/reassessment.json
```

If a response is cut short, `AGENT_RESPONSE_TRUNCATED` stops the run without executing partial
tool calls. Adjust `META_MAX_COMPLETION_TOKENS` or `META_REASONING_EFFORT` before retrying;
reasoning and visible output share the completion-token budget.
`META_MAX_TOTAL_COMPLETION_TOKENS` limits cumulative reserved output capacity, while
`META_MAX_REQUEST_BYTES` and `META_MAX_SESSION_REQUEST_BYTES` bound serialized input per request
and session. The controller reserves the full per-response output limit before each call,
even if earlier responses used fewer tokens. `model_budget` reports both these reservations
and provider-reported usage; missing usage and dollar cost remain explicitly unknown.

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

Each new trial records UTC `started_at` and `finished_at` times around child execution and
cleanup. These include setup and warm-up; `wall_time_ms` measures only the synchronized
benchmark region. Historical records without trial timestamps remain readable and are labeled
as missing that metadata.

Add `--scenario transfer_bound` or `--scenario compute_bound` to compare placement for a
control workload. Each condition uses the same scenario, seed and operation count; only
the launcher CPU placement changes.

These timings describe the bundled workload. They do not establish the cause of a slowdown
in a different application or measure GPU idle intervals after the intervention.

### Exercise a training or inference model

Use `cnn_inference` or `cnn_training` for a small image classifier with actual convolutional
forward passes; training also runs cross-entropy, backpropagation and SGD. Both use batches
of 32 seeded synthetic RGB images already on the GPU, with fixed warm-up and 200 measured
batches. Output checks verify finite values and whether parameters changed as expected.
These workloads measure model execution under controlled CPU contention, not dataset loading,
model accuracy or production traffic.

```bash
gpu-os-copilot experiment --scenario cnn_training --cpus 2 --repeats 5 --timeout 120
gpu-os-copilot experiment --scenario cnn_training --cpus 2 --repeats 5 --timeout 120 --confirm
```

Choose a treatment CPU from your available affinity set; CPU 2 is only an example. Check
CPU topology because different logical CPU IDs can share a core. To retain variation across
multiple batches for both models, run the study script from this checkout:

```bash
python scripts/model_study.py --out artifacts/model-study --treatment-cpu 2 --batches 3
python scripts/model_study.py --out artifacts/model-study --treatment-cpu 2 --batches 3 --confirm
```

The plan shows resource and time bounds before execution. A new output directory is required;
it retains all trials, per-batch ranges and standard deviations, and the range of batch effects.
An improvement in one environment is not a guarantee for a different application or host.

## Read the results

- **Measurements:** timing summaries computed from the supplied trace and observation window.
  GPU idle fraction refers to captured activity for that workload and device.
- **Candidate explanations:** interpretations of the evidence, with confidence labels and
  missing-data notes. A candidate is not a confirmed cause.
- **Agent decisions:** verified decisions about the recorded affinity throughput effect.
  `inconclusive` means the evidence does not meet the verification policy. Hypotheses remain
  candidate explanations; no decision establishes CPU scheduling as the sole cause.

Use `--json` with `demo-fixture` or `diagnose` for machine-readable output.

The JSON `observability` section reports how many observed CUDA API threads have validated
scheduling coverage, how many GPU activities have unambiguous API links, and why others remain
unlinked. Passing the API/GPU ordering check does not independently calibrate GPU clocks.
Measured traces currently have no independently verified GPU/CPU clock error bound, so
`sched_idle_overlap_ms` and `covered_thread_idle_overlap_bounds` remain unavailable. CPU delay
and GPU activity statistics remain available within their stated scopes.

Where scheduling calibration and ordering checks pass,
`conditional_covered_thread_idle_overlap_bounds` retains the calculation assuming correct
GPU/CPU conversion. These conditional values exclude unknown GPU clock error, unobserved
threads and profiler overhead; they do not bound actual overlap or establish a cause.
API identity links likewise retain a `nominal_launch_to_activity_ns` difference, while verified
`launch_to_activity_ns` is unavailable. Fixture calculations use a defined simulated timebase
and remain explicitly simulated.

Add `--help` to any command to see its options:

```bash
gpu-os-copilot --help
gpu-os-copilot diagnose --help
gpu-os-copilot experiment --help
```
