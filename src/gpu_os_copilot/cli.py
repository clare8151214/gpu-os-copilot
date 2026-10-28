"""Offline-first CLI. Hardware execution and paid API calls require distinct opt-ins."""

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from gpu_os_copilot.analysis.metrics import summarize
from gpu_os_copilot.analysis.rule_baseline import diagnose
from gpu_os_copilot.config import AgentConfig, CopilotError
from gpu_os_copilot.reports.render import render
from gpu_os_copilot.schemas import (
    CaptureRequest,
    ExperimentRequest,
    Provenance,
    TraceBundle,
    TraceRequest,
)
from gpu_os_copilot.worker.local import LocalWorker


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Read-only environment and capability report")
    demo = commands.add_parser("demo-fixture", help="Run a clearly simulated offline trace")
    demo.add_argument(
        "--scenario",
        choices=["cpu_starvation", "transfer_bound", "compute_bound"],
        default="cpu_starvation",
    )
    diagnosis = commands.add_parser("diagnose", help="Analyze canonical JSON or Nsight SQLite")
    diagnosis.add_argument("--trace", type=Path, required=True)
    diagnosis.add_argument("--start-ns", type=int, help="Required for SQLite; excludes warmup")
    diagnosis.add_argument("--end-ns", type=int, help="Required for SQLite; excludes shutdown")
    diagnosis.add_argument("--device-id", type=int, default=0)
    diagnosis.add_argument(
        "--capture-complete",
        action="store_true",
        help="Owner asserts the SQLite capture covers the entire selected window",
    )
    diagnosis.add_argument(
        "--simulated", action="store_true", help="Label synthetic SQLite as simulated"
    )
    diagnosis.add_argument(
        "--confirm",
        action="store_true",
        help="Allow bounded project-child experiments on real input in Meta mode",
    )
    for command in (demo, diagnosis):
        command.add_argument("--json", action="store_true", help="Emit structured evidence")
        command.add_argument("--mode", choices=["rules", "mock", "meta"], default="rules")
        command.add_argument("--allow-paid-api", action="store_true")
        command.add_argument("--max-steps", type=int, default=None)
        command.add_argument("--audit-dir", type=Path, default=Path("artifacts/agent"))
    trace = commands.add_parser("trace", help="Plan or capture the bounded bundled workload")
    trace.add_argument(
        "--scenario",
        choices=["cpu_contention", "transfer_bound", "compute_bound"],
        default="cpu_contention",
    )
    trace.add_argument("--out", type=Path, required=True)
    trace.add_argument("--cpu-context-switches", action="store_true")
    trace.add_argument("--cuda-trace-mode", choices=["cuda", "cuda-sw"], default="cuda-sw")
    trace.add_argument(
        "--cpus", type=int, nargs=1, help="Launcher CPU; defaults to first available"
    )
    trace.add_argument("--timeout", type=int, default=60)
    trace.add_argument("--confirm", action="store_true", help="Run bounded profiling and export")
    experiment = commands.add_parser(
        "experiment", help="Plan or explicitly execute bounded affinity A/B"
    )
    experiment.add_argument("--kind", choices=["affinity_ab"], default="affinity_ab")
    experiment.add_argument(
        "--scenario",
        choices=["cpu_contention", "transfer_bound", "compute_bound"],
        default="cpu_contention",
    )
    experiment.add_argument(
        "--cpus", type=int, nargs="+", help="Treatment CPU IDs; defaults to second CPU"
    )
    experiment.add_argument("--repeats", type=int, default=5)
    experiment.add_argument("--timeout", type=int, default=60)
    experiment.add_argument("--confirm", action="store_true")
    experiment.add_argument("--experiment-id", default=None)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    worker = LocalWorker()
    try:
        if args.command == "doctor":
            print(worker.capabilities().model_dump_json(indent=2))
            return 0
        if args.command == "trace":
            if args.confirm:
                cpus = worker.capabilities().cpu_affinity
                request = CaptureRequest(
                    experiment_id=f"capture-{uuid4().hex[:16]}",
                    scenario=args.scenario,
                    cpu_set=args.cpus or cpus[:1],
                    contender_cpu=cpus[0],
                    timeout_s=args.timeout,
                    cuda_trace_mode=args.cuda_trace_mode,
                    cpu_context_switches=args.cpu_context_switches,
                    owner_approved=True,
                    provenance=Provenance(
                        source="nsys_sqlite",
                        confidence="derived",
                        description="Owner-approved bounded profiler request",
                    ),
                )
                result = worker.capture_trace(request, args.out)
                output = result.model_dump(mode="json", exclude={"trace"})
                if result.trace is not None:
                    output["trace_path"] = str(
                        args.out.resolve().with_name(args.out.name + ".capture") / "trace.json"
                    )
                    output["gpu_idle_fraction"] = summarize(result.trace).gpu_idle_fraction
                    output["event_count"] = len(result.trace.events)
                print(json.dumps(output, indent=2))
                return 0 if result.status == "ok" else 2
            print(
                json.dumps(
                    worker.plan_trace(
                        args.scenario,
                        args.out,
                        args.cpu_context_switches,
                        args.cuda_trace_mode,
                    ),
                    indent=2,
                )
            )
            return 0
        if args.command == "experiment":
            cpus = args.cpus if args.cpus is not None else worker.capabilities().cpu_affinity[1:2]
            result = worker.run_experiment(
                ExperimentRequest(
                    experiment_id=args.experiment_id or f"manual-{uuid4().hex[:16]}",
                    workload_id="launch_test",
                    scenario=args.scenario,
                    candidate_cpu_set=cpus,
                    repeats=args.repeats,
                    timeout_s=args.timeout,
                    owner_approved=args.confirm,
                    provenance=Provenance(
                        source="benchmark",
                        confidence="derived",
                        description="Owner CLI experiment request",
                    ),
                )
            )
            print(result.model_dump_json(indent=2))
            return 0 if result.status in {"dry_run", "ok"} else 2
        if args.command == "demo-fixture":
            trace = worker.collect_trace(
                TraceRequest(
                    experiment_id=f"fixture-{args.scenario}",
                    workload_id=args.scenario,
                    scenario=args.scenario,
                    provenance=Provenance(
                        source="fixture",
                        confidence="simulated",
                        description="Offline fixture request",
                    ),
                )
            ).trace
        elif args.trace.suffix.lower() == ".json":
            if args.trace.stat().st_size > 64 * 1024 * 1024:
                raise CopilotError("TRACE_INPUT_TOO_LARGE: canonical JSON limit is 64 MiB")
            trace = TraceBundle.model_validate_json(args.trace.read_text())
        else:
            if args.start_ns is None or args.end_ns is None:
                raise CopilotError("SQLITE_WINDOW_REQUIRED: provide --start-ns and --end-ns")
            trace = worker.import_sqlite(
                args.trace,
                start_ns=args.start_ns,
                end_ns=args.end_ns,
                device_id=args.device_id,
                capture_complete=args.capture_complete,
                simulated=args.simulated,
            )
        summary = summarize(trace)
        hypotheses = diagnose(trace, summary)
        output = {
            "summary": summary.model_dump(mode="json"),
            "hypotheses": [h.model_dump() for h in hypotheses],
        }
        if args.mode != "rules":
            from gpu_os_copilot.agent.fake_client import FakeClient
            from gpu_os_copilot.agent.meta_client import MetaClient
            from gpu_os_copilot.agent.runner import run_agent
            from gpu_os_copilot.agent.tools import ToolExecutor

            config = AgentConfig.from_environment()
            if args.max_steps is not None:
                config.max_steps = args.max_steps
            client = (
                FakeClient(trace)
                if args.mode == "mock"
                else MetaClient(config, allow_paid_api=args.allow_paid_api)
            )
            executor = ToolExecutor(
                worker,
                trace,
                owner_approved=getattr(args, "confirm", False)
                and args.mode == "meta"
                and trace.provenance.confidence != "simulated",
                max_experiments=config.max_experiments,
            )
            config.dry_run = not executor.owner_approved
            output["agent"] = run_agent(client, executor, config, args.audit_dir)
        if args.json:
            print(json.dumps(output, indent=2))
        else:
            print(render(summary, hypotheses))
            if "agent" in output:
                print(json.dumps(output["agent"], indent=2))
        return 0
    except CopilotError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    except (ValidationError, ValueError, OSError, RuntimeError, sqlite3.Error) as exc:
        # Never expose environment values or full validation inputs in CLI errors.
        print(
            f"Error: {type(exc).__name__}. Check inputs and documented prerequisites.",
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
