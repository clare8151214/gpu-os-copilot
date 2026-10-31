"""Run one frozen case in a subprocess whose PYTHONPATH selects the agent version."""

import argparse
import hashlib
import json
import socket
from pathlib import Path

from evaluation_common import LIMITS, GuardedClient, ReplayExecutor, digest

from gpu_os_copilot.agent.meta_client import MetaClient
from gpu_os_copilot.agent.runner import run_agent
from gpu_os_copilot.analysis.rule_baseline import diagnose
from gpu_os_copilot.config import AgentConfig, CopilotError
from gpu_os_copilot.schemas import AgentDecision, Hypothesis


class OfflineClient:
    mode = "mock"

    def __init__(self, config, run_id):
        self.config, self.run_id, self.turn = config, run_id, 0

    def complete(self, messages, tools):
        self.turn += 1
        names = (
            ["read_recorded_comparison", "read_repeated_comparisons"]
            if self.turn == 1
            else ["inspect_timeline"]
        )
        if self.turn == 1 and any(
            tool["function"]["name"] == "read_profiled_scheduler_comparisons" for tool in tools
        ):
            names.append("read_profiled_scheduler_comparisons")
        if tools and self.turn <= 2:
            return {
                "message": {
                    "tool_calls": [
                        {
                            "id": f"offline-{self.turn}-{i}",
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": json.dumps(
                                    {
                                        "run_id": self.run_id,
                                        "event_types": ["gpu_kernel", "cuda_api"],
                                        "limit": 10,
                                    }
                                    if name == "inspect_timeline"
                                    else {}
                                ),
                            },
                        }
                        for i, name in enumerate(names)
                    ]
                },
                "usage": None,
                "response_id": f"offline-{self.turn}",
            }
        fields = {
            "category": "unknown",
            "statement": "Scripted software check only.",
            "missing_evidence": ["Live model evaluation"],
            "counterevidence_note": "No model quality claim is made by this scripted check.",
            "testable_prediction": "Live inference may choose different actions.",
        }
        hypothesis = Hypothesis(
            **{key: value for key, value in fields.items() if key in Hypothesis.model_fields}
        )
        decision = AgentDecision(
            hypotheses=[hypothesis],
            decision="inconclusive",
            rationale="Offline protocol validation; not a Meta evaluation.",
        )
        return {
            "message": {"content": decision.model_dump_json()},
            "usage": None,
            "response_id": "offline-final",
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--method", choices=["rules", "legacy", "enhanced"], required=True)
    parser.add_argument("--runner-sha256", required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--allow-paid-api", action="store_true")
    parser.add_argument("--inspect-input", action="store_true", help="Offline input hashes only")
    args = parser.parse_args()
    if args.offline:

        def reject_network(*unused_args, **unused_kwargs):
            raise CopilotError("OFFLINE_NETWORK_DISABLED")

        socket.socket.connect = reject_network
    if args.offline and args.allow_paid_api:
        parser.error("Offline and paid modes are mutually exclusive")
    if args.inspect_input and not args.offline:
        parser.error("Input inspection requires --offline")
    if args.method != "rules" and not args.offline and not args.allow_paid_api:
        parser.error("Explicit API opt-in required")
    runner_path = Path(run_agent.__code__.co_filename).resolve()
    if hashlib.sha256(runner_path.read_bytes()).hexdigest() != args.runner_sha256:
        raise ValueError("Agent runner does not match frozen source")
    case = json.loads(args.case.read_text())
    ex = ReplayExecutor(case, case_directory=args.case.parent)
    if args.inspect_input:
        print(
            json.dumps(
                {
                    "opening_sha256": digest(ex.evidence()),
                    "common_tools_sha256": digest(ex.tool_schemas()),
                }
            )
        )
        return 0
    if args.out is None:
        parser.error("Execution requires --out")
    args.out.mkdir(parents=True, exist_ok=False, mode=0o700)
    result = {
        "method": args.method,
        "case_id": ex.trace.run_id,
        "status": "ok",
        "evaluation_mode": "offline_protocol"
        if args.offline
        else "rules"
        if args.method == "rules"
        else "meta",
        "runner_sha256": args.runner_sha256,
        "opening_sha256": digest(ex.evidence()),
        "common_tools_sha256": digest(ex.tool_schemas()),
        "new_hardware_experiments": 0,
    }
    if args.method == "rules":
        hypotheses = diagnose(ex.trace, ex.summary_for_rules())
        selects = hypotheses[0].category in {"launch_overhead", "cpu_scheduling"}
        if selects:
            ex.execute("read_recorded_comparison", "{}")
            ex.execute("read_repeated_comparisons", "{}")
            if ex.scheduler is not None:
                ex.execute("read_profiled_scheduler_comparisons", "{}")
        result["agent"] = {
            "mode": "rules",
            "decision": {
                "hypotheses": [h.model_dump() for h in hypotheses],
                "decision": "inconclusive",
                "selected_experiment": {"name": "affinity_ab"} if selects else None,
            },
        }
        if ex.scheduler is not None:
            result["agent"]["scheduling_evidence"] = ex.scheduling_evidence()
            result["rules_extension"] = (
                "Historical symptom rule plus one recorded scheduling read on affinity selection; "
                "deterministic study assessment is reported without a causal interpretation."
            )
        result["usage"] = {
            "requests": 0,
            "reported_total_tokens": 0,
            "usage_complete": True,
            "cost_usd": 0,
        }
    else:
        config = AgentConfig(**{k: v for k, v in LIMITS.items() if k in AgentConfig.model_fields})
        client = None
        try:
            delegate = (
                OfflineClient(config, ex.trace.run_id)
                if args.offline
                else MetaClient(config, allow_paid_api=True)
            )
            client = GuardedClient(
                delegate, ex.evidence(), ex.tool_schemas(), args.out / "usage-checkpoint.json"
            )
            result["agent"] = run_agent(client, ex, config, args.out / "audit")
        except CopilotError as exc:
            result.update(status="failed", error_code=str(exc).split(":")[0])
        result["usage"] = client.report() if client else {"requests": 0, "usage_complete": False}
    result["comparison_reads"] = ex.reads
    result["scheduler_reads"] = ex.scheduler_reads
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps({"case_id": ex.trace.run_id, "method": args.method, "status": result["status"]})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
