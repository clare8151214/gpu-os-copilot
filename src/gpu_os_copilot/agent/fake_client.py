"""Scripted protocol replay, explicitly NOT a Meta model or evidence of AI reasoning."""

import json

from gpu_os_copilot.analysis.metrics import summarize
from gpu_os_copilot.analysis.rule_baseline import diagnose


def tool_call(identifier, name, arguments):
    return {
        "id": identifier,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


class FakeClient:
    mode = "mock"

    def __init__(self, trace):
        self.trace = trace
        self.turn = 0

    def complete(self, messages, tools):
        self.turn += 1
        hypotheses = diagnose(self.trace, summarize(self.trace))
        if self.turn == 1:
            calls = [
                tool_call("summary", "get_performance_summary", {"run_id": self.trace.run_id}),
                tool_call("available", "list_available_experiments", {}),
            ]
        elif self.turn == 2:
            calls = [
                tool_call(
                    "timeline",
                    "inspect_timeline",
                    {
                        "run_id": self.trace.run_id,
                        "event_types": ["gpu_kernel", "memcpy", "cpu_sched"],
                        "limit": 20,
                    },
                )
            ]
            available = (
                next(
                    json.loads(m["content"])
                    for m in messages
                    if m.get("tool_call_id") == "available"
                )
                .get("result", {})
                .get("experiments", [])
            )
            if hypotheses[0].category == "cpu_scheduling" and available:
                calls.append(
                    tool_call(
                        "plan",
                        "run_affinity_ab",
                        {
                            "workload_id": "launch_test",
                            "candidate_cpu_set": available[0]["candidate_cpu_sets"][0],
                            "repeats": 5,
                        },
                    )
                )
        else:
            plan = [m for m in messages if m.get("tool_call_id") == "plan"]
            status = (
                json.loads(plan[0]["content"]).get("result", {}).get("status") if plan else None
            )
            return {
                "message": {
                    "role": "assistant",
                    "content": json.dumps(
                        {
                            "hypotheses": [h.model_dump() for h in hypotheses],
                            "selected_experiment": {
                                "name": "affinity_ab",
                                "justification": "Scripted offline protocol exercise",
                            }
                            if plan
                            else None,
                            "decision": "inconclusive",
                            "rationale": (
                                f"MOCK replay consumed tool result status={status}. "
                                "No real Meta inference or measured speedup is established."
                            ),
                        }
                    ),
                },
                "usage": None,
                "response_id": "mock-final",
            }
        return {
            "message": {"role": "assistant", "content": None, "tool_calls": calls},
            "usage": None,
            "response_id": f"mock-{self.turn}",
        }
