"""Reassess a saved model proposal using recorded batches without any API/GPU calls."""

import argparse
import hashlib
import json
from pathlib import Path

from gpu_os_copilot.agent.critic import verify_decision
from gpu_os_copilot.schemas import AgentDecision, ExperimentResult, TraceBundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--batch", type=Path, action="append", required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= len(args.batch) <= 8:
        parser.error("Provide 1-8 recorded batches")
    trace = TraceBundle.model_validate_json(args.trace.read_text())
    results = [ExperimentResult.model_validate_json(path.read_text()) for path in args.batch]
    saved = json.loads(args.proposal.read_text())
    if "agent" in saved:
        saved = saved["agent"]
    if isinstance(saved.get("decision"), dict):
        saved = saved["decision"]
    proposed = AgentDecision.model_validate(saved)
    effective, verification = verify_decision(proposed, trace, results)
    report = {
        "schema_version": "1.0",
        "mode": "offline_critic_replay",
        "new_model_calls": 0,
        "new_hardware_experiments": 0,
        "proposed_decision": proposed.model_dump(),
        "decision": effective.model_dump(),
        "verification": verification,
        "sources_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [args.trace, args.proposal, *args.batch]
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as output:
        output.write(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"decision": effective.decision, "reason_codes": verification["reason_codes"]})
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
