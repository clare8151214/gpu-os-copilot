"""Compare validated before/after trace JSON; no profiling or privileged execution."""

import argparse
import json
from pathlib import Path

from gpu_os_copilot.analysis.compare_traces import compare_profiled_traces
from gpu_os_copilot.schemas import TraceBundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    traces = []
    for path in (args.baseline, args.treatment):
        if path.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("Trace input exceeds 64 MiB")
        traces.append(TraceBundle.model_validate_json(path.read_text()))
    result = compare_profiled_traces(*traces)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
