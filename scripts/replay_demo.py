"""Read recorded measurements and actual Meta choices without GPU work or API requests."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "examples/measured-demo.json",
    )
    parser.add_argument(
        "--section",
        choices=["all", "trace", "agent", "comparison", "controls", "limits"],
        default="all",
    )
    args = parser.parse_args()
    data = json.loads(args.data.read_text())
    if data["kind"] != "recorded_measured_demo":
        raise ValueError("Expected recorded measured evidence")
    print("GPU-OS Copilot | RECORDED MEASURED RUN REPLAY")
    print("No new GPU workload or model request is executed.\n")
    for row in data["scenarios"]:
        if args.section in {"trace", "agent", "comparison"} and row["scenario"] != "cpu_contention":
            continue
        if args.section == "controls" and row["scenario"] == "cpu_contention":
            continue
        if args.section == "limits":
            break
        print(row["scenario"])
        if args.section in {"all", "trace", "controls"}:
            print(f"  Profiled workload idle: {row['profiled_metrics']['gpu_idle_fraction']:.3%}")
            print("  Rule candidates: " + ", ".join(row["rules"]))
        if args.section in {"all", "agent", "controls"}:
            print(f"  Real Meta model: {row['meta']['model']}")
            for index, name in enumerate(row["meta"]["tools"], start=1):
                print(f"    {index}. {name}")
            print(f"  Model decision: {row['meta']['decision']}")
            print(f"  Selected experiment: {row['meta']['selected_experiment']}")
        if args.section in {"all", "comparison", "controls"}:
            comparison = row["comparison"]
            print(f"  Unprofiled A/B origin: {row['experiment_origin']}")
            for condition in ("baseline", "treatment"):
                stats = comparison[condition]
                print(
                    f"  {condition:9} n={stats['n']} median={stats['median_wall_time_ms']:.6f} ms "
                    f"range={stats['min_wall_time_ms']:.6f}-{stats['max_wall_time_ms']:.6f}"
                )
            change = comparison["median_throughput_change_fraction"]
            print(f"  Median throughput change: {change:+.3%} (descriptive)")
        print()
    if args.section in {"all", "limits"}:
        for limitation in data["limitations"]:
            print(f"- {limitation}")


if __name__ == "__main__":
    main()
