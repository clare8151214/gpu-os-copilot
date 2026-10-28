"""Render a three-minute captioned replay from actual exported measurements (requires ffmpeg)."""

import argparse
import json
import subprocess
import sys
import textwrap
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("artifacts/demo"))
    parser.add_argument(
        "--font", type=Path, default=Path("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "examples/measured-demo.json").read_text())
    cpu, transfer, compute = data["scenarios"]
    folder = args.out.resolve()
    folder.mkdir(parents=True, exist_ok=False)

    def replay(section):
        return (
            subprocess.check_output(
                [sys.executable, str(root / "scripts/replay_demo.py"), "--section", section],
                text=True,
            )
            .split("\n", 3)[-1]
            .strip()
        )

    workload = (root / "src/gpu_os_copilot/workloads/launch_test.py").read_text()
    start = workload.index('        with torch.cuda.nvtx.range("copilot_measure"):')
    end = workload.index("        if child is not None and not child.is_alive():", start)
    code = textwrap.dedent(workload[start:end]).rstrip()
    control_lines = []
    for row in (transfer, compute):
        comparison = row["comparison"]
        change = comparison["median_throughput_change_fraction"]
        repeat = json.loads(
            (root / f"examples/timestamped-comparisons/{row['scenario']}.json").read_text()
        )["comparison"]["median_throughput_change_fraction"]
        control_lines += [
            row["scenario"],
            f"  Profiled workload idle: {row['profiled_metrics']['gpu_idle_fraction']:.3%}",
            f"  Rules: {', '.join(row['rules'])}",
            f"  Meta: {row['meta']['decision']}; selected experiment: None",
            "  Agent abstained despite having permission to run the test.",
            f"  Manual A/B (5 + 5 each): first {change:+.3%}, repeat {repeat:+.3%}",
            "  Overlapping timing ranges; no meaningful improvement established.",
            "",
        ]
    scenes = [
        (
            25,
            "GPU-OS Copilot",
            "A GPU can wait on the host, even when its kernels are fast.",
            "Read CUDA evidence -> ask Meta to choose a test -> measure the result.\n\n"
            "A command-line tool for CPU/GPU performance debugging.\n\n"
            "This demo replays actual recorded CUDA measurements and Meta tool calls.\n"
            "No new benchmark or API request runs during playback.\n\n"
            "Measured host: RTX 5060 | CUDA 12.8 | PyTorch 2.11.0+cu128 | WSL2\n"
            "Workload: deterministic short CUDA operations, controlled CPU contention.",
        ),
        (
            20,
            "1. Inspect measured evidence",
            "One closed NVTX window; startup and warm-up excluded.",
            "$ python scripts/replay_demo.py --section trace\n\n" + replay("trace") + "\n\n"
            f"GPU busy: {cpu['profiled_metrics']['gpu_busy_ms']:.4f} ms\n"
            "Observed idle applies to this captured workload, not every process on the GPU.\n"
            "CPU runnable-to-running data: unavailable.",
        ),
        (
            20,
            "2. The measurement is implemented",
            "Source: src/gpu_os_copilot/workloads/launch_test.py",
            code + "\n\n"
            "200 operations, fixed inputs, 10 warm-up operations, seed 42.\n"
            "Explicit synchronization measures completed GPU work.\n"
            "Profiler diagnostics and unprofiled timing trials remain separate.",
        ),
        (
            30,
            "3. Meta chooses and uses tools",
            "Actual Muse Spark tool sequence; not the offline fake client.",
            "$ python scripts/replay_demo.py --section agent\n\n" + replay("agent") + "\n\n"
            "run_affinity_ab launched 5 baseline + 5 treatment trials.\n"
            "compare_benchmark_results returned those measured trial statistics.\n"
            "The final decision was made after the tool results were consumed.",
        ),
        (
            25,
            "4. Check the effect again",
            "Only project-owned launch processes change CPU placement.",
            replay("comparison") + "\n\n"
            "Same workload protocol; the repeat's much smaller effect limits the claim.\n"
            "Ranges overlap. A large, stable speedup is not established.\n"
            "These measurements do not prove a sole scheduler cause.",
        ),
        (
            25,
            "5. Check the negative controls",
            "Changing CPU placement should not solve every bottleneck.",
            "\n".join(control_lines),
        ),
        (
            20,
            "6. Bounds and limitations",
            "Measured facts are kept separate from model interpretations.",
            "Tools: validated arguments, fixed workload, one permitted intervention.\n"
            "Timeouts, child cleanup, unique run IDs and retained failures.\n\n"
            "CPU scheduling alignment was unavailable on this host.\n"
            "No post-intervention GPU-idle trace was collected.\n"
            "Five repetitions per condition per batch; effect size varies between batches.\n"
            "Scenarios are labeled; this is not a blinded model evaluation.\n"
            "Rules identify similar symptoms; no AI superiority is claimed.\n"
            "Meta's demonstrated role: tool selection, execution and feedback interpretation.",
        ),
        (
            15,
            "Try the recorded demo",
            "Install, inspect the source, and reproduce the analysis offline.",
            "github.com/clare8151214/gpu-os-copilot\n\n"
            "python3 -m venv .venv\n"
            "source .venv/bin/activate\n"
            "python -m pip install -c requirements-dev.lock -e .\n"
            "python scripts/replay_demo.py\n\n"
            "README: own-trace capture, Meta configuration and bounded experiments.\n"
            "examples/: sanitized traces, all timing trials and recorded tool choices.\n\n"
            "GPU-OS Copilot | Evidence -> experiment -> qualified conclusion",
        ),
    ]
    start_seconds = 0
    with (folder / "render.log").open("x") as log:
        for index, (duration, title, subtitle, body) in enumerate(scenes):
            fields = {"title": title, "subtitle": subtitle, "body": body}
            end_seconds = start_seconds + duration
            fields["footer"] = (
                f"RECORDED MEASURED REPLAY | {start_seconds // 60}:{start_seconds % 60:02d}"
                f" - {end_seconds // 60}:{end_seconds % 60:02d} / 3:00"
            )
            filters = ["drawbox=x=70:y=65:w=90:h=7:color=0x2dd4bf:t=fill"]
            for field, y, size, color in (
                ("title", 100, 48, "white"),
                ("subtitle", 180, 25, "0x94a3b8"),
                ("body", 270, 24, "0xe2e8f0"),
                ("footer", 1000, 22, "0x2dd4bf"),
            ):
                textfile = folder / f"{index}-{field}.txt"
                textfile.write_text(fields[field])
                filters.append(
                    f"drawtext=fontfile='{args.font}':textfile='{textfile}':"
                    f"expansion=none:fontcolor={color}:fontsize={size}:x=70:y={y}:line_spacing=12"
                )
            frame = folder / f"scene-{index}.png"
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=c=0x0f172a:s=1920x1080",
                    "-vf",
                    ",".join(filters),
                    "-frames:v",
                    "1",
                    "-threads",
                    "1",
                    str(frame),
                ],
                check=True,
                stdout=log,
                stderr=log,
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-loop",
                    "1",
                    "-framerate",
                    "10",
                    "-i",
                    str(frame),
                    "-t",
                    str(duration),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "24",
                    "-pix_fmt",
                    "yuv420p",
                    "-threads",
                    "2",
                    str(folder / f"{index}.mp4"),
                ],
                check=True,
                stdout=log,
                stderr=log,
            )
            start_seconds += duration
            print(f"Rendered scene {index + 1}/{len(scenes)}", flush=True)
        playlist = folder / "concat.txt"
        playlist.write_text("".join(f"file '{index}.mp4'\n" for index in range(len(scenes))))
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(playlist),
                "-c",
                "copy",
                "-movflags",
                "+faststart",
                str(folder / "demo.mp4"),
            ],
            check=True,
            stdout=log,
            stderr=log,
        )
    assert start_seconds == 180


if __name__ == "__main__":
    main()
