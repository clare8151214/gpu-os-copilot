import argparse

from gpu_os_copilot.cli import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-paid-api", action="store_true", required=True)
    parser.parse_args()
    raise SystemExit(
        main(
            [
                "demo-fixture",
                "--scenario",
                "cpu_starvation",
                "--mode",
                "meta",
                "--allow-paid-api",
            ]
        )
    )
