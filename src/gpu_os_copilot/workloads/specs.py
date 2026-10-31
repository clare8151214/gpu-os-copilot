"""Fixed, reviewable workloads; no downloaded code, datasets or executable parameters."""

WORKLOAD_VERSION = 2
SCENARIO_SHAPES = {
    "cpu_contention": (256, 256),
    "transfer_bound": (2048, 2048),
    "compute_bound": (2048, 2048),
    "cnn_inference": (32, 3, 32, 32),
    "cnn_training": (32, 3, 32, 32),
}
MODEL_SCENARIOS = frozenset({"cnn_inference", "cnn_training"})


def expected_configuration(scenario: str) -> dict:
    config = {
        "scenario": scenario,
        "workload_version": WORKLOAD_VERSION,
        "seed": 42,
        "shape": list(SCENARIO_SHAPES[scenario]),
        "dtype": "float32",
        "warmup": 10,
        "operations": 200,
        "synchronized": True,
    }
    if scenario in MODEL_SCENARIOS:
        config.update(
            model="conv3x3-3x16-relu-pool2-conv3x3-16x32-relu-gap-linear32x10-v1",
            mode="training" if scenario == "cnn_training" else "inference",
            batch_size=32,
            data="seeded synthetic 32x32 RGB images; resident on GPU",
            operation_unit="batch",
            optimizer="SGD(lr=0.01)" if scenario == "cnn_training" else "none",
            loss="cross_entropy" if scenario == "cnn_training" else "none",
            cudnn_benchmark=False,
            cudnn_deterministic=True,
        )
    return config
