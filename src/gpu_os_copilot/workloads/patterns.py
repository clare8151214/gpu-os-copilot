"""Fixed bottleneck-pattern workloads with ablation twins that remove exactly one factor."""

FAMILIES = {
    "sync_device": "remove per-operation torch.cuda.synchronize()",
    "sync_readback": "remove per-operation .item() readback",
    "sync_periodic": "remove periodic torch.cuda.synchronize()",
    "launch_burst": "replay the identical operation as one CUDA graph",
    "overlap_copy": "remove side-stream pinned H2D copies",
    "serial_copy": "remove same-stream pinned H2D copies",
    "mixed": "remove one named factor (sync, launch or copy) of a seeded multi-factor draw",
}
MIXED_FACTORS = {
    "sync": "remove periodic torch.cuda.synchronize()",
    "launch": "replay the kernels of each operation as one CUDA graph",
    "copy": "remove pinned H2D copies",
}

# Mirrors the preregistered protocol (new-bottleneck-protocol-v1); do not tune after holdout.
PATTERN_VARIANTS = {
    "sync_device_a": ("sync_device", {"shape": (512, 512)}),
    "sync_device_b": ("sync_device", {"shape": (768, 768)}),
    "sync_device_c": ("sync_device", {"shape": (1024, 1024)}),
    "sync_device_d": ("sync_device", {"shape": (1536, 1536)}),
    "sync_readback_a": ("sync_readback", {"shape": (512, 512)}),
    "sync_readback_b": ("sync_readback", {"shape": (768, 768)}),
    "sync_readback_c": ("sync_readback", {"shape": (1024, 1024)}),
    "sync_readback_d": ("sync_readback", {"shape": (1536, 1536)}),
    "sync_periodic_a": ("sync_periodic", {"shape": (768, 768), "every": 1}),
    "sync_periodic_b": ("sync_periodic", {"shape": (896, 896), "every": 1}),
    "sync_periodic_c": ("sync_periodic", {"shape": (1024, 1024), "every": 1}),
    "sync_periodic_d": ("sync_periodic", {"shape": (1024, 1024), "every": 2}),
    "launch_burst_a": ("launch_burst", {"shape": (640, 640), "burst": 24}),
    "launch_burst_b": ("launch_burst", {"shape": (768, 768), "burst": 16}),
    "launch_burst_c": ("launch_burst", {"shape": (768, 768), "burst": 32}),
    "launch_burst_d": ("launch_burst", {"shape": (1024, 1024), "burst": 32}),
    "overlap_copy_a": ("overlap_copy", {"shape": (1792, 1792), "copy_mb": 8}),
    "overlap_copy_b": ("overlap_copy", {"shape": (2048, 2048), "copy_mb": 10}),
    "overlap_copy_c": ("overlap_copy", {"shape": (2048, 2048), "copy_mb": 12}),
    "overlap_copy_d": ("overlap_copy", {"shape": (2304, 2304), "copy_mb": 12}),
    "serial_copy_a": ("serial_copy", {"shape": (1792, 1792), "copy_mb": 8}),
    "serial_copy_b": ("serial_copy", {"shape": (2048, 2048), "copy_mb": 10}),
    "serial_copy_c": ("serial_copy", {"shape": (2048, 2048), "copy_mb": 12}),
    "serial_copy_d": ("serial_copy", {"shape": (2304, 2304), "copy_mb": 12}),
    # Holdout-only draws (protocol amendment 1, seed 20261103); never collected for dev.
    **{
        f"mixed_r{i}": (
            "mixed",
            {"shape": (n, n), "sync_every": e, "burst": b, "copy_mb": c, "copy_mode": m},
        )
        for i, (n, e, b, c, m) in enumerate(
            [
                (768, 0, 24, 2, "overlap"),
                (768, 4, 0, 2, "overlap"),
                (1024, 2, 8, 8, "serial"),
                (1536, 1, 0, 8, "overlap"),
                (768, 0, 8, 2, "overlap"),
                (1536, 1, 24, 0, "none"),
                (512, 4, 0, 8, "overlap"),
                (1024, 1, 0, 8, "serial"),
            ],
            1,
        )
    },
}


def mixed_factors(params: dict) -> list[str]:
    present = {"sync": params["sync_every"] > 0, "launch": params["burst"] > 0}
    present["copy"] = params["copy_mb"] > 0
    return [name for name in MIXED_FACTORS if present[name]]


def check_factor(scenario: str, ablated: bool, factor: str | None) -> None:
    family, params = PATTERN_VARIANTS[scenario]
    if family != "mixed":
        if factor is not None:
            raise ValueError("Only mixed scenarios take an ablation factor")
    elif ablated != (factor is not None) or factor not in {None, *mixed_factors(params)}:
        raise ValueError("Mixed ablation requires one factor present in the draw")


def pattern_configuration(scenario: str, ablated: bool, factor: str | None = None) -> dict:
    check_factor(scenario, ablated, factor)
    family, params = PATTERN_VARIANTS[scenario]
    config = {
        "pattern_family": family,
        "pattern_parameters": {
            key: list(value) if isinstance(value, tuple) else value for key, value in params.items()
        },
        "ablated": ablated,
        "ablation": (MIXED_FACTORS[factor] if factor else FAMILIES[family]) if ablated else "none",
    }
    if family == "mixed":
        config["ablated_factor"] = factor
    return config


def _graph(torch, eager):
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):
            eager()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        eager()
    return graph.replay


def _mixed(torch, params, factor):
    x = torch.ones(params["shape"], device="cuda")
    y, z = torch.ones_like(x), torch.empty_like(x)
    tiny = torch.ones((64, 64), device="cuda")

    def eager_kernels():
        torch.mm(x, y, out=z)
        for _ in range(params["burst"]):
            tiny.add_(0.001)

    kernels = _graph(torch, eager_kernels) if factor == "launch" else eager_kernels
    copying = params["copy_mb"] > 0 and factor != "copy"
    elements = max(1, params["copy_mb"]) * 1024 * 1024 // 4
    host = torch.ones(elements, pin_memory=True) if copying else None
    device = torch.empty(elements, device="cuda") if copying else None
    side = torch.cuda.Stream() if copying and params["copy_mode"] == "overlap" else None
    syncing = params["sync_every"] > 0 and factor != "sync"
    count = [0]

    def operation():
        if copying:
            if side is None:
                device.copy_(host, non_blocking=True)
            else:
                with torch.cuda.stream(side):
                    device.copy_(host, non_blocking=True)
        kernels()
        if side is not None:
            torch.cuda.current_stream().wait_stream(side)
        count[0] += 1
        if syncing and count[0] % params["sync_every"] == 0:
            torch.cuda.synchronize()

    return operation


def build_pattern(torch, scenario: str, ablated: bool, factor: str | None = None):
    """Return one operation; tensors and CUDA graphs are created before measurement."""
    check_factor(scenario, ablated, factor)
    family, params = PATTERN_VARIANTS[scenario]
    if family == "mixed":
        return _mixed(torch, params, factor)
    x = torch.ones(params["shape"], device="cuda")
    if family in {"sync_device", "sync_readback"}:

        def operation():
            x.add_(0.001)
            if ablated:
                return
            if family == "sync_device":
                torch.cuda.synchronize()
            else:
                x[0, 0].item()

        return operation
    y, z = torch.ones_like(x), torch.empty_like(x)
    if family == "sync_periodic":
        count = [0]

        def operation():
            torch.mm(x, y, out=z)
            count[0] += 1
            if not ablated and count[0] % params["every"] == 0:
                torch.cuda.synchronize()

        return operation
    if family == "launch_burst":
        tiny = torch.ones((64, 64), device="cuda")

        def eager():
            torch.mm(x, y, out=z)
            for _ in range(params["burst"]):
                tiny.add_(0.001)

        return _graph(torch, eager) if ablated else eager
    elements = params["copy_mb"] * 1024 * 1024 // 4
    host = torch.ones(elements, pin_memory=True)
    device = torch.empty(elements, device="cuda")
    side = torch.cuda.Stream() if family == "overlap_copy" else None

    def operation():
        if not ablated:
            if side is None:
                device.copy_(host, non_blocking=True)
            else:
                with torch.cuda.stream(side):
                    device.copy_(host, non_blocking=True)
        torch.mm(x, y, out=z)
        if side is not None and not ablated:
            torch.cuda.current_stream().wait_stream(side)

    return operation
