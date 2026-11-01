"""Preregistered labeling rule (new-bottleneck-protocol-v1); thresholds are fixed in advance."""

FAMILY_CATEGORY = {
    "sync_device": "synchronization",
    "sync_readback": "synchronization",
    "sync_periodic": "synchronization",
    "launch_burst": "launch_overhead",
    "overlap_copy": "data_transfer",
    "serial_copy": "data_transfer",
}
VERIFIED_GAIN = 0.10
RULED_OUT_GAIN = 0.03
DOMINANT_GAIN = 1.0


def ground_truth(family: str, batch_gains: list[float], kernel_busy_fraction: float | None):
    category = FAMILY_CATEGORY[family]
    result = {
        "version": "ablation-label-v1",
        "factor_category": category,
        "batch_gains": batch_gains,
        "kernel_busy_fraction": kernel_busy_fraction,
    }
    if len(batch_gains) < 3:
        return {**result, "status": "excluded", "reason": "FEWER_THAN_THREE_BATCHES"}
    if all(gain >= VERIFIED_GAIN for gain in batch_gains):
        accepted = {category}
        if min(batch_gains) < DOMINANT_GAIN:
            if kernel_busy_fraction is None:
                return {**result, "status": "excluded", "reason": "KERNEL_BUSY_UNAVAILABLE"}
            if kernel_busy_fraction >= 0.5:
                accepted.add("gpu_compute")
        return {
            **result,
            "status": "verified",
            "accepted_top_categories": sorted(accepted),
            "factor_in_top2_required": True,
        }
    if all(abs(gain) <= RULED_OUT_GAIN for gain in batch_gains):
        if kernel_busy_fraction is not None and kernel_busy_fraction >= 0.8:
            return {
                **result,
                "status": "ruled_out",
                "accepted_top_categories": ["gpu_compute"],
                "ruled_out_category_not_first": category,
            }
        return {**result, "status": "excluded", "reason": "RULED_OUT_WITHOUT_DENSE_KERNELS"}
    return {**result, "status": "excluded", "reason": "AMBIGUOUS_ABLATION"}


FACTOR_CATEGORY = {"sync": "synchronization", "launch": "launch_overhead", "copy": "data_transfer"}
CO_DOMINANT_RATIO = 1.5


def ground_truth_mixed(factor_gains: dict[str, list[float]], kernel_busy_fraction: float | None):
    """Protocol amendment 1: the factor whose single removal recovers most time is the label."""
    result = {
        "version": "ablation-label-mixed-v1",
        "factor_gains": factor_gains,
        "kernel_busy_fraction": kernel_busy_fraction,
    }
    if not factor_gains or any(len(gains) < 3 for gains in factor_gains.values()):
        return {**result, "status": "excluded", "reason": "FEWER_THAN_THREE_BATCHES"}
    verified = {f: min(g) for f, g in factor_gains.items() if all(x >= VERIFIED_GAIN for x in g)}
    ruled_out = {f for f, g in factor_gains.items() if all(abs(x) <= RULED_OUT_GAIN for x in g)}
    if not verified:
        if ruled_out == set(factor_gains) and (kernel_busy_fraction or 0) >= 0.8:
            return {**result, "status": "ruled_out", "accepted_top_categories": ["gpu_compute"]}
        return {**result, "status": "excluded", "reason": "NO_VERIFIED_FACTOR"}
    best = max(verified.values())
    accepted = {
        FACTOR_CATEGORY[f] for f, gain in verified.items() if gain * CO_DOMINANT_RATIO >= best
    }
    if best < DOMINANT_GAIN:
        if kernel_busy_fraction is None:
            return {**result, "status": "excluded", "reason": "KERNEL_BUSY_UNAVAILABLE"}
        if kernel_busy_fraction >= 0.5:
            accepted.add("gpu_compute")
    return {
        **result,
        "status": "verified",
        "accepted_top_categories": sorted(accepted),
        "verified_factors": sorted(verified),
        "ruled_out_factors": sorted(ruled_out),
    }
