"""Classify CUDA runtime/driver API names; host-side single-clock facts only."""

import re

API_CLASSES = ("launch", "sync", "memcpy", "memset", "alloc", "other")

_VERSION_SUFFIX = re.compile(r"_v\d+$")
_SAFE_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]{0,95}")

# Explicit host-blocking waits. Stream/event waits enqueued on the device are not host waits.
_SYNC = {
    "cudaDeviceSynchronize",
    "cudaStreamSynchronize",
    "cudaEventSynchronize",
    "cuCtxSynchronize",
    "cuStreamSynchronize",
    "cuEventSynchronize",
}
_PREFIXES = (
    ("launch", ("cudaLaunch", "cuLaunch", "cudaGraphLaunch", "cuGraphLaunch")),
    ("memcpy", ("cudaMemcpy", "cuMemcpy")),
    ("memset", ("cudaMemset", "cuMemset", "cuMemsetD")),
    (
        "alloc",
        (
            "cudaMalloc",
            "cudaFree",
            "cudaHostAlloc",
            "cudaHostRegister",
            "cuMemAlloc",
            "cuMemFree",
            "cuMemHostAlloc",
        ),
    ),
)


def normalize_api_name(raw) -> str | None:
    """Strip the CUPTI version suffix; reject anything that is not a plain identifier."""
    if not isinstance(raw, str):
        return None
    name = _VERSION_SUFFIX.sub("", raw)
    return name if _SAFE_NAME.fullmatch(name) else None


def classify_api(name: str) -> str:
    if name in _SYNC:
        return "sync"
    for api_class, prefixes in _PREFIXES:
        if name.startswith(prefixes):
            return api_class
    return "other"
