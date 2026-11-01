import pytest

from gpu_os_copilot.analysis.api_classes import classify_api, normalize_api_name


@pytest.mark.parametrize(
    "raw,name,api_class",
    [
        ("cudaLaunchKernel_v7000", "cudaLaunchKernel", "launch"),
        ("cudaLaunchKernelExC_v11060", "cudaLaunchKernelExC", "launch"),
        ("cuLaunchKernel", "cuLaunchKernel", "launch"),
        ("cudaGraphLaunch_v10000", "cudaGraphLaunch", "launch"),
        ("cudaDeviceSynchronize_v3020", "cudaDeviceSynchronize", "sync"),
        ("cudaStreamSynchronize_v3020", "cudaStreamSynchronize", "sync"),
        ("cudaEventSynchronize_v3020", "cudaEventSynchronize", "sync"),
        ("cudaStreamWaitEvent_v3020", "cudaStreamWaitEvent", "other"),  # Device-side wait.
        ("cudaMemcpyAsync_v3020", "cudaMemcpyAsync", "memcpy"),
        ("cudaMemsetAsync_v3020", "cudaMemsetAsync", "memset"),
        ("cudaMalloc_v3020", "cudaMalloc", "alloc"),
        ("cudaHostAlloc_v3020", "cudaHostAlloc", "alloc"),
        ("cuModuleLoadData", "cuModuleLoadData", "other"),
    ],
)
def test_documented_cuda_names(raw, name, api_class):
    assert normalize_api_name(raw) == name
    assert classify_api(name) == api_class


@pytest.mark.parametrize("raw", [None, 7, "", "has space", "x" * 200, "1cuda", "a\nb"])
def test_unsafe_or_malformed_names_are_rejected(raw):
    assert normalize_api_name(raw) is None
