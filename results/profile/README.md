# Stock profile

`stock-exclusive-kernel-time.txt` is from a torch-profiler trace of the stock Humming service. The trace covered 21 forward passes: one prefill and twenty 8-token verify-plus-draft steps of a code prompt. The output comes from `tools/kernel_time.py`, which bills each instant to the earliest-started running kernel. That way, kernels launched with PDL (programmatic dependent launch) aren't billed for the GEMM they wait on.

Columns: kernel, exclusive ms, share, calls, µs per call, ms spent waiting behind another kernel.
