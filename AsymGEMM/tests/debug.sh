# compute-sanitizer python3 test_bf16.py > debug.log 2>&1
compute-sanitizer ./asymCompKernelMain_bf16 > debug.log 2>&1

# TORCH_SHOW_CPP_STACKTRACES=1 TORCH_CPP_LOG_LEVEL=INFO \
# python3 -u -X faulthandler test_fp8.py 2>&1 | tee run_torch_stack.log
