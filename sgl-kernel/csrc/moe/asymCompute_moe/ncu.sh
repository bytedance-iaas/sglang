/opt/nvidia/nsight-compute/2025.2.1/ncu -f -o group_gemm_asymetricCompute --set full --section MemoryWorkloadAnalysis_Chart \
 --replay-mode kernel --launch-count 5 --clock-control none \
 python3 kernel_test.py