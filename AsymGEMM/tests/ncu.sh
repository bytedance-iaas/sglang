# python3 test_bf16.py

/opt/nvidia/nsight-compute/2025.2.1/ncu -f -o asymCompKernelMain_bf16_256 --set full --section MemoryWorkloadAnalysis_Chart \
 --replay-mode kernel --launch-count 2 --clock-control none asymCompKernelMain_bf16
 
#  asymCompKernelMain_bf16
#  python3 test_bf16.py