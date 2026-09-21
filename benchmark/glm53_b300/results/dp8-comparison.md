# DP Attention8 性能初筛

同一0.236单机8×B300、固定镜像/source753bbc090，原生MTP5/1/6。generation6仅在上一候选上新增DP8/DP attention；实际每rank chunk由32768变为4096，调度保守系数也按源码缩放，因此不是仅改变张量分片的隔离实验。接口补丁未部署。

四档C1每点3请求、C10每点10请求，52/52成功，8点及runner均exit0。每点清缓存、独立预热、相同请求文件、max_tokens512和真实EOS。TTFT计首个有效thinking或content，吞吐包含全部completion tokens。小样本仅作初筛，不代表稳定容量。

| 输入 | 并发 | TTFT avg s | TPOT avg ms | output tok/s |
|---|---:|---:|---:|---:|
| 16k | 1 | 1.239 | 6.97 | 86.74 |
| 64k | 1 | 4.456 | 7.83 | 57.12 |
| 100k | 1 | 7.021 | 7.46 | 33.19 |
| 224k | 1 | 17.731 | 7.33 | 19.42 |
| 16k | 10 | 3.490 | 12.47 | 344.96 |
| 64k | 10 | 12.131 | 21.78 | 158.63 |
| 100k | 10 | 18.923 | 24.63 | 138.45 |
| 224k | 10 | 44.121 | 67.91 | 48.56 |

与TP8 MTP5/1/6相比，C10的100k/224k TTFT由24.869/66.250秒降到18.923/44.121秒；TPOT由100.28/182.76ms降到24.63/67.91ms。全部C10 TTFT仍超项目门槛；C1四档TTFT和TPOT则变慢，不能判DP8全面优胜。

224k prefill发现CUDA allocator申请约1.82GB失败警告（free约0.27GB），随后prefill继续，最终所有请求成功且Pod无重启。该警告不等于终态OOM失败，但表明显存临时空间值得检查；完整服务日志已归档；8个SQLite、runner exit0及52/52成功均已核对。

下一候选为固定版本已有的interleave prefill CP8，以验证单请求上下文分片是否改善长输入prefill。其dp_size必须为1，不能与DP8叠加；源码注明仅验证过Hopper，B300需实测真实请求。保留DP8作为并发吞吐候选，后续根据CP结果决定chunk/KV预算调优。

[测量汇总](dp8-mtp516-summary.json) · [运行身份](dp8-mtp516-runtime-identity.json) · [原始数据](dp8-mtp516-perf.tar.gz)
