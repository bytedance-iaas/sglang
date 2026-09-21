# 原生 interleave CP8 性能初筛

generation7、同一0.236单机8×B300、原镜像/source753bbc090、无tracked修改。以TP8 MTP5/1/6配置加原生prefill CP interleave，dp_size=1；框架自动设attention CP8/attention TP1、dense TP1、DP attention和CP-v2，并禁用prefill graph。chunk32768/static0.85/MTP5/1/6保持。B300真实16k SSE预热通过有效输出和DONE检查；这不代替长上下文质量验收。

同请求文件、独立预热、每点flush cache、max_tokens512、真实EOS，四档C1各3/C10各10，共52/52成功，8点和runner均exit0。TTFT计thinking或content首个有效token，吞吐包含全部completion tokens。小样本用于初筛。

| 输入 | 并发 | TTFT avg s | TTFT P90 s | TPOT avg ms | TPOT P90 ms | output tok/s |
|---|---:|---:|---:|---:|---:|---:|
| 16k | 1 | 0.447 | 0.663 | 5.85 | 6.03 | 140.55 |
| 64k | 1 | 1.312 | 1.532 | 6.43 | 6.46 | 104.04 |
| 100k | 1 | 2.246 | 2.512 | 5.52 | 5.96 | 76.51 |
| 224k | 1 | 4.710 | 4.930 | 6.43 | 7.51 | 47.81 |
| 16k | 10 | 2.388 | 3.490 | 15.10 | 25.18 | 390.30 |
| 64k | 10 | 7.004 | 10.999 | 24.62 | 37.03 | 213.52 |
| 100k | 10 | 11.047 | 17.616 | 36.92 | 52.45 | 138.19 |
| 224k | 10 | 24.719 | 39.848 | 68.58 | 112.43 | 70.98 |

C10四档TTFT比TP8/DP8均改善，224k由TP8 66.250s、DP8 44.121s降到24.719s。16k TTFT avg/P90通过当前点门槛，但TPOT P90 25.18ms仍超过22ms；其余C10 TTFT仍不达标。不能宣称完整性能通过。CP8单请求TTFT最佳，但decode速度仍不及TP8；长输入并发TPOT也不一定优于DP8。

下一步固定CP8与MTP，比较chunk16k/64k对prefill及decode干扰的取舍，保留当前32k为参照；只有实测改善的配置才扩大样本并补五档及负载边界。接口问题继续暂缓。完整服务日志无allocator OOM警告/traceback，原始归档8个SQLite及exit0已校验；切换前Pod Ready/restart0、runner已退出。

[汇总](cp8-mtp516-summary.json) · [运行身份](cp8-mtp516-runtime-identity.json) · [原始数据](cp8-mtp516-perf.tar.gz)
