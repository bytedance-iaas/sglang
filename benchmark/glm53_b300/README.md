# GLM-5.3 FP8 / single-node 8×B300 benchmark archive

Paused on 2026-09-22. This branch preserves the measured setup and scripts; it does not claim final acceptance or deploy an inference-source fix.

## Reproduce the retained candidate

- Source: `753bbc090a6907cdddb69ce35abbc4444a6fd465` (this branch's base).
- Image: `iaas-gpu-cn-beijing.cr.volces.com/serving/sglang@sha256:27d8f8e072165e7deff4d665b2193a9c6812457679d30ce5da31ecccaa840d3e`.
- SGLang 0.5.21.dev28+g753bbc090; Torch 2.13 cu130; FlashInfer 0.6.18.
- `deployment.yaml`: retained CP8/CuTe DSL/DeepGEMM candidate, Service and client Pod. Review namespace, node selector and host mounts for the target environment. Applying it replaces the selected task deployment; the archive operation did not apply it.
- `launch.sh`: identical serving arguments for running inside the image, with `/models/GLM-5.3` mounted. Kubernetes manifest additionally supplies persistent cache environment/mounts.
- Prefill CP interleave requires dp_size=1; TP8, EP1, MTP steps/topk/draft=5/1/6, chunk32768, FP8 KV, static0.85, context262144. Max running/graph256 is not validated long-input capacity.

## Benchmark (run inside the client Pod)

Install EvalScope at commit `0f0706ea3d90c022b1aca14b923112f473a20e81` in `/work/bench/venv`, and requests/Transformers 5.12.1 for the tools. Copy the scripts to `/work/requests` before running:

```bash
python3 /work/requests/generate_project_requests.py --model-path /models/GLM-5.3 --output-dir /work/requests/project --copies 100
python3 /work/requests/run_exploratory_perf.py /work/results/NEW_UNIQUE_RUN
```

The runner uses the service `http://glm53-b300-opt.vketest.svc:30000`. Each of 16k/64k/100k/224k has C1 n3 and C10 n10 (52 measured requests), with independent warmup/cache flush. All requests preserve max_tokens512, temperature1, top_p1, streaming and real EOS. Check both process exit and Total=Success=expected, Failed=0; EvalScope exit0 alone is insufficient.

TTFT counts the first effective reasoning OR content token; output throughput includes all completion tokens. Empty role events do not count. Original thinking.disabled is ignored by this image, a documented API incompatibility; no API draft is applied. Real EOS makes output lengths differ between runs.

## Results and limitations

`results/configuration-comparison.csv` preserves 58 points (seven successful configurations × eight points, plus two DeepEP-normal failure-candidate points). Summary JSON and readable comparisons are adjacent. Each successful configuration completed 52/52; no formal full-acceptance claim.

Retained CP8+CuTe DSL C10 TTFT avg: 2.362/7.003/11.012/24.690 s at 16k/64k/100k/224k. Only the 16k small-sample latency check passes; long input misses SLO. EP8 standard improves short-input decode but regresses long input. DeepEP normal crashed at 16k C10 (0/10), with a draft attention batch-size assertion.

The later flashinfer_trtllm_routed candidate is NOT the deployment here: short arithmetic passed, 16k warmup hit a 180s read timeout, runner exit1, no measured points. Native stacks stopped in cooperative-routing INFO logging / spdlog mutex, followed by 300s scheduler watchdog and restart. No logging workaround was applied.

Full five-bin/100-request verification, capacity boundaries, quality, cache/TTL and final API acceptance remain incomplete. Profile covers two prefill chunks of one 64k request, not the full workload. Raw SQLite, traces and logs remain in the original workspace evidence directory and task storage; this Git archive contains compact summaries, not all raw data. `drafts/` preserves the unfinished, untested API patch without applying it.
