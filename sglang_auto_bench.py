import os
import re
import logging
import subprocess
import time
import argparse
import math
import csv
import glob
import json
import urllib.error
import urllib.request
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Log directory configuration
LOG_DIR = "benchmark_logs"
RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
CURRENT_RUN_DIR = os.path.join(LOG_DIR, f"run_{RUN_ID}")

if not os.path.exists(CURRENT_RUN_DIR):
    os.makedirs(CURRENT_RUN_DIR)
    logging.info(f"Created log directory: {CURRENT_RUN_DIR}")

BENCH_HOST = "localhost"
BENCH_PORT = "31000"
MODEL_PATH = "/models/MiniMax-M3-FP8/"
DEFAULT_BENCHMARK_SEED = 5
DECODE_SUFFIX_LEN = 0
DECODE_WARMUP_PROMPTS = 10
MMLU_MAX_QUESTIONS = 32
MMLU_MAX_TOKENS = 4
MMLU_ACCURACY_FLOOR = 0.60
PREFILL_CACHE_WORKING_SET_MULTIPLIER = 1
PREFILL_CACHE_MIN_HIT_RATE = 50.0
PREFILL_CACHE_BOUNDARY_MAX_RUNS = 6
PREFILL_CACHE_FINAL_HIT_REPETITIONS = 4
PREFILL_CACHE_KV_PAGE_SIZE = 128
PREFILL_CACHE_READY_MAX_REPLAYS = 3
PREFILL_CACHE_READY_SETTLE_SECONDS = 2.0
PREFILL_CACHE_READY_HIT_RATE_TOLERANCE = 0.1
PREFILL_CACHE_POOL_MAX_INIT_ATTEMPTS = 2
PREFILL_CACHE_MIN_REQUEST_WAVES = 4

# Use a per-process starting point for prefill datasets. Prefill-only consumes a
# new seed per run; prefill-cache consumes one per miss/hit pair and reuses it
# inside that pair.
_PREFILL_SEED_MODULUS = 2**32
_PREFILL_SEED_BASE = (time.time_ns() ^ os.getpid()) % _PREFILL_SEED_MODULUS
_prefill_seed_step = 0


def _next_benchmark_seed(prefill_only):
    global _prefill_seed_step
    if not prefill_only:
        return DEFAULT_BENCHMARK_SEED

    seed = (_PREFILL_SEED_BASE + _prefill_seed_step) % _PREFILL_SEED_MODULUS
    _prefill_seed_step += 1
    return seed


def _extract_input_throughput(output, total_throughput=None, output_throughput=None):
    """Extract input throughput, falling back to total minus output throughput."""
    match = re.search(
        r"Input token throughput\s*\(tok/s\):\s*([0-9.]+)",
        output or "",
        re.IGNORECASE,
    )
    if match:
        return float(match.group(1))
    if total_throughput is not None and output_throughput is not None:
        return total_throughput - output_throughput
    return None


def _extract_cache_report(output):
    """Parse server-reported cache token statistics from bench_serving output."""
    patterns = {
        "duration": r"Benchmark duration\s*\(s\):\s*([0-9.]+)",
        "prompt_tokens": r"Total prompt tokens:\s*([0-9]+)",
        "cached_tokens": r"Total cached tokens:\s*([0-9]+)",
        "cache_hit_rate": r"Cache hit rate:\s*([0-9.]+)%",
    }
    values = {}
    for name, pattern in patterns.items():
        match = re.search(pattern, output or "", re.IGNORECASE)
        if not match:
            return None
        values[name] = float(match.group(1))

    duration = values["duration"]
    prompt_tokens = int(values["prompt_tokens"])
    cached_tokens = int(values["cached_tokens"])
    if duration <= 0 or prompt_tokens <= 0 or cached_tokens > prompt_tokens:
        return None

    uncached_tokens = prompt_tokens - cached_tokens
    # Compute the rate from the exact token counters instead of the rounded
    # percentage printed by bench_serving.
    cache_hit_rate = cached_tokens / prompt_tokens * 100.0
    return {
        "duration": duration,
        "prompt_tokens": prompt_tokens,
        "cached_tokens": cached_tokens,
        "uncached_tokens": uncached_tokens,
        "cache_hit_rate": cache_hit_rate,
        "cache_hit_input_throughput": cached_tokens / duration,
        "cache_miss_input_throughput": uncached_tokens / duration,
    }


def _prefill_cache_usable_kv_tokens(
    max_kv_pool_size,
    context_len,
    page_size=PREFILL_CACHE_KV_PAGE_SIZE,
):
    """Apply one KV page of headroom per context to a physical pool size."""
    max_kv_pool_size = int(max_kv_pool_size)
    context_len = int(context_len)
    page_size = int(page_size)
    if max_kv_pool_size <= 0:
        raise ValueError("max_kv_pool_size must be positive")
    if page_size <= 0:
        raise ValueError("page_size must be positive")
    if context_len <= page_size:
        raise ValueError("context_len must be greater than page_size")

    # This is the integer form of the requested utilization:
    # (context_len / page_size - 1) * page_size / context_len.
    return max_kv_pool_size * (context_len - page_size) // context_len


def _parse_prometheus_labels(raw_labels):
    """Parse the quoted label values needed by the KV-pool metrics."""
    return {
        match.group(1): match.group(2)
        for match in re.finditer(
            r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"',
            raw_labels or "",
        )
    }


def _parse_kv_pool_metrics(metrics_text):
    """Return a conservative logical KV-pool capacity from Prometheus text.

    SGLang emits one copy of these gauges per TP/PP rank.  Those ranks form one
    logical engine pool, so summing the series would multiply the capacity.
    The minimum positive value is conservative when rank-local values differ.

    HiCache's default write-through policy duplicates device-resident KV in the
    host tier, while write-back can make the tiers partly exclusive.  Taking
    max(device, host) is correct for write-through and a safe lower bound for
    write-back; adding both would overstate the default configuration.
    """
    metric_values = {
        "sglang:max_total_num_tokens": [],
        "sglang:hicache_host_total_tokens": [],
    }
    state_metric_names = {
        "sglang:kv_available_tokens",
        "sglang:kv_used_tokens",
        "sglang:kv_evictable_tokens",
    }
    device_state_samples = {}
    sample_pattern = re.compile(
        r"^(sglang:(?:max_total_num_tokens|hicache_host_total_tokens|"
        r"kv_available_tokens|kv_used_tokens|kv_evictable_tokens))"
        r"(?:\{([^}]*)\})?\s+"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*$"
    )
    for raw_line in (metrics_text or "").splitlines():
        match = sample_pattern.match(raw_line.strip())
        if not match:
            continue
        labels = _parse_prometheus_labels(match.group(2))
        engine_type = labels.get("engine_type")
        if engine_type is not None and engine_type != "prefill":
            continue
        value = float(match.group(3))
        if math.isfinite(value) and value >= 0:
            metric_name = match.group(1)
            int_value = int(value)
            if metric_name in metric_values:
                metric_values[metric_name].append(int_value)
            elif metric_name in state_metric_names:
                label_key = tuple(sorted(labels.items()))
                device_state_samples.setdefault(label_key, {})[
                    metric_name
                ] = int_value

    device_values = metric_values["sglang:max_total_num_tokens"]
    host_values = metric_values["sglang:hicache_host_total_tokens"]
    device_tokens = min(device_values) if device_values else 0
    device_capacity_source = "max_total_num_tokens"
    if device_tokens <= 0:
        inferred_device_values = [
            sum(rank_samples[name] for name in state_metric_names)
            for rank_samples in device_state_samples.values()
            if state_metric_names.issubset(rank_samples)
        ]
        if inferred_device_values:
            # This sum can be below max_total_num_tokens when protected tokens
            # create an accounting gap, but remains a safe usable capacity.
            device_tokens = min(inferred_device_values)
            device_capacity_source = "kv_state_gauges_lower_bound"
    host_tokens = min(host_values) if host_values else 0
    effective_tokens = max(device_tokens, host_tokens)
    if effective_tokens <= 0:
        return None
    return {
        "max_kv_pool_size": effective_tokens,
        "device_total_tokens": device_tokens,
        "device_capacity_source": device_capacity_source,
        "hicache_host_total_tokens": host_tokens,
        "device_series": len(device_values),
        "hicache_host_series": len(host_values),
    }


def _read_kv_pool_size_from_metrics(metrics_url=None, timeout=2.0):
    """Read KV-pool capacity from an explicit or automatically derived URL."""
    if metrics_url:
        candidate_urls = [metrics_url]
    else:
        candidate_urls = [f"http://{BENCH_HOST}:{BENCH_PORT}/metrics"]
        if str(BENCH_PORT) != "30000":
            candidate_urls.append(f"http://{BENCH_HOST}:30000/metrics")

    failures = []
    for candidate_url in dict.fromkeys(candidate_urls):
        try:
            request = urllib.request.Request(
                candidate_url,
                headers={"Accept": "text/plain"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                metrics_text = response.read().decode("utf-8", errors="replace")
            pool_metrics = _parse_kv_pool_metrics(metrics_text)
            if pool_metrics is None:
                failures.append(f"{candidate_url}: KV pool gauges missing")
                continue
            pool_metrics["metrics_url"] = candidate_url
            return pool_metrics
        except (OSError, ValueError, urllib.error.URLError) as exc:
            failures.append(f"{candidate_url}: {exc}")

    if failures:
        logging.info(
            "Prefill-cache KV-pool metrics auto-discovery unavailable: "
            + "; ".join(failures)
        )
    return None


def _aggregate_benchmark_results(results, durations):
    """Aggregate repeated benchmark results without losing their raw output."""
    if len(results) != len(durations) or not results:
        raise ValueError("results and durations must be non-empty and have equal size")
    if len(results) == 1:
        return results[0]

    def mean_metric(index):
        values = [result[index] for result in results if result[index] is not None]
        return sum(values) / len(values) if values else None

    def duration_weighted_metric(index):
        samples = [
            (result[index], duration)
            for result, duration in zip(results, durations)
            if result[index] is not None and duration > 0
        ]
        total_duration = sum(duration for _, duration in samples)
        if total_duration <= 0:
            return None
        return sum(value * duration for value, duration in samples) / total_duration

    combined_output = "\n\n".join(
        f"===== Prefill-cache hit replay {index} =====\n{result[6]}"
        for index, result in enumerate(results, start=1)
    )
    return (
        mean_metric(0),
        mean_metric(1),
        mean_metric(2),
        duration_weighted_metric(3),
        duration_weighted_metric(4),
        duration_weighted_metric(5),
        combined_output,
    )


BUILTIN_MMLU_SMOKE_SET = [
    {
        "subject": "abstract_algebra",
        "question": "Which set with the usual addition and multiplication is a field?",
        "choices": ["The integers", "The rational numbers", "The natural numbers", "The even integers"],
        "answer": "B",
    },
    {
        "subject": "anatomy",
        "question": "Which organelle is the primary site of ATP production in eukaryotic cells?",
        "choices": ["Ribosome", "Golgi apparatus", "Mitochondrion", "Lysosome"],
        "answer": "C",
    },
    {
        "subject": "college_physics",
        "question": "For an object in uniform circular motion, the centripetal acceleration points",
        "choices": ["tangent to the path", "away from the center", "toward the center", "opposite the velocity"],
        "answer": "C",
    },
    {
        "subject": "high_school_chemistry",
        "question": "A solution with pH 3 is how many times more acidic than a solution with pH 5?",
        "choices": ["2", "10", "100", "1000"],
        "answer": "C",
    },
    {
        "subject": "high_school_world_history",
        "question": "The Treaty of Versailles formally ended which war?",
        "choices": ["World War I", "World War II", "The Crimean War", "The Seven Years' War"],
        "answer": "A",
    },
    {
        "subject": "logical_fallacies",
        "question": "Attacking a person's character instead of their argument is called",
        "choices": ["straw man", "ad hominem", "false dilemma", "appeal to probability"],
        "answer": "B",
    },
    {
        "subject": "machine_learning",
        "question": "Which technique is primarily used to reduce overfitting by randomly disabling units during training?",
        "choices": ["Dropout", "Beam search", "Tokenization", "Batch decoding"],
        "answer": "A",
    },
    {
        "subject": "professional_medicine",
        "question": "Hemoglobin is responsible primarily for transporting",
        "choices": ["oxygen", "insulin", "bile", "urea"],
        "answer": "A",
    },
]

def run_benchmark(
    input_len,
    output_len,
    req_rate,
    max_conn,
    num_prompts_multiplier=2,
    decode_only=False,
    decode_suffix_len=DECODE_SUFFIX_LEN,
    warmup_prompts=0,
    dataset_name="random",
    dataset_path=None,
    gsp_system_prompt_len=None,
    num_prompts_override=None,
    num_prompts_cap=None,
    prefill_only=False,
    prefill_cache=False,
    benchmark_seed=None,
    flush_cache=False,
    cache_report=False,
    log_label=None,
    gsp_ordered=False,
    gsp_repeat_count=1,
):
    """
    Run the sglang benchmark and extract TTFT and TPOT.
    """
    if num_prompts_override is not None:
        num_prompts = int(num_prompts_override)
    else:
        num_prompts = max_conn * num_prompts_multiplier
    if num_prompts_cap is not None:
        num_prompts = min(num_prompts, int(num_prompts_cap))
    # Most bench_serving modes need enough prompts to avoid unstable tiny runs.
    # In prefill-cache fixed-pool mode, however, num_prompts_override is the
    # exact KV-resident working set size. Raising it to 10 silently overfills
    # long-context pools such as 128K, so honor the override exactly there.
    if not (prefill_cache and num_prompts_override is not None):
        num_prompts = max(num_prompts, 10)
    gsp_repeat_count = int(gsp_repeat_count)
    if gsp_repeat_count < 1:
        raise ValueError("gsp_repeat_count must be at least 1")
    if gsp_repeat_count > 1 and dataset_name != "generated-shared-prefix":
        raise ValueError(
            "gsp_repeat_count is only supported for generated-shared-prefix"
        )
    if benchmark_seed is None:
        benchmark_seed = _next_benchmark_seed(prefill_only)

    random_prefix_len = 0
    random_input_len = int(input_len)
    if decode_only:
        random_input_len = min(int(decode_suffix_len), int(input_len))
        random_prefix_len = max(0, int(input_len) - random_input_len)

    def resolve_gsp_config(current_num_prompts):
        if prefill_only or prefill_cache:
            # Both prefill modes use independent full-length prompts. In
            # prefill-cache mode this ensures that only the second benchmark
            # run hits cache populated by the first run, instead of requests
            # unexpectedly sharing a system prompt within one run.
            question_len = int(input_len)
            system_prompt_len = 0
            # Capping the group count at 64 and using ceil(N / groups) can
            # create more prompts than --num-prompts requested. A group per
            # prompt keeps the generated working set exact and avoids
            # accidental sharing.
            num_groups = current_num_prompts
            prompts_per_group = 1
        else:
            question_len = (
                min(int(decode_suffix_len), int(input_len))
                if decode_only
                else min(128, int(input_len))
            )
            system_prompt_len = gsp_system_prompt_len
            if system_prompt_len is None:
                system_prompt_len = max(1, int(input_len) - int(question_len))
            num_groups = 1 if decode_only else min(64, current_num_prompts)
            prompts_per_group = max(1, math.ceil(current_num_prompts / num_groups))
        return num_groups, prompts_per_group, system_prompt_len, question_len

    def build_cmd(current_num_prompts, current_req_rate, current_max_conn, current_output_len):
        cmd = [
            "python3", "-m", "sglang.bench_serving",
            "--host", BENCH_HOST,
            "--port", BENCH_PORT,
            "--model", MODEL_PATH,
            "--dataset-name", dataset_name,
            "--num-prompts", str(current_num_prompts),
            "--request-rate", str(current_req_rate),
            "--max-concurrency", str(current_max_conn),
            "--seed", str(benchmark_seed),
        ]
        if dataset_path:
            cmd.extend(["--dataset-path", dataset_path])
        if flush_cache:
            # bench_serving flushes after its own warmup and immediately before
            # the measured requests, so this run represents the miss case.
            cmd.append("--flush-cache")
        if cache_report:
            cmd.append("--cache-report")
        if prefill_cache:
            # bench_serving otherwise sends one unmeasured request before the
            # optional flush. Besides adding noise, that warmup becomes an
            # unexpected hit in hit runs. The miss run itself is the warmup
            # for the paired hit run, so disable it on both legs.
            cmd.extend(["--warmup-requests", "0"])

        if dataset_name.startswith("random"):
            cmd.extend([
                "--random-input-len", str(random_input_len),
                "--random-output-len", str(int(current_output_len)),
                "--random-range-ratio", "0" if decode_only else "1",
            ])
            if random_prefix_len > 0:
                cmd.extend(["--random-prefix-len", str(random_prefix_len)])
        elif dataset_name == "sharegpt":
            cmd.extend(["--sharegpt-output-len", str(int(current_output_len))])
        elif dataset_name == "sonnet":
            cmd.extend([
                "--sonnet-input-len", str(int(input_len)),
                "--sonnet-output-len", str(int(current_output_len)),
            ])
        elif dataset_name == "hf":
            cmd.extend(["--hf-output-len", str(int(current_output_len))])
        elif dataset_name == "custom":
            cmd.extend(["--custom-output-len", str(int(current_output_len))])
        elif dataset_name == "generated-shared-prefix":
            num_groups, prompts_per_group, system_prompt_len, question_len = resolve_gsp_config(current_num_prompts)
            cmd.extend([
                "--gsp-num-groups", str(int(num_groups)),
                "--gsp-prompts-per-group", str(int(prompts_per_group)),
                "--gsp-system-prompt-len", str(int(system_prompt_len)),
                "--gsp-question-len", str(int(question_len)),
                "--gsp-output-len", str(int(current_output_len)),
            ])
            if gsp_ordered:
                cmd.append("--gsp-ordered")
            if gsp_repeat_count > 1:
                cmd.extend(["--gsp-repeat-count", str(gsp_repeat_count)])

        return cmd
    
    cmd = build_cmd(num_prompts, req_rate, max_conn, output_len)
    
    env = os.environ.copy()
    env["HF_ENDPOINT"] = "https://hf-mirror.com"
    if prefill_cache and not flush_cache:
        # sglang.bench_serving implicitly POSTs /flush_cache whenever this
        # inherited CI variable is true, even without --flush-cache. Hit-only
        # runs must be controlled solely by the explicit CLI flag.
        env["SGLANG_IS_IN_CI"] = "false"
    
    cmd_str = " ".join(cmd)
    if decode_only and warmup_prompts > 0:
        warmup_cmd = build_cmd(warmup_prompts, "inf", 1, 1)
        warmup_cmd_str = " ".join(warmup_cmd)
        if dataset_name == "generated-shared-prefix":
            gsp_groups, gsp_ppg, gsp_sys_len, gsp_q_len = resolve_gsp_config(warmup_prompts)
            logging.info(
                f"Decode-only warmup: dataset=generated-shared-prefix, "
                f"groups={gsp_groups}, prompts_per_group={gsp_ppg}, "
                f"system_prompt_len={gsp_sys_len}, question_len={gsp_q_len}"
            )
        else:
            logging.info(
                f"Decode-only warmup: shared_prefix={random_prefix_len}, "
                f"suffix={random_input_len}, num_prompts={warmup_prompts}"
            )
        logging.debug(f"Warmup command: {warmup_cmd_str}")
        try:
            subprocess.run(warmup_cmd, env=env, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            warmup_output = e.stdout + "\n" + e.stderr
            log_filename = f"failed_warmup_{int(input_len)}_{int(output_len)}_mc{max_conn}.log"
            log_path = os.path.join(CURRENT_RUN_DIR, log_filename)
            with open(log_path, "w") as f:
                f.write(f"FAILED Warmup Command: {warmup_cmd_str}\n")
                f.write("="*80 + "\n")
                f.write(warmup_output)
            logging.warning(f"Decode-only warmup failed; continuing with measured run. Log: {log_path}")

    decode_detail = ""
    if decode_only and dataset_name == "generated-shared-prefix":
        gsp_groups, gsp_ppg, gsp_sys_len, gsp_q_len = resolve_gsp_config(num_prompts)
        decode_detail = (
            f", groups={gsp_groups}, prompts_per_group={gsp_ppg}, "
            f"system_prompt_len={gsp_sys_len}, question_len={gsp_q_len}"
        )
    elif decode_only:
        decode_detail = f", shared_prefix={random_prefix_len}, suffix={random_input_len}"

    request_count_detail = ""
    if dataset_name == "generated-shared-prefix":
        request_count_detail = (
            f", unique_prompts={num_prompts}, "
            f"repeat_count={gsp_repeat_count}, "
            f"total_requests={num_prompts * gsp_repeat_count}"
        )
    logging.info(
        f"Running benchmark: request_rate={req_rate}, max_concurrency={max_conn}, "
        f"dataset={dataset_name}, seed={benchmark_seed}"
        + request_count_detail
        + decode_detail
    )
    logging.debug(f"Command: {cmd_str}")
    
    try:
        # Run benchmark
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
        output = result.stdout + "\n" + result.stderr
        
        # Save output to log file
        mode_prefix = "decode_" if decode_only else ""
        if log_label:
            mode_prefix += f"{log_label}_"
        repeat_suffix = (
            f"_rep{gsp_repeat_count}" if gsp_repeat_count > 1 else ""
        )
        log_filename = f"{mode_prefix}bench_{dataset_name}_{int(input_len)}_{int(output_len)}_rate{req_rate}_mc{max_conn}_p{num_prompts}{repeat_suffix}.log"
        log_path = os.path.join(CURRENT_RUN_DIR, log_filename)
        with open(log_path, "w") as f:
            f.write(f"Command: {cmd_str}\n")
            f.write("="*80 + "\n")
            f.write(output)
        logging.info(f"Full benchmark output saved to: {log_path}")
        
        # Parse metrics
        # We try to extract Mean or Median or P99 TTFT/TPOT. You can adjust the regex based on exact sglang version.
        ttft_patterns = [r'Mean TTFT\s*\(ms\):\s*([0-9.]+)', r'Median TTFT\s*\(ms\):\s*([0-9.]+)', r'P99 TTFT\s*\(ms\):\s*([0-9.]+)', r'TTFT.*?([0-9.]+)']
        tpot_patterns = [r'Mean TPOT\s*\(ms\):\s*([0-9.]+)', r'Median TPOT\s*\(ms\):\s*([0-9.]+)', r'P99 TPOT\s*\(ms\):\s*([0-9.]+)', r'TPOT.*?([0-9.]+)']
        e2e_patterns = [r'Mean E2E Latency\s*\(ms\):\s*([0-9.]+)', r'Median E2E Latency\s*\(ms\):\s*([0-9.]+)', r'E2E Latency.*?([0-9.]+)']
        throughput_patterns = [r'Total token throughput\s*\(tok/s\):\s*([0-9.]+)', r'Total Throughput.*?([0-9.]+)']
        
        ttft, tpot, e2e_lat, throughput, out_throughput, req_throughput = None, None, None, None, None, None
        
        for p in ttft_patterns:
            m = re.search(p, output, re.IGNORECASE)
            if m:
                ttft = float(m.group(1))
                break
                
        for p in tpot_patterns:
            m = re.search(p, output, re.IGNORECASE)
            if m:
                tpot = float(m.group(1))
                break
        if tpot is None and int(output_len) <= 1:
            tpot = 0.0

        for p in e2e_patterns:
            m = re.search(p, output, re.IGNORECASE)
            if m:
                e2e_lat = float(m.group(1))
                break
                
        for p in throughput_patterns:
            m = re.search(p, output, re.IGNORECASE)
            if m:
                throughput = float(m.group(1))
                break
        
        out_throughput_patterns = [r'Output token throughput\s*\(tok/s\):\s*([0-9.]+)']
        for p in out_throughput_patterns:
            m = re.search(p, output, re.IGNORECASE)
            if m:
                out_throughput = float(m.group(1))
                break
        
        req_throughput_patterns = [r'Request throughput\s*\(req/s\):\s*([0-9.]+)']
        for p in req_throughput_patterns:
            m = re.search(p, output, re.IGNORECASE)
            if m:
                req_throughput = float(m.group(1))
                break
                
        return ttft, tpot, e2e_lat, throughput, out_throughput, req_throughput, output
    except subprocess.CalledProcessError as e:
        output = e.stdout + "\n" + e.stderr
        mode_prefix = "decode_" if decode_only else ""
        if log_label:
            mode_prefix += f"{log_label}_"
        log_filename = f"failed_{mode_prefix}bench_{dataset_name}_{int(input_len)}_{int(output_len)}_rate{req_rate}_mc{max_conn}_p{num_prompts}.log"
        log_path = os.path.join(CURRENT_RUN_DIR, log_filename)
        with open(log_path, "w") as f:
            f.write(f"FAILED Command: {cmd_str}\n")
            f.write("="*80 + "\n")
            f.write(output)
            
        logging.error(f"Benchmark command failed with exit code {e.returncode}")
        logging.error(f"Failed log saved to: {log_path}")
        return None, None, None, None, None, None, output
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return None, None, None, None, None, None, str(e)


def _warm_prefill_cache_pool(
    input_len,
    output_len,
    max_kv_pool_size,
    benchmark_seed,
    max_conn,
    dataset_name="generated-shared-prefix",
    dataset_path=None,
    page_size=PREFILL_CACHE_KV_PAGE_SIZE,
    pool_size_cap=None,
):
    """Populate a pool that stays within a measured KV-token budget.

    Generated prompt lengths can differ from the requested input length after
    tokenizer decode/encode and chat templating. Start from the requested
    length, then shrink and cold-fill again when the server's exact prompt-token
    report exceeds the supplied KV budget.
    """
    max_kv_pool_size = int(max_kv_pool_size)
    page_size = int(page_size)
    usable_kv_pool_size = _prefill_cache_usable_kv_tokens(
        max_kv_pool_size=max_kv_pool_size,
        context_len=input_len,
        page_size=page_size,
    )
    pool_utilization = usable_kv_pool_size / max_kv_pool_size
    output_tokens_per_prompt = max(0, int(output_len))
    estimated_tokens_per_prompt = max(
        1, int(input_len) + output_tokens_per_prompt
    )
    pool_size = usable_kv_pool_size // estimated_tokens_per_prompt
    if pool_size_cap is not None:
        pool_size = min(pool_size, int(pool_size_cap))
    if pool_size < 1:
        logging.error(
            f"max_kv_pool_size={max_kv_pool_size} leaves "
            f"usable_kv_tokens={usable_kv_pool_size} after the page-headroom "
            f"ratio and cannot hold one estimated "
            f"{estimated_tokens_per_prompt}-token requests."
        )
        return None

    max_calibration_attempts = 3
    for attempt in range(1, max_calibration_attempts + 1):
        warm_max_conn = max(1, min(int(pool_size), int(max_conn)))
        logging.info(
            "Prefill-cache pool cold-fill: "
            f"max_kv_tokens={max_kv_pool_size}, "
            f"usable_kv_tokens={usable_kv_pool_size} "
            f"({pool_utilization * 100.0:.2f}%, context_len={input_len}, "
            f"page_size={page_size}), pool_size={pool_size}, "
            f"RR=inf, MC={warm_max_conn}, seed={benchmark_seed}, "
            f"attempt={attempt}/{max_calibration_attempts}"
        )
        miss_result = run_benchmark(
            input_len=input_len,
            output_len=output_len,
            req_rate="inf",
            max_conn=warm_max_conn,
            num_prompts_multiplier=1,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            num_prompts_override=pool_size,
            prefill_cache=True,
            benchmark_seed=benchmark_seed,
            flush_cache=True,
            cache_report=True,
            log_label=(
                f"cache_pool_miss_seed{benchmark_seed}_attempt{attempt}"
            ),
            gsp_ordered=True,
        )
        if all(metric is None for metric in miss_result[:6]):
            logging.error("Prefill-cache pool cold-fill benchmark failed.")
            return None

        cold_cache_report = _extract_cache_report(miss_result[6])
        if cold_cache_report is None:
            logging.error(
                "Prefill-cache pool cold-fill did not return a valid cache report."
            )
            return None
        if cold_cache_report["cached_tokens"] > 0:
            logging.error(
                "Prefill-cache pool cold-fill was not cold: reported "
                f"{cold_cache_report['cached_tokens']} cached tokens "
                f"({cold_cache_report['cache_hit_rate']:.2f}%)."
            )
            return None

        measured_kv_tokens = (
            cold_cache_report["prompt_tokens"]
            + pool_size * output_tokens_per_prompt
        )
        if measured_kv_tokens <= usable_kv_pool_size:
            logging.info(
                "Prefill-cache pool cold-fill verified: "
                f"pool_size={pool_size}, measured_kv_tokens="
                f"{measured_kv_tokens}/{usable_kv_pool_size} usable "
                f"({measured_kv_tokens / usable_kv_pool_size * 100.0:.2f}%), "
                "cache hit rate=0.00%."
            )
            return {
                "pool_size": pool_size,
                "prompt_tokens": cold_cache_report["prompt_tokens"],
                "measured_kv_tokens": measured_kv_tokens,
                "max_kv_pool_size": max_kv_pool_size,
                "usable_kv_pool_size": usable_kv_pool_size,
                "pool_utilization": pool_utilization,
                "page_size": page_size,
                "calibration_attempts": attempt,
            }

        adjusted_pool_size = min(
            pool_size - 1,
            int(pool_size * usable_kv_pool_size / measured_kv_tokens),
        )
        if adjusted_pool_size < 1:
            logging.error(
                "Measured prompt size leaves room for fewer than one request: "
                f"measured_kv_tokens={measured_kv_tokens}, "
                f"usable_kv_pool_size={usable_kv_pool_size}, "
                f"max_kv_pool_size={max_kv_pool_size}."
            )
            return None
        logging.warning(
            "Cold-fill exceeded the KV-token budget; shrinking prompt pool "
            f"from {pool_size} to {adjusted_pool_size} and flushing again."
        )
        pool_size = adjusted_pool_size

    logging.error(
        "Unable to calibrate the prompt pool within the KV-token budget after "
        f"{max_calibration_attempts} cold-fill attempts."
    )
    return None


def run_prefill_cache_benchmark(
    input_len,
    output_len,
    req_rate,
    max_conn,
    num_prompts_multiplier=2,
    dataset_name="generated-shared-prefix",
    dataset_path=None,
    num_prompts_override=None,
    num_prompts_cap=None,
    hit_repetitions=1,
    benchmark_seed=None,
    skip_cache_miss=False,
    gsp_ordered=False,
    gsp_repeat_count=1,
    hit_log_label=None,
):
    """Run cache-hit measurements, optionally preceded by a cache-miss fill.

    The default behavior allocates a fresh seed for each pair, flushes the
    cache, and reuses the seed for the hit run. ``skip_cache_miss`` is for a
    caller-managed cache session: it requires an explicit seed and performs
    only the hit replay against an already populated pool.

    ``hit_repetitions`` replays the same generated working set without
    increasing its unique prompt count. Returns ``(hit_result,
    miss_input_throughput, hit_input_throughput, cache_hit_rate)``. All three
    cache metrics are calculated from the hit runs' server-reported cached
    token counters.
    """
    if dataset_name != "generated-shared-prefix":
        raise ValueError(
            "prefill-cache mode requires dataset_name=generated-shared-prefix"
        )
    if hit_repetitions < 1:
        raise ValueError("hit_repetitions must be at least 1")
    if skip_cache_miss and benchmark_seed is None:
        raise ValueError("skip_cache_miss requires an explicit benchmark_seed")

    if benchmark_seed is None:
        benchmark_seed = _next_benchmark_seed(True)
    common_args = {
        "input_len": input_len,
        "output_len": output_len,
        "req_rate": req_rate,
        "max_conn": max_conn,
        "num_prompts_multiplier": num_prompts_multiplier,
        "dataset_name": dataset_name,
        "dataset_path": dataset_path,
        "num_prompts_override": num_prompts_override,
        "num_prompts_cap": num_prompts_cap,
        "prefill_cache": True,
        "benchmark_seed": benchmark_seed,
        "cache_report": True,
        "gsp_ordered": gsp_ordered,
        "gsp_repeat_count": gsp_repeat_count,
    }

    if not skip_cache_miss:
        logging.info(
            f"Prefill-cache miss run: RR={req_rate}, MC={max_conn}, "
            f"seed={benchmark_seed}"
        )
        miss_result = run_benchmark(
            **common_args,
            flush_cache=True,
            log_label=f"cache_miss_seed{benchmark_seed}",
        )
        if all(metric is None for metric in miss_result[:6]):
            logging.warning(
                "Prefill-cache miss run failed; skipping the hit run because "
                "the dataset was not reliably populated."
            )
            return miss_result, None, None, None

        cold_cache_report = _extract_cache_report(miss_result[6])
        if cold_cache_report is None:
            logging.error(
                "Prefill-cache miss run did not return a valid --cache-report; "
                "cannot verify real cache hits."
            )
            return miss_result, None, None, None
        if cold_cache_report["cached_tokens"] > 0:
            logging.warning(
                "Cold-cache run unexpectedly reported "
                f"{cold_cache_report['cached_tokens']} cached tokens "
                f"({cold_cache_report['cache_hit_rate']:.2f}%)."
            )
        else:
            logging.info("Cold-cache run verified: cache hit rate=0.00%")
    else:
        logging.info(
            f"Prefill-cache session hit-only run: RR={req_rate}, "
            f"MC={max_conn}, seed={benchmark_seed}"
        )

    hit_results = []
    hit_cache_reports = []
    for replay_index in range(1, hit_repetitions + 1):
        replay_detail = (
            f", replay={replay_index}/{hit_repetitions}"
            if hit_repetitions > 1
            else ""
        )
        logging.info(
            f"Prefill-cache hit run: RR={req_rate}, MC={max_conn}, "
            f"seed={benchmark_seed}{replay_detail}"
        )
        log_label = hit_log_label or f"cache_hit_seed{benchmark_seed}"
        if hit_repetitions > 1:
            log_label += f"_round{replay_index}"
        hit_result = run_benchmark(
            **common_args,
            flush_cache=False,
            log_label=log_label,
        )
        hit_cache_report = _extract_cache_report(hit_result[6])
        if hit_cache_report is None:
            logging.error(
                "Prefill-cache hit run did not return a valid --cache-report; "
                "candidate will not participate in cache-hit optimization."
            )
            return hit_result, None, None, None
        logging.info(
            f"Prefill-cache hit replay {replay_index}: "
            f"cached_tokens={hit_cache_report['cached_tokens']}/"
            f"{hit_cache_report['prompt_tokens']}, "
            f"cache hit rate={hit_cache_report['cache_hit_rate']:.2f}%"
        )
        hit_results.append(hit_result)
        hit_cache_reports.append(hit_cache_report)

    total_duration = sum(report["duration"] for report in hit_cache_reports)
    total_prompt_tokens = sum(
        report["prompt_tokens"] for report in hit_cache_reports
    )
    total_cached_tokens = sum(
        report["cached_tokens"] for report in hit_cache_reports
    )
    total_uncached_tokens = total_prompt_tokens - total_cached_tokens
    miss_input_throughput = total_uncached_tokens / total_duration
    hit_input_throughput = total_cached_tokens / total_duration
    cache_hit_rate = total_cached_tokens / total_prompt_tokens * 100.0
    hit_result = _aggregate_benchmark_results(
        hit_results,
        [report["duration"] for report in hit_cache_reports],
    )
    logging.info(
        "Prefill-cache measured result: "
        f"cached_tokens={total_cached_tokens}/{total_prompt_tokens}, "
        f"cache hit rate={cache_hit_rate:.2f}%, "
        f"cache hit input throughput={hit_input_throughput:.2f} tok/s, "
        f"cache miss input throughput={miss_input_throughput:.2f} tok/s, "
        f"hit replays={hit_repetitions}"
    )
    return (
        hit_result,
        miss_input_throughput,
        hit_input_throughput,
        cache_hit_rate,
    )


def _verify_prefill_cache_pool_ready(
    input_len,
    output_len,
    pool_size,
    benchmark_seed,
    max_conn,
    dataset_name="generated-shared-prefix",
    dataset_path=None,
    page_size=PREFILL_CACHE_KV_PAGE_SIZE,
):
    """Require a full-pool cache hit before starting MC optimization.

    Completing the cold-fill requests does not guarantee that asynchronous
    HiCache writes are visible yet. Replay the complete, ordered working set
    outside optimization and use the replay itself as the readiness barrier.
    A failed replay also primes missing entries, so a later replay can recover
    without another flush.
    """
    pool_size = int(pool_size)
    validation_max_conn = max(1, min(pool_size, int(max_conn)))
    expected_page_hit_rate = (
        (int(input_len) - int(page_size)) / int(input_len) * 100.0
    )
    ready_hit_rate_floor = max(
        PREFILL_CACHE_MIN_HIT_RATE,
        expected_page_hit_rate - PREFILL_CACHE_READY_HIT_RATE_TOLERANCE,
    )
    observed_hit_rates = []

    logging.info(
        "Waiting for the cold-filled HiCache pool before full-pool "
        f"validation ({PREFILL_CACHE_READY_SETTLE_SECONDS:.1f}s)."
    )
    time.sleep(PREFILL_CACHE_READY_SETTLE_SECONDS)
    for attempt in range(1, PREFILL_CACHE_READY_MAX_REPLAYS + 1):
        logging.info(
            "Prefill-cache full-pool readiness validation: "
            f"pool_size={pool_size}, RR=inf, MC={validation_max_conn}, "
            f"seed={benchmark_seed}, attempt={attempt}/"
            f"{PREFILL_CACHE_READY_MAX_REPLAYS}, required_cache_hit="
            f"{ready_hit_rate_floor:.2f}%."
        )
        _, _, _, cache_hit_rate = run_prefill_cache_benchmark(
            input_len=input_len,
            output_len=output_len,
            req_rate="inf",
            max_conn=validation_max_conn,
            num_prompts_multiplier=PREFILL_CACHE_WORKING_SET_MULTIPLIER,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            num_prompts_override=pool_size,
            hit_repetitions=1,
            benchmark_seed=benchmark_seed,
            skip_cache_miss=True,
            gsp_ordered=True,
            hit_log_label=(
                f"cache_pool_verify_seed{benchmark_seed}_attempt{attempt}"
            ),
        )
        observed_hit_rates.append(cache_hit_rate)
        if (
            cache_hit_rate is not None
            and cache_hit_rate >= ready_hit_rate_floor
        ):
            logging.info(
                "Prefill-cache full pool is ready: "
                f"cache hit rate={cache_hit_rate:.2f}% >= "
                f"{ready_hit_rate_floor:.2f}% after {attempt} validation "
                "replay(s). MC optimization can start."
            )
            return {
                "ready": True,
                "attempts": attempt,
                "cache_hit_rate": cache_hit_rate,
                "cache_hit_rate_floor": ready_hit_rate_floor,
                "observed_hit_rates": observed_hit_rates,
            }

        logging.warning(
            "Full-pool cache is not ready yet: "
            f"cache hit rate={cache_hit_rate if cache_hit_rate is not None else 'N/A'}% "
            f"< {ready_hit_rate_floor:.2f}%. This validation replay is not "
            "included in optimization."
        )
        if attempt < PREFILL_CACHE_READY_MAX_REPLAYS:
            time.sleep(PREFILL_CACHE_READY_SETTLE_SECONDS)

    logging.error(
        "Prefill-cache full pool failed readiness validation after "
        f"{PREFILL_CACHE_READY_MAX_REPLAYS} identical replays; refusing to "
        "start hit-only MC optimization with an unverified working set."
    )
    return {
        "ready": False,
        "attempts": PREFILL_CACHE_READY_MAX_REPLAYS,
        "cache_hit_rate": observed_hit_rates[-1] if observed_hit_rates else None,
        "cache_hit_rate_floor": ready_hit_rate_floor,
        "observed_hit_rates": observed_hit_rates,
    }


def _parse_mmlu_csv(path):
    questions = []
    subject = os.path.splitext(os.path.basename(path))[0]
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 6:
                    continue
                answer = row[-1].strip().upper()
                if answer not in {"A", "B", "C", "D"}:
                    continue
                questions.append({
                    "subject": subject,
                    "question": row[0].strip(),
                    "choices": [row[1].strip(), row[2].strip(), row[3].strip(), row[4].strip()],
                    "answer": answer,
                })
    except Exception as e:
        logging.warning(f"Failed to parse MMLU csv {path}: {e}")
    return questions


def _load_mmlu_questions(max_questions=MMLU_MAX_QUESTIONS):
    candidates = [
        "mmlu",
        "MMLU",
        "data/mmlu",
        "datasets/mmlu",
        "/data/mmlu",
        "/data/MMLU",
        "/data/datasets/mmlu",
        "/workspace/data/mmlu",
    ]
    csv_paths = []
    for base in candidates:
        if not os.path.exists(base):
            continue
        csv_paths.extend(glob.glob(os.path.join(base, "test", "*.csv")))
        csv_paths.extend(glob.glob(os.path.join(base, "dev", "*.csv")))
        csv_paths.extend(glob.glob(os.path.join(base, "*.csv")))

    csv_paths = sorted(set(csv_paths))
    questions = []
    for path in csv_paths:
        for question in _parse_mmlu_csv(path):
            questions.append(question)
            if len(questions) >= max_questions:
                return questions, f"local_mmlu_csv:{os.path.dirname(path)}"

    return BUILTIN_MMLU_SMOKE_SET[:max_questions], "builtin_mmlu_smoke"


def _format_mmlu_prompt(item):
    choices = item["choices"]
    return (
        "Answer the following multiple choice question. "
        "Respond with only the letter A, B, C, or D.\n\n"
        f"Question: {item['question']}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n"
        "Answer:"
    )


def _extract_choice(text):
    if not text:
        return None
    match = re.search(r"\b([ABCD])\b", text.upper())
    return match.group(1) if match else None


def _query_completion(prompt, timeout_sec=60):
    url = f"http://{BENCH_HOST}:{BENCH_PORT}/v1/completions"
    payload = {
        "model": MODEL_PATH,
        "prompt": prompt,
        "max_tokens": MMLU_MAX_TOKENS,
        "temperature": 0,
        "stop": ["\n"],
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
        body = resp.read().decode("utf-8")
    parsed = json.loads(body)
    return parsed.get("choices", [{}])[0].get("text", "")


def run_mmlu_probe():
    questions, source = _load_mmlu_questions()
    log_path = os.path.join(CURRENT_RUN_DIR, "mmlu_probe.json")
    details = []
    correct = 0
    attempted = 0

    logging.info(f"=== MMLU Accuracy Probe: source={source}, questions={len(questions)} ===")
    for idx, item in enumerate(questions, start=1):
        prompt = _format_mmlu_prompt(item)
        try:
            raw_answer = _query_completion(prompt)
            pred = _extract_choice(raw_answer)
            is_correct = pred == item["answer"]
            attempted += 1
            correct += int(is_correct)
            details.append({
                "idx": idx,
                "subject": item["subject"],
                "prediction": pred,
                "answer": item["answer"],
                "correct": is_correct,
                "raw_answer": raw_answer,
            })
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, KeyError) as e:
            details.append({
                "idx": idx,
                "subject": item["subject"],
                "prediction": None,
                "answer": item["answer"],
                "correct": False,
                "error": str(e),
            })
            logging.warning(f"MMLU probe request failed at question {idx}: {e}")
            break

    accuracy = (correct / attempted) if attempted else None
    passed = accuracy is not None and accuracy >= MMLU_ACCURACY_FLOOR
    result = {
        "status": "ok" if attempted else "unavailable",
        "source": source,
        "correct": correct,
        "total": attempted,
        "accuracy": accuracy,
        "accuracy_floor": MMLU_ACCURACY_FLOOR,
        "passed": passed if attempted else None,
        "log_path": log_path,
        "details": details,
    }

    with open(log_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    if accuracy is None:
        logging.warning(f"MMLU Accuracy Probe unavailable. Details saved to {log_path}")
    else:
        logging.info(
            f"MMLU Accuracy Probe: {correct}/{attempted} = {accuracy:.2%} "
            f"({'PASS' if passed else 'FAIL'}; floor={MMLU_ACCURACY_FLOOR:.0%}). "
            f"Details saved to {log_path}"
        )
    return result


def _find_optimal_decode_only(
    input_len,
    output_len,
    slo_tpot,
    dp,
    decode_suffix_len,
    warmup_prompts,
    dataset_name,
    dataset_path,
    gsp_system_prompt_len,
    decode_kv_pool_tokens=994816,
):
    """
    Fast decode-only search. Strategy:
      - Lock RR=inf so MC is the only throttle (radix cache absorbs prefill).
      - Warm up the shared prefix once at the very start; subsequent runs skip warmup.
      - Doubling phase: start MC at max(dp*16, 64), double until TPOT > SLO or
        output throughput stops improving.
      - Bisection phase: 2-3 bisection steps between the last SLO-OK MC and the
        first SLO-violating (or throughput-declining) MC to lock the boundary.
      - Track SLO-best (TPOT <= slo_tpot, max output throughput) and Max-best
        (any TPOT, max output throughput) in a single pass; no separate Phase 2.
      - Final 4x re-test on the SLO-best config for a stable report number.

    Returns the same result dict shape as find_optimal_throughput().
    """
    mode_name = "Decode-Only"
    throughput_label = "Output token throughput"
    logging.info(
        f"=== Decode-Only Fast Search [{mode_name}] for Input: {input_len}, Output: {output_len} ==="
    )
    logging.info(f"SLO Target - TPOT: {slo_tpot}ms (TTFT ignored)")

    # MC starting point and ceilings.
    decode_kv_pool_tokens = int(decode_kv_pool_tokens)
    if input_len + output_len > decode_kv_pool_tokens:
        skip_reason = (
            f"input+output={input_len + output_len} exceeds decode KV pool "
            f"{decode_kv_pool_tokens}"
        )
        logging.warning(
            f"Skipping decode-only for input_len={input_len}: {skip_reason}."
        )
        return {
            "input_len": input_len,
            "output_len": output_len,
            "mode": "decode-only",
            "dataset_name": dataset_name,
            "slo": None,
            "max": None,
            "history": [],
            "interrupted": False,
            "skip_reason": skip_reason,
        }

    decode_mc_grid = [2, 4, 8, 12, 16, 24, 32, 64]
    mc_start = max(dp * 2, 16)
    mc_ceiling = min(
        128,
        max(1, int(0.85 * decode_kv_pool_tokens / max(1, input_len + output_len))),
    )
    mc_candidates = [mc for mc in decode_mc_grid if mc <= mc_ceiling]
    if not mc_candidates:
        # 1M capacity-profile runs may only fit a single resident request.
        # Keep the regular reported grid at 2/4/8/... when possible, but allow
        # MC=1 as a single-point fallback when the service can hold one request
        # and the safety-scaled ceiling drops below 2.
        mc_candidates = [1]
    mc = min(mc_candidates, key=lambda candidate: abs(candidate - min(mc_start, mc_ceiling)))

    # Decode-only num_prompts policy: keep exploration runs short.
    # Cap exploration so long-context probes do not overfill the decode KV pool.
    explore_cap = min(128, max(min(mc_start, mc_ceiling) * 2, 10))

    all_results = []
    best_valid_config = None
    best_valid_metrics = None
    best_max_config = None
    best_max_metrics = None
    best_overall_throughput = 0.0
    best_max_throughput = 0.0
    interrupted = False

    # Run shared-prefix warmup ONCE up front; subsequent iterations skip it.
    warmup_done = False

    def _bench(mc_value, num_prompts_cap=explore_cap, run_warmup=False):
        return run_benchmark(
            input_len,
            output_len,
            "inf",
            mc_value,
            num_prompts_multiplier=2,
            decode_only=True,
            decode_suffix_len=decode_suffix_len,
            warmup_prompts=warmup_prompts if run_warmup else 0,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            gsp_system_prompt_len=gsp_system_prompt_len,
            num_prompts_cap=num_prompts_cap,
        )

    def _record(phase, mc_value, ttft, tpot, e2e_lat, throughput, out_thr):
        in_thr = (throughput - out_thr) if (throughput and out_thr) else None
        slo_ok_flag = (tpot <= slo_tpot) if tpot is not None else None
        all_results.append({
            "iter": len(all_results) + 1, "phase": phase,
            "req_rate": "inf", "max_conn": mc_value,
            "ttft": ttft, "tpot": tpot, "e2e": e2e_lat,
            "throughput": throughput, "out_throughput": out_thr,
            "in_throughput": in_thr,
            "objective_throughput": out_thr,
            "slo_ok": slo_ok_flag,
        })

    def _update_bests(mc_value, ttft, tpot, e2e_lat, throughput, out_thr):
        nonlocal best_valid_config, best_valid_metrics, best_overall_throughput
        nonlocal best_max_config, best_max_metrics, best_max_throughput
        if out_thr is None:
            return
        if out_thr > best_max_throughput:
            logging.info(
                f"New Max {throughput_label}! {out_thr:.2f} tok/s "
                f"(was {best_max_throughput:.2f}) at MC={mc_value}"
            )
            best_max_throughput = out_thr
            best_max_config = ("inf", mc_value)
            best_max_metrics = (ttft, tpot, e2e_lat, throughput, out_thr)
        if tpot is not None and tpot <= slo_tpot and out_thr >= best_overall_throughput:
            logging.info(
                f"New Best SLO-Compliant {throughput_label}! {out_thr:.2f} tok/s "
                f"(was {best_overall_throughput:.2f}) at MC={mc_value}"
            )
            best_overall_throughput = out_thr
            best_valid_config = ("inf", mc_value)
            best_valid_metrics = (ttft, tpot, e2e_lat, throughput, out_thr)

    SLEEP_SEC = 0.5

    # Snap MC helper
    def _snap_mc(value):
        # Decode-only results are reported only on the agreed benchmark grid.
        # This avoids odd or hard-to-explain MC values while still allowing
        # non-DP-aligned stable points such as 12 and 24.
        value = max(1, int(round(value)))
        return min(mc_candidates, key=lambda candidate: abs(candidate - value))

    def _next_higher_mc(value):
        for candidate in mc_candidates:
            if candidate > value:
                return candidate
        return None

    def _next_lower_mc(value):
        for candidate in reversed(mc_candidates):
            if candidate < value:
                return candidate
        return None

    def _mid_grid_mc(low, high):
        between = [candidate for candidate in mc_candidates if low < candidate < high]
        if not between:
            return None
        return between[len(between) // 2]

    last_ok_mc = None  # highest MC seen with TPOT <= SLO
    first_bad_mc = None  # lowest MC seen with TPOT > SLO or crash

    try:
        # ===== Doubling phase =====
        consecutive_no_improvement = 0
        last_throughput = 0.0
        for d_iter in range(1, 11):  # cap at 10 doublings (covers up to MC ~ 64k)
            mc = _snap_mc(mc)
            logging.info(f"--- [Doubling] Iter {d_iter} (RR=inf, MC={mc}) ---")
            ttft, tpot, e2e_lat, throughput, out_thr, req_thr, _ = _bench(
                mc, run_warmup=(not warmup_done)
            )
            warmup_done = True

            _record(1, mc, ttft, tpot, e2e_lat, throughput, out_thr)

            if out_thr is None or tpot is None:
                logging.warning("Benchmark failed/unparseable. Treating as ceiling and stopping doubling.")
                first_bad_mc = mc if first_bad_mc is None else min(first_bad_mc, mc)
                mc_ceiling = min(mc_ceiling, mc)
                break

            logging.info(
                f"Result: TPOT={tpot:.2f}ms, E2E={e2e_lat:.2f}ms, "
                f"{throughput_label}={out_thr:.2f}tok/s"
            )
            _update_bests(mc, ttft, tpot, e2e_lat, throughput, out_thr)

            tpot_violated = tpot > slo_tpot
            improvement_ratio = (out_thr - last_throughput) / max(last_throughput, 1e-6)

            if tpot_violated:
                logging.info(f"TPOT {tpot:.2f}ms > SLO {slo_tpot}ms. Switching to bisection.")
                first_bad_mc = mc
                break

            # SLO ok at this MC
            last_ok_mc = mc
            consecutive_no_improvement = (
                consecutive_no_improvement + 1 if improvement_ratio < 0.05 else 0
            )
            last_throughput = out_thr

            if consecutive_no_improvement >= 2:
                logging.info(
                    "Output throughput plateaued for 2 doublings. Stopping doubling phase."
                )
                first_bad_mc = None  # no SLO violation; bisection unnecessary
                break

            next_mc = _next_higher_mc(mc)
            if next_mc is None:
                logging.info(f"MC ceiling {mc_ceiling} reached. Stopping doubling phase.")
                break
            mc = next_mc
            time.sleep(SLEEP_SEC)

        # If the initial safe probe is already above the TPOT SLO, search
        # downward first.  The previous logic stopped here with no SLO point,
        # which failed to answer the actual question: the maximum MC that still
        # satisfies TPOT <= SLO.
        if first_bad_mc is not None and last_ok_mc is None:
            mc = _next_lower_mc(first_bad_mc)
            while mc is not None:
                if mc == first_bad_mc:
                    break
                logging.info(f"--- [Backoff] (RR=inf, MC={mc}) ---")
                ttft, tpot, e2e_lat, throughput, out_thr, req_thr, _ = _bench(mc)
                _record(1, mc, ttft, tpot, e2e_lat, throughput, out_thr)

                if out_thr is None or tpot is None:
                    logging.warning("Backoff probe failed/unparseable. Treating as bad.")
                    first_bad_mc = min(first_bad_mc, mc)
                else:
                    logging.info(
                        f"Result: TPOT={tpot:.2f}ms, "
                        f"{throughput_label}={out_thr:.2f}tok/s"
                    )
                    _update_bests(mc, ttft, tpot, e2e_lat, throughput, out_thr)
                    if tpot <= slo_tpot:
                        last_ok_mc = mc
                        break
                    first_bad_mc = min(first_bad_mc, mc)

                mc = _next_lower_mc(mc)
                time.sleep(SLEEP_SEC)

        # ===== Bisection phase =====
        if last_ok_mc is not None and first_bad_mc is not None:
            logging.info(
                f"=== Bisection between MC={last_ok_mc} (OK) and MC={first_bad_mc} (BAD) ==="
            )
            for b_iter in range(1, 8):
                mid = _mid_grid_mc(last_ok_mc, first_bad_mc)
                if mid is None or mid == last_ok_mc or mid == first_bad_mc:
                    logging.info("Bisection converged at grid resolution.")
                    break
                logging.info(f"--- [Bisection] Iter {b_iter} (RR=inf, MC={mid}) ---")
                ttft, tpot, e2e_lat, throughput, out_thr, req_thr, _ = _bench(mid)
                _record(2, mid, ttft, tpot, e2e_lat, throughput, out_thr)

                if out_thr is None or tpot is None:
                    logging.warning("Bisection probe crashed. Treating as bad.")
                    first_bad_mc = mid
                    time.sleep(SLEEP_SEC)
                    continue

                logging.info(
                    f"Result: TPOT={tpot:.2f}ms, {throughput_label}={out_thr:.2f}tok/s"
                )
                _update_bests(mid, ttft, tpot, e2e_lat, throughput, out_thr)

                if tpot <= slo_tpot:
                    last_ok_mc = mid
                else:
                    first_bad_mc = mid
                time.sleep(SLEEP_SEC)
    except KeyboardInterrupt:
        logging.info("\nUser interrupted. Returning results collected so far...")
        interrupted = True

    result = {
        "input_len": input_len,
        "output_len": output_len,
        "mode": "decode-only",
        "dataset_name": dataset_name,
        "slo": None,
        "max": None,
        "history": all_results,
        "interrupted": interrupted,
    }

    # --- SLO-Compliant final report (with optional 4x re-test) ---
    if best_valid_config is not None:
        _, best_conn = best_valid_config
        b_ttft, b_tpot, b_e2e, b_throughput, b_out_thr = best_valid_metrics

        if not interrupted:
            logging.info(
                f"=== [SLO-Compliant] Optimal locked at MC={best_conn} "
                f"(Best {throughput_label}: {b_out_thr:.2f} tok/s) ==="
            )
            logging.info("Running final re-test with larger num_prompts for stable metrics...")
            f_ttft, f_tpot, f_e2e, f_throughput, f_out_thr, _, _ = run_benchmark(
                input_len,
                output_len,
                "inf",
                best_conn,
                num_prompts_multiplier=4,
                decode_only=True,
                decode_suffix_len=decode_suffix_len,
                warmup_prompts=0,  # already warmed up
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                gsp_system_prompt_len=gsp_system_prompt_len,
                num_prompts_cap=512,
            )
            final_ok = (
                f_tpot is not None and f_out_thr is not None and f_tpot <= slo_tpot
            )
            if final_ok:
                logging.info(
                    f"Final re-test: TPOT={f_tpot:.2f}ms, {throughput_label}={f_out_thr:.2f}tok/s"
                )
                final_metrics = (f_ttft, f_tpot, f_e2e, f_throughput, f_out_thr)
            else:
                logging.warning("Final re-test failed/violated SLO. Falling back to exploration metrics.")
                final_metrics = (b_ttft, b_tpot, b_e2e, b_throughput, b_out_thr)
        else:
            final_metrics = (b_ttft, b_tpot, b_e2e, b_throughput, b_out_thr)

        result["slo"] = {
            "best_req_rate": "inf",
            "best_max_conn": best_conn,
            "final_ttft": final_metrics[0],
            "final_tpot": final_metrics[1],
            "final_e2e": final_metrics[2],
            "final_throughput": final_metrics[3],
            "final_out_throughput": final_metrics[4],
            "final_in_throughput": (final_metrics[3] - final_metrics[4])
                if (final_metrics[3] and final_metrics[4]) else None,
            "final_objective_throughput": final_metrics[4],
        }
    else:
        logging.warning("No SLO-compliant configuration found in decode-only search.")

    # --- Max-Throughput track (any TPOT) ---
    if best_max_config is not None:
        _, max_conn_val = best_max_config
        m_ttft, m_tpot, m_e2e, m_throughput, m_out_thr = best_max_metrics
        max_final_metrics = best_max_metrics
        if result["slo"] is not None and best_max_config == best_valid_config:
            max_final_metrics = (
                result["slo"]["final_ttft"],
                result["slo"]["final_tpot"],
                result["slo"]["final_e2e"],
                result["slo"]["final_throughput"],
                result["slo"]["final_out_throughput"],
            )
        slo_ok = m_tpot is not None and m_tpot <= slo_tpot
        logging.info(
            f"=== [Max-Throughput] Highest {throughput_label}: {m_out_thr:.2f} tok/s "
            f"at MC={max_conn_val} (SLO compliant: {slo_ok}) ==="
        )
        mf_ttft, mf_tpot, mf_e2e, mf_throughput, mf_out_thr = max_final_metrics
        result["max"] = {
            "best_req_rate": "inf",
            "best_max_conn": max_conn_val,
            "final_ttft": mf_ttft,
            "final_tpot": mf_tpot,
            "final_e2e": mf_e2e,
            "final_throughput": mf_throughput,
            "final_out_throughput": mf_out_thr,
            "final_in_throughput": (mf_throughput - mf_out_thr)
                if (mf_throughput and mf_out_thr) else None,
            "final_objective_throughput": mf_out_thr,
            "slo_compliant": bool(slo_ok),
        }

    result["interrupted"] = interrupted
    return result if (result["slo"] or result["max"] or result["history"]) else None


def _find_optimal_prefill_cache(
    input_len,
    output_len,
    slo_ttft,
    slo_tpot,
    dp,
    dataset_name,
    dataset_path,
    max_kv_pool_size,
    metrics_url,
    page_size,
):
    """Search prefill-cache throughput with RR fixed at infinity.

    ``bench_serving`` uses ``max_concurrency`` as a client-side semaphore, so
    an infinite request rate keeps every concurrency slot occupied without
    allowing more than MC requests to reach the service at once.  Request rate
    is therefore redundant for this saturated-cache workload; only MC needs to
    be searched.
    """
    throughput_label = "Cache hit input throughput"
    logging.info(
        "=== Prefill-Cache MC-Only Search for "
        f"Input: {input_len}, Output: {output_len} ==="
    )
    logging.info(
        f"SLO Target - TTFT: {slo_ttft}ms, TPOT: {slo_tpot}ms; "
        "request_rate is fixed at inf."
    )

    pool_size_source = "explicit" if max_kv_pool_size is not None else None
    pool_metrics = None
    if max_kv_pool_size is None:
        pool_metrics = _read_kv_pool_size_from_metrics(metrics_url=metrics_url)
        if pool_metrics is not None:
            discovered_pool_size = pool_metrics["max_kv_pool_size"]
            discovered_usable_pool_size = _prefill_cache_usable_kv_tokens(
                max_kv_pool_size=discovered_pool_size,
                context_len=input_len,
                page_size=page_size,
            )
            minimum_estimated_kv_tokens = max(
                1, int(input_len) + max(0, int(output_len))
            )
            if discovered_usable_pool_size >= minimum_estimated_kv_tokens:
                max_kv_pool_size = discovered_pool_size
                pool_size_source = "metrics"
                logging.info(
                    "Prefill-cache KV pool discovered from metrics: "
                    f"effective={max_kv_pool_size}, "
                    f"usable={discovered_usable_pool_size}, "
                    f"device={pool_metrics['device_total_tokens']}, "
                    f"hicache_host={pool_metrics['hicache_host_total_tokens']}, "
                    f"url={pool_metrics['metrics_url']}."
                )
            else:
                skip_reason = (
                    f"usable KV pool {discovered_usable_pool_size} cannot hold "
                    f"one estimated {minimum_estimated_kv_tokens}-token "
                    "prefill-cache working-set item"
                )
                logging.warning(
                    "Metrics-reported KV pool is too small for the minimum "
                    f"1-request working set ({discovered_usable_pool_size} "
                    "usable < "
                    f"{minimum_estimated_kv_tokens}); skipping this input "
                    "length instead of producing invalid cold/hot data."
                )
                return {
                    "input_len": input_len,
                    "output_len": output_len,
                    "mode": "prefill-cache",
                    "dataset_name": dataset_name,
                    "max_kv_pool_size": discovered_pool_size,
                    "max_kv_pool_size_source": "metrics",
                    "kv_pool_metrics": pool_metrics,
                    "prefill_cache_usable_kv_tokens": discovered_usable_pool_size,
                    "prefill_cache_kv_page_size": page_size,
                    "slo": None,
                    "max": None,
                    "history": [],
                    "interrupted": False,
                    "skip_reason": skip_reason,
                }
        if max_kv_pool_size is None:
            pool_size_source = "per-mc-fallback"
            logging.info(
                "No usable KV-pool capacity was discovered; using per-MC "
                "cold-fill/hit boundary testing."
            )

    cache_session_seed = None
    cache_session_run_index = 0
    pool_info = None
    pool_size = None
    pool_readiness_info = None
    pool_init_attempts = 0
    pool_readiness_attempted = False
    if max_kv_pool_size is not None:
        cache_session_seed = _next_benchmark_seed(True)
        pool_size_cap = None
        for pool_init_attempt in range(
            1, PREFILL_CACHE_POOL_MAX_INIT_ATTEMPTS + 1
        ):
            pool_init_attempts = pool_init_attempt
            pool_info = _warm_prefill_cache_pool(
                input_len=input_len,
                output_len=output_len,
                max_kv_pool_size=max_kv_pool_size,
                benchmark_seed=cache_session_seed,
                max_conn=max(4, int(dp)),
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                page_size=page_size,
                pool_size_cap=pool_size_cap,
            )
            if pool_info is None:
                break
            pool_readiness_attempted = True
            pool_readiness_info = _verify_prefill_cache_pool_ready(
                input_len=input_len,
                output_len=output_len,
                pool_size=pool_info["pool_size"],
                benchmark_seed=cache_session_seed,
                max_conn=max(4, int(dp)),
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                page_size=page_size,
            )
            pool_info["readiness"] = pool_readiness_info
            if pool_readiness_info["ready"]:
                break

            current_pool_size = pool_info["pool_size"]
            if pool_init_attempt >= PREFILL_CACHE_POOL_MAX_INIT_ATTEMPTS:
                pool_info = None
                break
            last_hit_rate = pool_readiness_info["cache_hit_rate"]
            if last_hit_rate is not None and last_hit_rate > 0:
                shrink_ratio = min(
                    0.90,
                    max(
                        0.50,
                        last_hit_rate
                        / pool_readiness_info["cache_hit_rate_floor"]
                        * 0.95,
                    ),
                )
            else:
                shrink_ratio = 0.90
            adjusted_pool_size = max(
                10,
                min(current_pool_size - 1, int(current_pool_size * shrink_ratio)),
            )
            if adjusted_pool_size >= current_pool_size:
                pool_info = None
                break
            logging.warning(
                "Full-pool readiness validation failed; shrinking the "
                f"working set from {current_pool_size} to "
                f"{adjusted_pool_size}, then flushing and cold-filling it "
                "again before optimization."
            )
            pool_size_cap = adjusted_pool_size

        if pool_info is None:
            if pool_size_source == "metrics" and not pool_readiness_attempted:
                logging.warning(
                    "Metrics-derived prefill-cache pool could not be cold-filled; "
                    "falling back to per-MC cold-fill/hit boundary testing."
                )
                max_kv_pool_size = None
                pool_size_source = "per-mc-fallback"
                pool_metrics = None
                cache_session_seed = None
            else:
                logging.error(
                    "Unable to initialize and verify a fully cache-ready "
                    "prefill-cache pool; aborting this input length instead of "
                    "starting MC optimization with an unverified working set."
                )
                return None
        else:
            pool_size = pool_info["pool_size"]

    cache_fill_policy = (
        "per-MC cold-fill/hit pair"
        if pool_size is None
        else (
            f"max-kv-pool={max_kv_pool_size} tokens, pool_size={pool_size}, "
            f"usable-kv-pool={pool_info['usable_kv_pool_size']} tokens, "
            f"page-size={page_size}, "
            f"full-pool-ready-cache-hit="
            f"{pool_readiness_info['cache_hit_rate']:.2f}%, "
            "fixed-seed full-pool hit-only search"
        )
    )
    logging.info(
        "Prefill-cache search policy: RR=inf, MC exponential ramp + "
        f"boundary bisection, {cache_fill_policy}, cache-hit floor="
        f"{PREFILL_CACHE_MIN_HIT_RATE:.0f}%, no generated-shared-prefix "
        "request repetition."
    )

    def num_prompts(mc):
        # Keep the generated-shared-prefix dataset definition identical for
        # every fixed-pool probe.  Changing num_prompts also changes
        # gsp_num_groups (and therefore the generated dataset/cache key), so a
        # same-seed MC probe is not guaranteed to replay the cold-filled
        # working set unless the full pool size remains fixed.  MC must be the
        # only independent variable during this search.
        if pool_size is not None:
            return pool_size
        count = max(10, int(mc) * PREFILL_CACHE_WORKING_SET_MULTIPLIER)
        return count

    def request_repeat_count(mc):
        # Do not use --gsp-repeat-count in prefill-cache mode. Repeating the
        # same generated-shared-prefix dataset inside one benchmark run changes
        # the measured request stream and has produced invalid cache reports at
        # higher MC. Cold fill and hot replay should use the same unique prompt
        # set exactly once; stability comes from explicit final replays.
        return 1

    def run_probe(mc, hit_repetitions=1):
        nonlocal cache_session_run_index
        session_kwargs = {}
        if pool_size is not None:
            cache_session_run_index += 1
            session_kwargs = {
                "benchmark_seed": cache_session_seed,
                "skip_cache_miss": True,
                "gsp_ordered": True,
                "num_prompts_override": num_prompts(mc),
                "gsp_repeat_count": request_repeat_count(mc),
                "hit_log_label": (
                    f"cache_pool_hit_seed{cache_session_seed}_"
                    f"run{cache_session_run_index}"
                ),
            }
        return run_prefill_cache_benchmark(
            input_len=input_len,
            output_len=output_len,
            req_rate="inf",
            max_conn=mc,
            num_prompts_multiplier=PREFILL_CACHE_WORKING_SET_MULTIPLIER,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            hit_repetitions=hit_repetitions,
            **session_kwargs,
        )

    mc_ceiling = 1024
    if pool_size is not None:
        mc_ceiling = min(mc_ceiling, max(1, int(pool_size)))
    all_results = []
    observations = {}
    best_valid_config = None
    best_valid_metrics = None
    best_max_config = None
    best_max_metrics = None
    best_valid_throughput = 0.0
    best_max_throughput = 0.0
    interrupted = False

    def probe(mc, phase, label):
        nonlocal best_valid_config, best_valid_metrics
        nonlocal best_max_config, best_max_metrics
        nonlocal best_valid_throughput, best_max_throughput

        mc = max(1, min(int(mc), mc_ceiling))
        if mc in observations:
            return observations[mc]

        unique_prompts = num_prompts(mc)
        repeat_count = request_repeat_count(mc)
        total_requests = unique_prompts * repeat_count

        logging.info(
            f"--- [{label}] RR=inf, MC={mc}, unique_prompts="
            f"{unique_prompts}, repeat_count={repeat_count}, "
            f"total_requests={total_requests} ---"
        )
        (
            hit_result,
            cache_miss_input_thr,
            cache_hit_input_thr,
            cache_hit_rate,
        ) = run_probe(mc)
        ttft, tpot, e2e_lat, throughput, out_thr, req_thr, _ = hit_result

        initial_cache_hit_rate = cache_hit_rate
        cache_probe_attempts = 1
        if (
            pool_size is not None
            and (
                cache_hit_rate is None
                or cache_hit_rate < PREFILL_CACHE_MIN_HIT_RATE
            )
        ):
            logging.warning(
                "First fixed-pool cache-hit sample is below the validity "
                f"floor at MC={mc} (CacheHit="
                f"{cache_hit_rate if cache_hit_rate is not None else 'N/A'}%); "
                "replaying the identical working set once before classifying "
                "the MC boundary."
            )
            (
                hit_result,
                cache_miss_input_thr,
                cache_hit_input_thr,
                cache_hit_rate,
            ) = run_probe(mc)
            cache_probe_attempts = 2
            ttft, tpot, e2e_lat, throughput, out_thr, req_thr, _ = hit_result
            if (
                cache_hit_rate is not None
                and cache_hit_rate >= PREFILL_CACHE_MIN_HIT_RATE
            ):
                logging.info(
                    "Transient first cache miss recovered on identical replay: "
                    f"MC={mc}, CacheHit={cache_hit_rate:.2f}%. Only the replay "
                    "is included in optimization."
                )
            else:
                logging.warning(
                    "Identical replay remained below the cache-hit floor at "
                    f"MC={mc}; treating it as a cache-capacity boundary."
                )

        metrics = (
            ttft,
            tpot,
            e2e_lat,
            throughput,
            out_thr,
            cache_miss_input_thr,
            cache_hit_input_thr,
            cache_hit_rate,
        )
        parsed = all(
            value is not None
            for value in (ttft, tpot, e2e_lat, cache_hit_input_thr)
        )
        cache_ok = bool(
            parsed
            and cache_hit_rate is not None
            and cache_hit_rate >= PREFILL_CACHE_MIN_HIT_RATE
        )
        slo_ok = bool(
            cache_ok and ttft <= slo_ttft and tpot <= slo_tpot
        )
        observation = {
            "mc": mc,
            "metrics": metrics,
            "parsed": parsed,
            "cache_ok": cache_ok,
            "slo_ok": slo_ok,
            "objective": cache_hit_input_thr if parsed else None,
            "cache_probe_attempts": cache_probe_attempts,
            "initial_cache_hit_rate": initial_cache_hit_rate,
            "unique_prompts": unique_prompts,
            "request_repeat_count": repeat_count,
            "total_requests": total_requests,
        }
        observations[mc] = observation
        all_results.append(
            {
                "iter": len(all_results) + 1,
                "phase": phase,
                "req_rate": "inf",
                "max_conn": mc,
                "num_prompts": total_requests,
                "unique_prompts": unique_prompts,
                "request_repeat_count": repeat_count,
                "total_requests": total_requests,
                "ttft": ttft,
                "tpot": tpot,
                "e2e": e2e_lat,
                "throughput": throughput,
                "out_throughput": out_thr,
                "in_throughput": cache_hit_input_thr,
                "cache_miss_input_throughput": cache_miss_input_thr,
                "cache_hit_input_throughput": cache_hit_input_thr,
                "cache_hit_rate": cache_hit_rate,
                "cache_probe_attempts": cache_probe_attempts,
                "initial_cache_hit_rate": initial_cache_hit_rate,
                "objective_throughput": cache_hit_input_thr,
                "slo_ok": slo_ok if parsed else None,
            }
        )

        if not parsed:
            logging.warning(f"Probe failed/unparseable at RR=inf, MC={mc}.")
            return observation

        logging.info(
            f"Probe Result: RR=inf, MC={mc}, TTFT={ttft:.2f}ms, "
            f"TPOT={tpot:.2f}ms, {throughput_label}="
            f"{cache_hit_input_thr:.2f}tok/s, CacheHit={cache_hit_rate:.2f}%"
            + (f", ReqThr={req_thr:.2f}req/s" if req_thr else "")
        )
        if cache_ok and cache_hit_input_thr > best_max_throughput:
            logging.info(
                f"New Max {throughput_label}! {cache_hit_input_thr:.2f} "
                f"tok/s (was {best_max_throughput:.2f}) at MC={mc}"
            )
            best_max_throughput = cache_hit_input_thr
            best_max_config = ("inf", mc)
            best_max_metrics = metrics
        if slo_ok and cache_hit_input_thr > best_valid_throughput:
            logging.info(
                f"New Best SLO-Compliant {throughput_label}! "
                f"{cache_hit_input_thr:.2f} tok/s "
                f"(was {best_valid_throughput:.2f}) at MC={mc}"
            )
            best_valid_throughput = cache_hit_input_thr
            best_valid_config = ("inf", mc)
            best_valid_metrics = metrics
        return observation

    last_good_mc = None
    first_bad_mc = None
    try:
        # Exponentially find the SLO/cache boundary.  MC may exceed the unique
        # pool size because fixed requests are repeated in-place.
        mc = min(4, mc_ceiling)
        for _ in range(12):
            observation = probe(mc, 1, "MC Doubling")
            if not observation["slo_ok"]:
                first_bad_mc = mc
                break
            last_good_mc = mc
            if mc >= mc_ceiling:
                break
            mc = min(mc_ceiling, mc * 2)

        # If the initial MC is already invalid, search downward for a usable
        # low endpoint before attempting the boundary bisection.
        if first_bad_mc is not None and last_good_mc is None:
            mc = first_bad_mc // 2
            while mc >= 1:
                observation = probe(mc, 1, "MC Backoff")
                if observation["slo_ok"]:
                    last_good_mc = mc
                    break
                if mc == 1:
                    break
                mc = max(1, mc // 2)

        if (
            last_good_mc is not None
            and first_bad_mc is not None
            and first_bad_mc - last_good_mc > 1
        ):
            low, high = last_good_mc, first_bad_mc
            logging.info(
                f"=== MC boundary bisection: MC={low} OK, MC={high} BAD ==="
            )
            for _ in range(PREFILL_CACHE_BOUNDARY_MAX_RUNS):
                if high - low <= 1:
                    break
                mid = (low + high) // 2
                observation = probe(mid, 2, "MC Bisection")
                if observation["slo_ok"]:
                    low = mid
                else:
                    high = mid
                time.sleep(0.5)

        # Refine around the best observed cache-hit throughput.  This keeps the
        # exponential search fast while still sampling non-power-of-two MCs
        # when the throughput peak is inside a broad bracket.
        for _ in range(PREFILL_CACHE_BOUNDARY_MAX_RUNS):
            if best_max_config is None:
                break
            best_mc = best_max_config[1]
            tested = sorted(observations)
            left = max((value for value in tested if value < best_mc), default=None)
            right = min((value for value in tested if value > best_mc), default=None)
            intervals = []
            if left is not None and best_mc - left > 1:
                intervals.append((best_mc - left, left, best_mc))
            if right is not None and right - best_mc > 1:
                intervals.append((right - best_mc, best_mc, right))
            if not intervals:
                break
            _, low, high = max(intervals)
            mid = (low + high) // 2
            probe(mid, 3, "MC Peak Refinement")
            time.sleep(0.5)
    except KeyboardInterrupt:
        logging.info("\nUser interrupted. Returning results collected so far...")
        interrupted = True

    result = {
        "input_len": input_len,
        "output_len": output_len,
        "mode": "prefill-cache",
        "dataset_name": dataset_name,
        "max_kv_pool_size": max_kv_pool_size,
        "max_kv_pool_size_source": pool_size_source,
        "kv_pool_metrics": pool_metrics,
        "prefill_cache_pool_size": pool_size,
        "prefill_cache_usable_kv_tokens": (
            pool_info["usable_kv_pool_size"] if pool_info is not None else None
        ),
        "prefill_cache_kv_page_size": page_size,
        "prefill_cache_ready_validation_attempts": (
            pool_readiness_info["attempts"]
            if pool_readiness_info is not None
            else None
        ),
        "prefill_cache_ready_cache_hit_rate": (
            pool_readiness_info["cache_hit_rate"]
            if pool_readiness_info is not None
            else None
        ),
        "prefill_cache_ready_cache_hit_rate_floor": (
            pool_readiness_info["cache_hit_rate_floor"]
            if pool_readiness_info is not None
            else None
        ),
        "prefill_cache_pool_init_attempts": pool_init_attempts,
        "prefill_cache_pool_refills": max(0, pool_init_attempts - 1),
        "prefill_cache_min_request_waves": PREFILL_CACHE_MIN_REQUEST_WAVES,
        "prefill_cache_measured_kv_tokens": (
            pool_info["measured_kv_tokens"] if pool_info is not None else None
        ),
        "slo": None,
        "max": None,
        "history": all_results,
        "interrupted": interrupted,
    }

    if best_valid_config is not None:
        _, best_conn = best_valid_config
        final_metrics = best_valid_metrics
        if not interrupted:
            logging.info(
                "=== [SLO-Compliant] Optimal locked at req_rate=inf, "
                f"max_concurrency={best_conn} (Best {throughput_label}: "
                f"{best_valid_throughput:.2f}) ==="
            )
            logging.info(
                "Running final fixed-workset cache-hit replay "
                f"{PREFILL_CACHE_FINAL_HIT_REPETITIONS} times..."
            )
            (
                final_hit_result,
                final_miss_input_thr,
                final_hit_input_thr,
                final_hit_rate,
            ) = run_probe(
                best_conn,
                hit_repetitions=PREFILL_CACHE_FINAL_HIT_REPETITIONS,
            )
            final_candidate = (
                final_hit_result[0],
                final_hit_result[1],
                final_hit_result[2],
                final_hit_result[3],
                final_hit_result[4],
                final_miss_input_thr,
                final_hit_input_thr,
                final_hit_rate,
            )
            final_ok = (
                all(value is not None for value in final_candidate[:3])
                and final_hit_input_thr is not None
                and final_hit_rate is not None
                and final_candidate[0] <= slo_ttft
                and final_candidate[1] <= slo_tpot
                and final_hit_rate >= PREFILL_CACHE_MIN_HIT_RATE
            )
            if final_ok:
                final_metrics = final_candidate
                logging.info(
                    "Final cache-hit replay passed: "
                    f"TTFT={final_candidate[0]:.2f}ms, "
                    f"TPOT={final_candidate[1]:.2f}ms, "
                    f"{throughput_label}={final_hit_input_thr:.2f}tok/s, "
                    f"CacheHit={final_hit_rate:.2f}%."
                )
            else:
                logging.warning(
                    "Final fixed-workset replay failed validity constraints; "
                    "using the successful exploration metrics."
                )

        result["slo"] = {
            "best_req_rate": "inf",
            "best_max_conn": best_conn,
            "best_unique_prompts": num_prompts(best_conn),
            "best_request_repeat_count": request_repeat_count(best_conn),
            "best_total_requests": (
                num_prompts(best_conn) * request_repeat_count(best_conn)
            ),
            "final_ttft": final_metrics[0],
            "final_tpot": final_metrics[1],
            "final_e2e": final_metrics[2],
            "final_throughput": final_metrics[3],
            "final_out_throughput": final_metrics[4],
            "final_in_throughput": final_metrics[6],
            "final_cache_miss_input_throughput": final_metrics[5],
            "final_cache_hit_input_throughput": final_metrics[6],
            "final_cache_hit_rate": final_metrics[7],
            "final_objective_throughput": final_metrics[6],
        }
    else:
        logging.warning("Could not find any prefill-cache configuration that satisfies the SLO.")

    if best_max_config is not None:
        _, best_conn = best_max_config
        metrics = best_max_metrics
        max_slo_ok = bool(
            metrics[0] is not None
            and metrics[1] is not None
            and metrics[0] <= slo_ttft
            and metrics[1] <= slo_tpot
            and metrics[7] is not None
            and metrics[7] >= PREFILL_CACHE_MIN_HIT_RATE
        )
        logging.info(
            f"=== [Max-Throughput] Highest {throughput_label}: "
            f"{metrics[6]:.2f} tok/s at req_rate=inf, max_concurrency={best_conn} "
            f"(SLO compliant: {max_slo_ok}) ==="
        )
        result["max"] = {
            "best_req_rate": "inf",
            "best_max_conn": best_conn,
            "best_unique_prompts": num_prompts(best_conn),
            "best_request_repeat_count": request_repeat_count(best_conn),
            "best_total_requests": (
                num_prompts(best_conn) * request_repeat_count(best_conn)
            ),
            "final_ttft": metrics[0],
            "final_tpot": metrics[1],
            "final_e2e": metrics[2],
            "final_throughput": metrics[3],
            "final_out_throughput": metrics[4],
            "final_in_throughput": metrics[6],
            "final_cache_miss_input_throughput": metrics[5],
            "final_cache_hit_input_throughput": metrics[6],
            "final_cache_hit_rate": metrics[7],
            "final_objective_throughput": metrics[6],
            "slo_compliant": max_slo_ok,
        }

    return result if (result["slo"] or result["max"] or result["history"]) else None


def find_optimal_throughput(
    input_len,
    output_len,
    slo_ttft,
    slo_tpot,
    dp=1,
    decode_only=False,
    prefill_only=False,
    prefill_cache=False,
    decode_suffix_len=DECODE_SUFFIX_LEN,
    warmup_prompts=DECODE_WARMUP_PROMPTS,
    dataset_name="random",
    dataset_path=None,
    gsp_system_prompt_len=None,
    max_kv_pool_size=None,
    prefill_cache_metrics_url=None,
    kv_page_size=PREFILL_CACHE_KV_PAGE_SIZE,
    decode_kv_pool_tokens=994816,
):
    """
    Search for the optimal request_rate and max_concurrency using a coupled 2D search logic.
    dp: Data Parallelism degree. Standard/decode-only concurrency is aligned to
        dp, while prefill modes can use any positive integer.
    """
    kv_page_size = int(kv_page_size)
    if kv_page_size <= 0:
        raise ValueError("kv_page_size must be positive")
    if prefill_cache and int(input_len) <= kv_page_size:
        raise ValueError("input_len must be greater than kv_page_size")

    if max_kv_pool_size is not None:
        if not prefill_cache:
            raise ValueError(
                "max_kv_pool_size requires prefill_cache=True"
            )
        max_kv_pool_size = int(max_kv_pool_size)
        minimum_estimated_kv_tokens = 10 * max(
            1, int(input_len) + max(0, int(output_len))
        )
        usable_kv_pool_size = _prefill_cache_usable_kv_tokens(
            max_kv_pool_size=max_kv_pool_size,
            context_len=input_len,
            page_size=kv_page_size,
        )
        if usable_kv_pool_size < minimum_estimated_kv_tokens:
            raise ValueError(
                f"max_kv_pool_size must provide at least "
                f"{minimum_estimated_kv_tokens} usable KV tokens for 10 "
                "estimated requests after page headroom"
            )

    if decode_only:
        return _find_optimal_decode_only(
            input_len=input_len,
            output_len=output_len,
            slo_tpot=slo_tpot,
            dp=dp,
            decode_suffix_len=decode_suffix_len,
            warmup_prompts=warmup_prompts,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            gsp_system_prompt_len=gsp_system_prompt_len,
            decode_kv_pool_tokens=decode_kv_pool_tokens,
        )

    if prefill_cache and dataset_name != "generated-shared-prefix":
        raise ValueError(
            "prefill-cache mode requires dataset_name=generated-shared-prefix"
        )

    if prefill_cache:
        return _find_optimal_prefill_cache(
            input_len=input_len,
            output_len=output_len,
            slo_ttft=slo_ttft,
            slo_tpot=slo_tpot,
            dp=dp,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            max_kv_pool_size=max_kv_pool_size,
            metrics_url=prefill_cache_metrics_url,
            page_size=kv_page_size,
        )

    mc_granularity = 1 if (prefill_only or prefill_cache) else max(1, int(dp))

    def _snap_mc(value):
        if mc_granularity > 1:
            return int(
                max(
                    mc_granularity,
                    round(float(value) / mc_granularity) * mc_granularity,
                )
            )
        return max(1, int(round(value)))

    req_rate = 1.0
    max_conn = _snap_mc(4)
    
    # Two tracks:
    # 1. SLO-compliant best (must satisfy TTFT/TPOT SLOs)
    # 2. Max throughput best (may violate SLOs, just tracks the highest throughput observed)
    best_valid_config = None
    best_valid_metrics = None
    best_max_config = None
    best_max_metrics = None
    
    mode_name = (
        "Prefill-Cache"
        if prefill_cache
        else ("Prefill-Only" if prefill_only else "Standard")
    )
    logging.info(f"=== Phase 1: SLO-Compliant Search [{mode_name}] for Input: {input_len}, Output: {output_len} ===")
    logging.info(f"SLO Target - TTFT: {slo_ttft}ms, TPOT: {slo_tpot}ms")
    if prefill_cache:
        if max_kv_pool_size is None:
            cache_fill_policy = "per-config cold-fill/hit pair"
        else:
            cache_fill_policy = (
                f"max-kv-pool={max_kv_pool_size} tokens, "
                "fixed-seed hit-only search"
            )
        logging.info(
            "Prefill-cache search policy: "
            f"num_prompts={PREFILL_CACHE_WORKING_SET_MULTIPLIER} * MC "
            f"(minimum 10), {cache_fill_policy}, "
            f"cache-hit floor={PREFILL_CACHE_MIN_HIT_RATE:.0f}%, "
            f"adaptive MC probe budget={PREFILL_CACHE_BOUNDARY_MAX_RUNS}."
        )
    
    throughput_label = (
        "Cache hit input throughput"
        if prefill_cache
        else "Total token throughput"
    )

    def objective_throughput(total_thr, out_thr, input_thr=None):
        return input_thr if prefill_cache else total_thr

    def effective_prompt_multiplier(requested_multiplier):
        # Keep the prefill-cache working set independent of the search phase.
        # Mixing 1x local probes with 2x main probes makes MC comparisons
        # invalid and can accidentally cross the KV-cache capacity cliff.
        if prefill_cache:
            return PREFILL_CACHE_WORKING_SET_MULTIPLIER
        return requested_multiplier

    def search_num_prompts(probe_mc, requested_multiplier=2):
        num_prompts = max(
            10,
            int(probe_mc) * effective_prompt_multiplier(requested_multiplier),
        )
        if prefill_cache_pool_size is not None:
            num_prompts = min(num_prompts, prefill_cache_pool_size)
        return num_prompts

    cache_session_seed = None
    cache_session_run_index = 0
    prefill_cache_pool_size = None
    prefill_cache_pool_info = None
    if max_kv_pool_size is not None:
        cache_session_seed = _next_benchmark_seed(True)
        prefill_cache_pool_info = _warm_prefill_cache_pool(
            input_len=input_len,
            output_len=output_len,
            max_kv_pool_size=max_kv_pool_size,
            benchmark_seed=cache_session_seed,
            max_conn=max(4, int(dp)),
            dataset_name=dataset_name,
            dataset_path=dataset_path,
        )
        if prefill_cache_pool_info is None:
            logging.error(
                "Unable to initialize the prefill-cache pool; aborting this "
                "search instead of reporting mixed cache state."
            )
            return None
        prefill_cache_pool_size = prefill_cache_pool_info["pool_size"]

    def _run_search_benchmark(
        probe_rr,
        probe_mc,
        num_prompts_multiplier,
        hit_repetitions=1,
    ):
        nonlocal cache_session_run_index
        num_prompts_multiplier = effective_prompt_multiplier(
            num_prompts_multiplier
        )
        if prefill_cache:
            cache_session_kwargs = {}
            if max_kv_pool_size is not None:
                cache_session_run_index += 1
                cache_session_kwargs = {
                    "benchmark_seed": cache_session_seed,
                    "skip_cache_miss": True,
                    "gsp_ordered": True,
                    "num_prompts_override": search_num_prompts(
                        probe_mc, num_prompts_multiplier
                    ),
                    "hit_log_label": (
                        f"cache_pool_hit_seed{cache_session_seed}_"
                        f"run{cache_session_run_index}"
                    ),
                }
            (
                hit_result,
                miss_input_thr,
                hit_input_thr,
                cache_hit_rate,
            ) = run_prefill_cache_benchmark(
                input_len=input_len,
                output_len=output_len,
                req_rate=probe_rr,
                max_conn=probe_mc,
                num_prompts_multiplier=num_prompts_multiplier,
                dataset_name=dataset_name,
                dataset_path=dataset_path,
                hit_repetitions=hit_repetitions,
                **cache_session_kwargs,
            )
            return (*hit_result, miss_input_thr, hit_input_thr, cache_hit_rate)

        benchmark_result = run_benchmark(
            input_len,
            output_len,
            probe_rr,
            probe_mc,
            num_prompts_multiplier=num_prompts_multiplier,
            decode_only=False,
            prefill_only=prefill_only,
            decode_suffix_len=decode_suffix_len,
            warmup_prompts=0,
            dataset_name=dataset_name,
            dataset_path=dataset_path,
            gsp_system_prompt_len=gsp_system_prompt_len,
        )
        input_thr = _extract_input_throughput(
            benchmark_result[6], benchmark_result[3], benchmark_result[4]
        )
        return (*benchmark_result, None, input_thr, None)

    # We use a greedy frontier search prioritizing the active mode's throughput objective.
    consecutive_failures = 0
    max_iterations = 45 if decode_only else 40
    mc_ceiling = 4096 if decode_only else 1024  # Start with a high ceiling
    if prefill_cache_pool_size is not None:
        mc_ceiling = min(mc_ceiling, prefill_cache_pool_size)
    rr_ceiling = 1000.0 if decode_only else 100.0  # Start with a high ceiling for request rate
    best_overall_throughput = 0.0
    best_max_throughput = 0.0
    all_results = []  # Collect every iteration's results for the final report
    visited_configs = set()
    boundary_probe_done = False
    interrupted = False
    SLEEP_SEC = 0.5
    LOCAL_PROBE_MAX_RUNS = PREFILL_CACHE_BOUNDARY_MAX_RUNS

    def _normalize_rr(value):
        return max(0.2, round(round(float(value) * 5) / 5, 1))

    def _coarse_mc_step(value):
        step = max(4, int(math.ceil(float(value) * 0.10)))
        alignment = mc_granularity if mc_granularity > 1 else 4
        return max(
            alignment,
            int(math.ceil(step / alignment)) * alignment,
        )

    def _record_probe(
        phase,
        probe_rr,
        probe_mc,
        ttft,
        tpot,
        e2e_lat,
        throughput,
        out_thr,
        req_thr=None,
        cache_miss_input_thr=None,
        cache_hit_input_thr=None,
        cache_hit_rate=None,
    ):
        nonlocal best_max_throughput, best_max_config, best_max_metrics
        nonlocal best_overall_throughput, best_valid_config, best_valid_metrics

        obj_thr = objective_throughput(
            throughput, out_thr, cache_hit_input_thr
        )
        ttft_ok = None if ttft is None else (True if decode_only else ttft <= slo_ttft)
        tpot_ok = None if tpot is None else tpot <= slo_tpot
        cache_ok = (
            None
            if prefill_cache and cache_hit_rate is None
            else (
                not prefill_cache
                or cache_hit_rate >= PREFILL_CACHE_MIN_HIT_RATE
            )
        )
        slo_ok = (
            ttft_ok and tpot_ok and cache_ok
            if (
                ttft_ok is not None
                and tpot_ok is not None
                and cache_ok is not None
            )
            else None
        )
        in_thr = cache_hit_input_thr
        all_results.append({
            "iter": len(all_results) + 1, "phase": phase,
            "req_rate": probe_rr, "max_conn": probe_mc,
            "num_prompts": search_num_prompts(probe_mc),
            "ttft": ttft, "tpot": tpot, "e2e": e2e_lat,
            "throughput": throughput, "out_throughput": out_thr, "in_throughput": in_thr,
            "cache_miss_input_throughput": cache_miss_input_thr,
            "cache_hit_input_throughput": cache_hit_input_thr if prefill_cache else None,
            "cache_hit_rate": cache_hit_rate if prefill_cache else None,
            "objective_throughput": obj_thr,
            "slo_ok": slo_ok,
        })

        if obj_thr is None:
            return None, None, None, None, None

        if cache_ok and obj_thr > best_max_throughput:
            logging.info(f"New Max {throughput_label} (any SLO)! {obj_thr:.2f} tok/s (was {best_max_throughput:.2f})")
            best_max_throughput = obj_thr
            best_max_config = (probe_rr, probe_mc)
            best_max_metrics = (
                ttft, tpot, e2e_lat, throughput, out_thr,
                cache_miss_input_thr, cache_hit_input_thr, cache_hit_rate,
            )

        if slo_ok and obj_thr >= best_overall_throughput:
            logging.info(f"New Best SLO-Compliant {throughput_label}! {obj_thr:.2f} tok/s (was {best_overall_throughput:.2f})")
            best_overall_throughput = obj_thr
            best_valid_config = (probe_rr, probe_mc)
            best_valid_metrics = (
                ttft, tpot, e2e_lat, throughput, out_thr,
                cache_miss_input_thr, cache_hit_input_thr, cache_hit_rate,
            )

        return obj_thr, ttft_ok, tpot_ok, req_thr, cache_ok

    def _run_phase1_probe(probe_rr, probe_mc, prompt_multiplier=None):
        probe_rr = _normalize_rr(probe_rr)
        probe_mc = max(1, int(round(probe_mc)))
        cfg_key = (probe_rr, probe_mc)
        if cfg_key in visited_configs:
            return None, None, None, None, None
        visited_configs.add(cfg_key)

        (
            ttft, tpot, e2e_lat, throughput, out_thr, req_thr, _,
            cache_miss_input_thr, cache_hit_input_thr, cache_hit_rate,
        ) = _run_search_benchmark(
            probe_rr,
            probe_mc,
            effective_prompt_multiplier(prompt_multiplier or 2),
        )
        obj_thr, ttft_ok, tpot_ok, _, cache_ok = _record_probe(
            1, probe_rr, probe_mc, ttft, tpot, e2e_lat, throughput, out_thr,
            req_thr, cache_miss_input_thr, cache_hit_input_thr, cache_hit_rate,
        )
        if obj_thr is None:
            logging.warning(f"Probe failed/unparseable at RR={probe_rr:.2f}, MC={probe_mc}.")
        else:
            logging.info(
                f"Probe Result: RR={probe_rr:.2f}, MC={probe_mc}, TTFT={ttft:.2f}ms, "
                f"TPOT={tpot:.2f}ms, {throughput_label}={obj_thr:.2f}tok/s"
                + (
                    f", CacheHit={cache_hit_rate:.2f}%"
                    if prefill_cache and cache_hit_rate is not None
                    else ""
                )
                + (f", ReqThr={req_thr:.2f}req/s" if req_thr else "")
            )
        return obj_thr, ttft_ok, tpot_ok, req_thr, cache_ok

    def _probe_ttft_boundary(safe_rr, safe_mc, failed_rr, failed_mc):
        """Bracket and bisect MC at the failed RR instead of scanning by one."""
        if decode_only:
            return None

        target_rr = _normalize_rr(failed_rr)
        # failed_mc is the known-invalid high endpoint. If the latest known
        # good MC is not below it, create a coarse lower endpoint first.
        mc_high = failed_mc
        mc_low = safe_mc
        if mc_low > mc_high:
            mc_low = mc_high
        if mc_low == mc_high:
            mc_low = max(1, mc_high - _coarse_mc_step(mc_high))

        logging.info(
            f"=== Adaptive MC-boundary probe: RR={target_rr:.2f}, "
            f"known_bad_MC={mc_high}, initial_low_MC={mc_low}, "
            f"max_runs={LOCAL_PROBE_MAX_RUNS} ==="
        )
        probe_runs = 0
        low_result = None
        low_is_known_good = (
            _normalize_rr(safe_rr) == target_rr
            and safe_mc == mc_low
            and safe_mc < failed_mc
        )
        if low_is_known_good:
            low_ttft_ok = True
            low_tpot_ok = True
            low_cache_ok = True
            low_req_thr = None
        else:
            low_result = _run_phase1_probe(target_rr, mc_low)
        if low_result is not None and low_result[0] is None:
            # The low endpoint can already be visited when safe/failed MC are
            # equal. Try one coarser point rather than falling back to MC+1.
            retry_low = max(1, mc_low - _coarse_mc_step(mc_low))
            if retry_low != mc_low:
                mc_low = retry_low
                low_result = _run_phase1_probe(target_rr, mc_low)
        if not low_is_known_good and (
            low_result is None or low_result[0] is None
        ):
            logging.info("Adaptive MC probe found no unvisited low endpoint.")
            return None

        if not low_is_known_good:
            probe_runs += 1
            (
                _, low_ttft_ok, low_tpot_ok, low_req_thr, low_cache_ok
            ) = low_result
        low_ok = bool(low_ttft_ok and low_tpot_ok and low_cache_ok)
        if not low_ok:
            logging.info(
                f"Adaptive MC probe: MC={mc_low} is still invalid at "
                f"RR={target_rr:.2f}; no valid MC bracket at this RR."
            )
            return None
        if low_req_thr is not None and low_req_thr < target_rr * 0.5:
            logging.info(
                f"Adaptive MC probe: actual request throughput {low_req_thr:.2f} "
                f"req/s << requested {target_rr:.2f}; stopping."
            )
            return target_rr, mc_low

        best_good_mc = mc_low
        while mc_high - mc_low > max(1, mc_granularity):
            if probe_runs >= LOCAL_PROBE_MAX_RUNS:
                break
            mid = (mc_low + mc_high) // 2
            if mc_granularity > 1:
                mid = max(
                    mc_granularity,
                    round(mid / mc_granularity) * mc_granularity,
                )
            if mid <= mc_low or mid >= mc_high:
                break

            mid_result = _run_phase1_probe(target_rr, mid)
            if mid_result[0] is None:
                # A visited midpoint cannot narrow the bracket reliably.
                break
            probe_runs += 1
            _, mid_ttft_ok, mid_tpot_ok, _, mid_cache_ok = mid_result
            mid_ok = bool(mid_ttft_ok and mid_tpot_ok and mid_cache_ok)
            if mid_ok:
                mc_low = mid
                best_good_mc = mid
            else:
                mc_high = mid
            time.sleep(SLEEP_SEC)

        logging.info(
            f"Adaptive MC probe converged: RR={target_rr:.2f}, "
            f"best_good_MC={best_good_mc}, first_bad_MC<={mc_high}, "
            f"runs={probe_runs}."
        )
        return target_rr, best_good_mc
    
    try:
        for iteration in range(1, max_iterations + 1):
            # Snap req_rate to nearest 0.2 grid to avoid over-precision (e.g. 3.0175...)
            req_rate = round(round(req_rate * 5) / 5, 1)
            req_rate = max(0.2, req_rate)
            
            max_conn = _snap_mc(max_conn)
            cfg_key = (req_rate, max_conn)
            if cfg_key in visited_configs:
                logging.info(
                    f"Search converged before repeating visited config "
                    f"RR={req_rate:.2f}, MC={max_conn}."
                )
                break
            visited_configs.add(cfg_key)
            
            logging.info(f"--- Iteration {iteration} (RR={req_rate:.2f}, MC={max_conn}) ---")
            (
                ttft, tpot, e2e_lat, throughput, out_thr, req_thr, _,
                cache_miss_input_thr, cache_hit_input_thr, cache_hit_rate,
            ) = _run_search_benchmark(
                req_rate,
                max_conn,
                effective_prompt_multiplier(2),
            )
            
            obj_thr = objective_throughput(
                throughput, out_thr, cache_hit_input_thr
            )
            if ttft is None or tpot is None or e2e_lat is None or obj_thr is None:
                # Record failed iteration
                all_results.append({
                    "iter": len(all_results) + 1, "phase": 1, "req_rate": req_rate, "max_conn": max_conn,
                    "num_prompts": search_num_prompts(max_conn),
                    "ttft": None, "tpot": None, "e2e": None,
                    "throughput": None, "out_throughput": None, "in_throughput": None,
                    "cache_miss_input_throughput": cache_miss_input_thr,
                    "cache_hit_input_throughput": cache_hit_input_thr,
                    "cache_hit_rate": cache_hit_rate,
                    "slo_ok": None,
                })
                logging.warning("Benchmark command failed or metrics unparseable. Treating as saturation failure and backing off.")
                consecutive_failures += 1
                
                mc_ceiling = min(mc_ceiling, max_conn)
                
                if best_valid_config is not None:
                    safe_rr, safe_mc = best_valid_config
                    logging.info(f"Falling back towards last known good config (RR={safe_rr}, MC={safe_mc}).")
                    req_rate = max(0.2, (req_rate + safe_rr) / 2)
                    max_conn = _snap_mc(
                        min(max_conn, int((max_conn + safe_mc) / 2))
                    )
                else:
                    req_rate = max(0.2, req_rate * 0.5)
                    max_conn = _snap_mc(max_conn * 0.5)
                
                if consecutive_failures >= 4:
                    logging.info("Search converged after multiple consecutive failures.")
                    break
                
                time.sleep(2)
                continue
                
            logging.info(
                f"Result: TTFT={ttft:.2f}ms, TPOT={tpot:.2f}ms, E2E={e2e_lat:.2f}ms, "
                f"{throughput_label}={obj_thr:.2f}tok/s"
            )
            
            ttft_ok = True if decode_only else ttft <= slo_ttft
            tpot_ok = tpot <= slo_tpot
            cache_ok = (
                not prefill_cache
                or (
                    cache_hit_rate is not None
                    and cache_hit_rate >= PREFILL_CACHE_MIN_HIT_RATE
                )
            )
            cache_limited = prefill_cache and not cache_ok
            if cache_limited:
                logging.warning(
                    f"Cache hit rate {cache_hit_rate:.2f}% is below the "
                    f"{PREFILL_CACHE_MIN_HIT_RATE:.0f}% floor; treating "
                    f"MC={max_conn} as a KV-cache capacity boundary."
                )
            
            in_thr = cache_hit_input_thr
            all_results.append({
                "iter": len(all_results) + 1, "phase": 1, "req_rate": req_rate, "max_conn": max_conn,
                "num_prompts": search_num_prompts(max_conn),
                "ttft": ttft, "tpot": tpot, "e2e": e2e_lat,
                "throughput": throughput, "out_throughput": out_thr, "in_throughput": in_thr,
                "cache_miss_input_throughput": cache_miss_input_thr,
                "cache_hit_input_throughput": cache_hit_input_thr if prefill_cache else None,
                "cache_hit_rate": cache_hit_rate if prefill_cache else None,
                "objective_throughput": obj_thr,
                "slo_ok": ttft_ok and tpot_ok and cache_ok,
            })
            
            if cache_ok and obj_thr > best_max_throughput:
                logging.info(f"New Max {throughput_label} (any SLO)! {obj_thr:.2f} tok/s (was {best_max_throughput:.2f})")
                best_max_throughput = obj_thr
                best_max_config = (req_rate, max_conn)
                best_max_metrics = (
                    ttft, tpot, e2e_lat, throughput, out_thr,
                    cache_miss_input_thr, cache_hit_input_thr, cache_hit_rate,
                )
            
            if ttft_ok and tpot_ok and cache_ok:
                if obj_thr >= best_overall_throughput:
                    logging.info(f"New Best SLO-Compliant {throughput_label}! {obj_thr:.2f} tok/s (was {best_overall_throughput:.2f})")
                    best_overall_throughput = obj_thr
                    best_valid_config = (req_rate, max_conn)
                    best_valid_metrics = (
                        ttft, tpot, e2e_lat, throughput, out_thr,
                        cache_miss_input_thr, cache_hit_input_thr, cache_hit_rate,
                    )
                    
                consecutive_failures = 0
                
                expected_duration_sec = e2e_lat / 1000.0
                theoretical_mc = int(req_rate * expected_duration_sec)
                mc_slack = 1.5 if decode_only else 1.2
                suggested_mc = int(theoretical_mc * mc_slack) + 2
                mc_step = _coarse_mc_step(max_conn)
                target_mc = min(suggested_mc, max_conn + mc_step, mc_ceiling)

                if decode_only:
                    if tpot < slo_tpot * 0.55:
                        step = max(2.0, req_rate * 0.50)
                        logging.info(f"Decode TPOT has significant headroom. Increasing Request Rate (+{step:.1f}).")
                        req_rate += step
                    elif tpot < slo_tpot * 0.80:
                        step = max(1.0, req_rate * 0.25)
                        logging.info(f"Decode TPOT is healthy. Increasing Request Rate (+{step:.1f}).")
                        req_rate += step
                    elif max_conn < target_mc:
                        logging.info(f"Decode TPOT is near SLO. Holding RR and increasing MC to {target_mc}.")
                        max_conn = target_mc
                    else:
                        step = max(0.5, req_rate * 0.10)
                        logging.info(f"Decode TPOT is at the boundary. Testing a small RR step (+{step:.1f}).")
                        req_rate += step
                    if max_conn < target_mc:
                        max_conn = target_mc
                else:
                    if ttft < slo_ttft * 0.5:
                        logging.info("TTFT has significant headroom. Increasing Request Rate (+1.0).")
                        req_rate += 1.0
                        if max_conn < target_mc:
                            max_conn = target_mc
                    elif ttft < slo_ttft * 0.85:
                        logging.info("TTFT is approaching SLO. Gently increasing Request Rate (+0.5).")
                        req_rate += 0.5
                        if max_conn < target_mc:
                            max_conn = target_mc
                    elif max_conn < target_mc:
                        logging.info(f"TTFT is very close to SLO. Holding RR and increasing MC to {target_mc}")
                        max_conn = target_mc
                    else:
                        logging.info("System is perfectly balanced at the SLO boundary. Testing limits slightly (+0.2 RR).")
                        req_rate += 0.2
                
                # Clamp to ceilings to avoid revisiting known-bad regions
                if req_rate > rr_ceiling:
                    req_rate = rr_ceiling
                if max_conn > mc_ceiling:
                    max_conn = mc_ceiling
                
                # Check if clamped values snap back to current config (converged)
                snapped_rr = max(0.2, round(round(req_rate * 5) / 5, 1))
                snapped_mc = _snap_mc(max_conn)
                if best_valid_config and snapped_rr == best_valid_config[0] and snapped_mc == best_valid_config[1]:
                    logging.info(f"Search converged at ceiling boundary (RR={snapped_rr}, MC={snapped_mc}).")
                    break
            else:
                consecutive_failures += 1
                failed_rr, failed_mc = req_rate, max_conn
                
                if best_valid_config:
                    safe_rr, safe_mc = best_valid_config

                    if (not ttft_ok or cache_limited) and not boundary_probe_done:
                        boundary_probe_done = True
                        boundary_config = _probe_ttft_boundary(
                            safe_rr, safe_mc, failed_rr, failed_mc
                        )
                        if boundary_config is not None:
                            boundary_rr, boundary_mc = boundary_config
                            logging.info(
                                "Adaptive MC probe found an SLO/cache-valid "
                                f"point at RR={boundary_rr:.2f}, MC={boundary_mc}; "
                                "continuing just above that RR."
                            )
                            consecutive_failures = 0
                            req_rate = _normalize_rr(boundary_rr + 0.2)
                            max_conn = boundary_mc
                            time.sleep(SLEEP_SEC)
                            continue
                        safe_rr, safe_mc = best_valid_config

                    if cache_limited:
                        if max_conn > safe_mc:
                            logging.info(
                                "Cache working set exceeded the hit-rate floor. "
                                "Reducing Max Concurrency and setting a hard MC ceiling."
                            )
                            mc_ceiling = min(mc_ceiling, max_conn - 1)
                            max_conn = max(
                                safe_mc, int((safe_mc + max_conn) / 2)
                            )
                            if req_rate > safe_rr:
                                req_rate = max(
                                    safe_rr, (safe_rr + req_rate) / 2
                                )
                        else:
                            # The same MC was cache-valid at a lower RR, so this
                            # is scheduling pressure rather than a larger cache
                            # working set. Back off RR without poisoning the MC
                            # capacity ceiling below a known-good value.
                            logging.info(
                                "Cache hit rate dropped at an already known-good "
                                "MC. Reducing Request Rate instead of the MC ceiling."
                            )
                            rr_ceiling = min(rr_ceiling, failed_rr - 0.2)
                            req_rate = max(
                                safe_rr, (safe_rr + failed_rr) / 2
                            )
                            max_conn = safe_mc
                    elif not tpot_ok:
                        logging.info("TPOT exceeded SLO. Reducing Max Concurrency and setting ceiling.")
                        mc_ceiling = min(mc_ceiling, max_conn - 1)
                        max_conn = max(safe_mc, int((safe_mc + max_conn) / 2))
                        if req_rate > safe_rr:
                            req_rate = max(safe_rr, (safe_rr + req_rate) / 2)
                    
                    if not cache_limited and not ttft_ok:
                        logging.info("TTFT exceeded SLO. Queue is building up. Reducing Request Rate.")
                        rr_ceiling = min(rr_ceiling, failed_rr - 0.2)
                        req_rate = max(safe_rr, (safe_rr + req_rate) / 2)
                        if tpot_ok:
                            max_conn = max(safe_mc, int(max_conn * 0.95))
                    
                    snapped_rr = max(0.2, round(round(req_rate * 5) / 5, 1))
                    snapped_mc = _snap_mc(max_conn)
                    at_or_below_safe = (snapped_rr <= safe_rr and snapped_mc <= safe_mc)
                    at_or_above_fail = (snapped_rr >= failed_rr and snapped_mc >= failed_mc)
                    if at_or_below_safe or at_or_above_fail:
                        logging.info(f"Search converged: no unexplored gap between best valid (RR={safe_rr}, MC={safe_mc}) and failure (RR={failed_rr}, MC={failed_mc}).")
                        break
                else:
                    if cache_limited:
                        logging.info(
                            "Cache working set exceeded the hit-rate floor before "
                            "an SLO-valid point was found. Reducing MC only."
                        )
                        mc_ceiling = min(mc_ceiling, max_conn - 1)
                        max_conn = _snap_mc(int(max_conn * 0.75))
                    elif not tpot_ok:
                        logging.info("TPOT exceeded SLO. Reducing Max Concurrency and setting ceiling.")
                        mc_ceiling = min(mc_ceiling, max_conn - 1)
                        max_conn = _snap_mc(max_conn * 0.75)
                    
                    if not cache_limited and not ttft_ok:
                        logging.info("TTFT exceeded SLO. Queue is building up. Reducing Request Rate.")
                        rr_ceiling = min(rr_ceiling, failed_rr - 0.2)
                        req_rate = max(0.2, req_rate * 0.8)
                        if tpot_ok:
                            max_conn = _snap_mc(max_conn * 0.9)

                if consecutive_failures >= 4:
                    logging.info("Search converged after multiple consecutive failures.")
                    break
                
                if req_rate < 0.1:
                    logging.info("Request rate dropped too low. System cannot sustain SLO.")
                    break
                    
            time.sleep(2)
    except KeyboardInterrupt:
        logging.info("\nUser interrupted. Returning results collected so far...")
        interrupted = True
    result = {
        "input_len": input_len,
        "output_len": output_len,
        "mode": (
            "prefill-cache"
            if prefill_cache
            else ("prefill-only" if prefill_only else "standard")
        ),
        "dataset_name": dataset_name,
        "max_kv_pool_size": max_kv_pool_size,
        "prefill_cache_pool_size": prefill_cache_pool_size,
        "prefill_cache_measured_kv_tokens": (
            prefill_cache_pool_info["measured_kv_tokens"]
            if prefill_cache_pool_info is not None
            else None
        ),
        "slo": None,
        "max": None,
        "history": all_results,
        "interrupted": interrupted,
    }
    
    # --- Track 1: SLO-compliant best ---
    if best_valid_config:
        best_req, best_conn = best_valid_config
        (
            b_ttft, b_tpot, b_e2e, b_throughput, b_out_thr,
            b_miss_input_thr, b_hit_input_thr, b_cache_hit_rate,
        ) = best_valid_metrics
        b_obj_thr = objective_throughput(
            b_throughput, b_out_thr, b_hit_input_thr
        )
        
        if not interrupted:
            logging.info(f"=== [SLO-Compliant] Optimal range locked at req_rate={best_req}, max_concurrency={best_conn} (Best {throughput_label}: {b_obj_thr:.2f}) ===")
            if prefill_cache:
                logging.info(
                    "Running final prefill-cache test with a fixed "
                    "max_concurrency-sized unique prompt set replayed "
                    f"{PREFILL_CACHE_FINAL_HIT_REPETITIONS} times..."
                )
                final_prompt_multiplier = 1
                final_hit_repetitions = PREFILL_CACHE_FINAL_HIT_REPETITIONS
            else:
                logging.info("Running final test with num_prompts = 4 * max_concurrency to get more stable long-term performance...")
                final_prompt_multiplier = 4
                final_hit_repetitions = 1
            
            (
                f_ttft, f_tpot, f_e2e, f_throughput, f_out_thr, _, _,
                f_miss_input_thr, f_hit_input_thr, f_cache_hit_rate,
            ) = _run_search_benchmark(
                best_req,
                best_conn,
                final_prompt_multiplier,
                hit_repetitions=final_hit_repetitions,
            )
            
            final_slo_ok = (
                f_ttft is not None
                and f_tpot is not None
                and objective_throughput(
                    f_throughput, f_out_thr, f_hit_input_thr
                ) is not None
                and (decode_only or f_ttft <= slo_ttft)
                and f_tpot <= slo_tpot
                and (
                    not prefill_cache
                    or (
                        f_cache_hit_rate is not None
                        and f_cache_hit_rate >= PREFILL_CACHE_MIN_HIT_RATE
                    )
                )
            )
            if final_slo_ok:
                f_obj_thr = objective_throughput(
                    f_throughput, f_out_thr, f_hit_input_thr
                )
                logging.info(f"=== [SLO-Compliant] Final Test Result for {input_len}/{output_len} ===")
                final_sample_label = (
                    f"{PREFILL_CACHE_FINAL_HIT_REPETITIONS} fixed-workset hit replays"
                    if prefill_cache
                    else "4x prompts"
                )
                logging.info(f"Metrics ({final_sample_label}): TTFT={f_ttft:.2f}ms, TPOT={f_tpot:.2f}ms, {throughput_label}={f_obj_thr:.2f}tok/s")
                logging.info("Final test passed SLO constraints.")
                final_metrics = (
                    f_ttft, f_tpot, f_e2e, f_throughput, f_out_thr,
                    f_miss_input_thr, f_hit_input_thr, f_cache_hit_rate,
                )
            else:
                if (
                    f_ttft is None
                    or f_tpot is None
                    or objective_throughput(
                        f_throughput, f_out_thr, f_hit_input_thr
                    ) is None
                ):
                    logging.warning(
                        "[SLO-Compliant] Final fixed-workset replay test FAILED to execute. "
                        "Falling back to exploration metrics."
                        if prefill_cache
                        else "[SLO-Compliant] Final 4x test FAILED to execute. Falling back to 2x exploration metrics."
                    )
                else:
                    final_test_name = (
                        "fixed-workset replay"
                        if prefill_cache
                        else "4x"
                    )
                    cache_status = (
                        f", CacheHit: {f_cache_hit_rate:.2f}%/"
                        f"{PREFILL_CACHE_MIN_HIT_RATE:.0f}%"
                        if prefill_cache and f_cache_hit_rate is not None
                        else ""
                    )
                    logging.warning(
                        f"[SLO-Compliant] Final {final_test_name} test "
                        f"VIOLATED validity constraints "
                        f"(TTFT: {f_ttft:.2f}/{slo_ttft}, "
                        f"TPOT: {f_tpot:.2f}/{slo_tpot}{cache_status})."
                    )
                    logging.warning("Falling back to the successful exploration metrics for the final report.")
                final_metrics = best_valid_metrics
        else:
            logging.info("[SLO-Compliant] Skipping final stability test due to user interrupt. Using exploration metrics.")
            final_metrics = best_valid_metrics

        result["slo"] = {
            "best_req_rate": best_req,
            "best_max_conn": best_conn,
            "final_ttft": final_metrics[0],
            "final_tpot": final_metrics[1],
            "final_e2e": final_metrics[2],
            "final_throughput": final_metrics[3],
            "final_out_throughput": final_metrics[4],
            "final_in_throughput": final_metrics[6],
            "final_cache_miss_input_throughput": final_metrics[5],
            "final_cache_hit_input_throughput": (
                final_metrics[6] if prefill_cache else None
            ),
            "final_cache_hit_rate": final_metrics[7] if prefill_cache else None,
            "final_objective_throughput": objective_throughput(
                final_metrics[3], final_metrics[4], final_metrics[6]
            ),
        }
    else:
        logging.warning("Could not find any configuration that satisfies the SLO.")
        # Find the closest-to-SLO result as a reference
        closest = None
        closest_score = float('inf')
        for h in all_results:
            if h["ttft"] is None or h["tpot"] is None:
                continue
            ttft_over = 0 if decode_only else max(0, (h["ttft"] - slo_ttft) / slo_ttft)
            tpot_over = max(0, (h["tpot"] - slo_tpot) / slo_tpot)
            score = ttft_over + tpot_over
            if score < closest_score:
                closest_score = score
                closest = h
        if closest:
            # ANSI terminal highlight
            WARN = "\033[1;33m"
            BG = "\033[43;30m"
            RESET = "\033[0m"
            print(f"\n{BG}{'='*80}{RESET}")
            print(f"{BG}  ⚠️  NO SLO-COMPLIANT RESULT — Showing closest match (NEAR-MISS)  ⚠️          {RESET}")
            print(f"{BG}{'='*80}{RESET}")
            print(f"{WARN}  RR = {closest['req_rate']}, MC = {closest['max_conn']}{RESET}")
            print(f"{WARN}  TTFT = {closest['ttft']:.2f}ms  (SLO = {slo_ttft}ms, over by {closest['ttft'] - slo_ttft:+.2f}ms){RESET}")
            print(f"{WARN}  TPOT = {closest['tpot']:.2f}ms  (SLO = {slo_tpot}ms, over by {closest['tpot'] - slo_tpot:+.2f}ms){RESET}")
            closest_obj_thr = objective_throughput(
                closest["throughput"],
                closest.get("out_throughput"),
                closest.get("in_throughput"),
            )
            print(f"{WARN}  {throughput_label} = {closest_obj_thr:.2f} tok/s{RESET}")
            print(f"{BG}{'='*80}{RESET}\n")
            result["slo"] = {
                "best_req_rate": closest["req_rate"],
                "best_max_conn": closest["max_conn"],
                "final_ttft": closest["ttft"],
                "final_tpot": closest["tpot"],
                "final_e2e": closest["e2e"],
                "final_throughput": closest["throughput"],
                "final_out_throughput": closest.get("out_throughput"),
                "final_in_throughput": closest.get("in_throughput"),
                "final_cache_miss_input_throughput": closest.get(
                    "cache_miss_input_throughput"
                ),
                "final_cache_hit_input_throughput": closest.get(
                    "cache_hit_input_throughput"
                ),
                "final_cache_hit_rate": closest.get("cache_hit_rate"),
                "final_objective_throughput": closest_obj_thr,
                "near_miss": True,
            }
    
    # ========== Phase 2: Max-Throughput Search ==========
    if not interrupted:
        if best_max_config:
            p2_rr, p2_mc = best_max_config
        elif best_valid_config:
            p2_rr, p2_mc = best_valid_config
        else:
            p2_rr, p2_mc = 1.0, _snap_mc(4)
        
        # Start one step above the current best. Decode-only can ramp faster
        # because the shared prefix keeps prefill mostly out of the hot path.
        p2_rr += max(2.0, p2_rr * 0.25) if decode_only else 1.0
        # Phase 2 only respects crash-ceiling (mc_ceiling), NOT rr_ceiling (which is SLO-related)
        p2_mc_ceiling = mc_ceiling
        consecutive_declines = 0
        
        logging.info(f"\n=== Phase 2: Max {throughput_label} Search [{mode_name}] for Input: {input_len}, Output: {output_len} ===")
        logging.info(f"Starting from RR={p2_rr:.1f}, MC={p2_mc} (Phase 1 best {throughput_label}: {best_max_throughput:.2f} tok/s)")
        
        try:
            for p2_i in range(1, 21):
                # Snap to grid
                p2_rr = round(round(p2_rr * 5) / 5, 1)
                p2_rr = max(0.2, p2_rr)
                p2_mc = _snap_mc(p2_mc)
                p2_mc = min(p2_mc, p2_mc_ceiling)
                
                # Aggressively set MC via Little's Law BEFORE running
                # Use estimated E2E from Phase 1's best observation
                if best_max_metrics:
                    est_e2e_sec = best_max_metrics[2] / 1000.0  # E2E in seconds
                elif best_valid_metrics:
                    est_e2e_sec = best_valid_metrics[2] / 1000.0
                else:
                    est_e2e_sec = 30.0
                needed_mc = int(p2_rr * est_e2e_sec * 1.2) + 2
                p2_mc = max(p2_mc, min(needed_mc, p2_mc_ceiling))
                p2_mc = _snap_mc(p2_mc)
                
                logging.info(f"--- Phase 2 Iter {p2_i} (RR={p2_rr:.2f}, MC={p2_mc}) ---")
                (
                    ttft, tpot, e2e_lat, throughput, out_thr, req_thr, _,
                    cache_miss_input_thr, cache_hit_input_thr, cache_hit_rate,
                ) = _run_search_benchmark(
                    p2_rr,
                    p2_mc,
                    effective_prompt_multiplier(2),
                )
                
                in_thr = cache_hit_input_thr
                p2_cache_ok = (
                    not prefill_cache
                    or (
                        cache_hit_rate is not None
                        and cache_hit_rate >= PREFILL_CACHE_MIN_HIT_RATE
                    )
                )
                slo_ok_flag = (
                    (decode_only or ttft <= slo_ttft)
                    and tpot <= slo_tpot
                    and p2_cache_ok
                    if (ttft is not None and tpot is not None)
                    else None
                )
                all_results.append({
                    "iter": len(all_results) + 1, "phase": 2,
                    "req_rate": p2_rr, "max_conn": p2_mc,
                    "num_prompts": search_num_prompts(p2_mc),
                    "ttft": ttft, "tpot": tpot, "e2e": e2e_lat,
                    "throughput": throughput, "out_throughput": out_thr, "in_throughput": in_thr,
                    "cache_miss_input_throughput": cache_miss_input_thr,
                    "cache_hit_input_throughput": cache_hit_input_thr if prefill_cache else None,
                    "cache_hit_rate": cache_hit_rate if prefill_cache else None,
                    "objective_throughput": objective_throughput(
                        throughput, out_thr, cache_hit_input_thr
                    ),
                    "slo_ok": slo_ok_flag,
                })
                
                obj_thr = objective_throughput(
                    throughput, out_thr, cache_hit_input_thr
                )
                if ttft is None or tpot is None or e2e_lat is None or obj_thr is None:
                    logging.warning("Phase 2: Benchmark crashed. Reducing MC and retrying.")
                    p2_mc_ceiling = min(p2_mc_ceiling, p2_mc)
                    consecutive_declines += 1
                    if consecutive_declines >= 2:
                        logging.info("Phase 2: Multiple crashes. Stopping max-throughput search.")
                        break
                    p2_mc = _snap_mc(p2_mc * 0.75)
                    time.sleep(2)
                    continue
                
                logging.info(f"Phase 2 Result: TTFT={ttft:.2f}ms, TPOT={tpot:.2f}ms, {throughput_label}={obj_thr:.2f}tok/s"
                             + (f", ReqThr={req_thr:.2f}req/s" if req_thr else ""))

                if not p2_cache_ok:
                    logging.info(
                        f"Phase 2: cache hit rate {cache_hit_rate:.2f}% is below "
                        f"the {PREFILL_CACHE_MIN_HIT_RATE:.0f}% floor at MC={p2_mc}. "
                        "Treating this as the KV-cache capacity ceiling and stopping."
                    )
                    p2_mc_ceiling = min(p2_mc_ceiling, p2_mc - 1)
                    mc_ceiling = min(mc_ceiling, p2_mc - 1)
                    break
                
                # Saturation detection: if actual request throughput << requested rate, system is overwhelmed
                if req_thr is not None and req_thr < p2_rr * 0.5:
                    logging.info(f"Phase 2: System saturated — actual req throughput {req_thr:.2f} req/s << requested {p2_rr:.2f}. Stopping.")
                    # Still record this throughput if it's the best
                    if obj_thr > best_max_throughput:
                        best_max_throughput = obj_thr
                        best_max_config = (p2_rr, p2_mc)
                        best_max_metrics = (
                            ttft, tpot, e2e_lat, throughput, out_thr,
                            cache_miss_input_thr, cache_hit_input_thr, cache_hit_rate,
                        )
                    break
                
                if obj_thr > best_max_throughput:
                    logging.info(f"New Max {throughput_label}! {obj_thr:.2f} tok/s (was {best_max_throughput:.2f})")
                    best_max_throughput = obj_thr
                    best_max_config = (p2_rr, p2_mc)
                    best_max_metrics = (
                        ttft, tpot, e2e_lat, throughput, out_thr,
                        cache_miss_input_thr, cache_hit_input_thr, cache_hit_rate,
                    )
                    consecutive_declines = 0
                else:
                    consecutive_declines += 1
                    if consecutive_declines >= 3:
                        logging.info(f"Phase 2: {throughput_label} declining consistently. Max-throughput search converged.")
                        break
                
                # Next step: increase RR, MC will be recalculated at top of loop via Little's law
                p2_rr += max(2.0, p2_rr * 0.20) if decode_only else 1.0
                # Update E2E estimate for next MC calculation
                est_e2e_sec = e2e_lat / 1000.0
                
                time.sleep(2)
        except KeyboardInterrupt:
            logging.info("\nUser interrupted Phase 2. Returning results collected so far...")
            interrupted = True
    
    # --- Track 2: Max-Throughput best (may violate SLO) ---
    if best_max_config:
        max_req, max_conn_val = best_max_config
        (
            m_ttft, m_tpot, m_e2e, m_throughput, m_out_thr,
            m_miss_input_thr, m_hit_input_thr, m_cache_hit_rate,
        ) = best_max_metrics
        m_obj_thr = objective_throughput(
            m_throughput, m_out_thr, m_hit_input_thr
        )
        slo_ok = (
            (decode_only or m_ttft <= slo_ttft)
            and m_tpot <= slo_tpot
            and (
                not prefill_cache
                or (
                    m_cache_hit_rate is not None
                    and m_cache_hit_rate >= PREFILL_CACHE_MIN_HIT_RATE
                )
            )
        )
        logging.info(f"=== [Max-Throughput] Highest {throughput_label} observed: {m_obj_thr:.2f} tok/s at req_rate={max_req}, max_concurrency={max_conn_val} (SLO compliant: {slo_ok}) ===")
        
        result["max"] = {
            "best_req_rate": max_req,
            "best_max_conn": max_conn_val,
            "final_ttft": m_ttft,
            "final_tpot": m_tpot,
            "final_e2e": m_e2e,
            "final_throughput": m_throughput,
            "final_out_throughput": m_out_thr,
            "final_in_throughput": m_hit_input_thr,
            "final_cache_miss_input_throughput": m_miss_input_thr,
            "final_cache_hit_input_throughput": (
                m_hit_input_thr if prefill_cache else None
            ),
            "final_cache_hit_rate": (
                m_cache_hit_rate if prefill_cache else None
            ),
            "final_objective_throughput": m_obj_thr,
            "slo_compliant": slo_ok,
        }
    else:
        logging.warning("No benchmark runs produced throughput data.")
    
    result["interrupted"] = interrupted
    return result if (result["slo"] or result["max"] or result["history"]) else None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-search SGLang serving benchmark pressure points.")
    parser.add_argument(
        "--dataset-name",
        default="random",
        help=(
            "Dataset name passed to sglang.bench_serving. Defaults to random. "
            "Common values include random, random-ids, generated-shared-prefix, "
            "sharegpt, sonnet, hf, custom, burstgpt."
        ),
    )
    parser.add_argument(
        "--decode-only",
        action="store_true",
        help=(
            "Use a shared random prefix for every request so radix cache can "
            "serve most prefill tokens and the search stresses decode."
        ),
    )
    parser.add_argument(
        "--prefill-only",
        action="store_true",
        help="Set benchmark output length to 1 to isolate prefill performance.",
    )
    parser.add_argument(
        "--prefill-cache",
        action="store_true",
        help=(
            "Benchmark generated-shared-prefix with system_prompt_len=0 and real "
            "server-reported cache token counts. Request rate is fixed at inf and "
            "the search varies only max concurrency. KV-pool capacity is read from "
            "prefill-worker metrics when available, or overridden by "
            "--max-kv-pool-size; otherwise each MC runs a cold-fill/hit pair. "
            "Output length is set to 1."
        ),
    )
    parser.add_argument(
        "--max-kv-pool-size",
        type=int,
        default=None,
        metavar="TOKENS",
        help=(
            "With --prefill-cache, bound the deterministic cold-filled working set "
            "by this many KV-cache tokens. The effective prompt count is derived "
            "from input/output lengths and calibrated with server-reported prompt "
            "tokens; it then becomes the MC search ceiling."
        ),
    )
    parser.add_argument(
        "--prefill-cache-metrics-url",
        default=None,
        metavar="URL",
        help=(
            "Optional prefill-worker Prometheus /metrics URL used to discover "
            "KV-pool capacity. Without this option, prefill-cache mode tries "
            "the benchmark endpoint and the same host on port 30000. An "
            "explicit --max-kv-pool-size always takes precedence."
        ),
    )
    parser.add_argument(
        "--kv-page-size",
        type=int,
        default=PREFILL_CACHE_KV_PAGE_SIZE,
        metavar="TOKENS",
        help=(
            "KV-cache page size used by the prefill server (default: 256). "
            "With --prefill-cache, the cold-fill budget uses "
            "(context_len / page_size - 1) * page_size / context_len of the "
            "physical KV pool. This must match the server's --page-size."
        ),
    )
    parser.add_argument(
        "--only-input-len",
        type=int,
        default=None,
        metavar="TOKENS",
        help="Run only the test case whose input_len matches this value.",
    )
    parser.add_argument(
        "--decode-kv-pool-tokens",
        type=int,
        default=994816,
        metavar="TOKENS",
        help=(
            "Decode worker KV-pool token capacity used by --decode-only to skip "
            "unsafe long-context cases. Set this to the service log's "
            "max_total_num_tokens when running a 1M-capacity profile."
        ),
    )
    args = parser.parse_args()
    if sum((args.decode_only, args.prefill_only, args.prefill_cache)) > 1:
        parser.error(
            "--decode-only, --prefill-only, and --prefill-cache are mutually exclusive"
        )
    if args.max_kv_pool_size is not None:
        if not args.prefill_cache:
            parser.error("--max-kv-pool-size requires --prefill-cache")
        if args.max_kv_pool_size <= 0:
            parser.error("--max-kv-pool-size must be positive")
    if args.prefill_cache_metrics_url and not args.prefill_cache:
        parser.error("--prefill-cache-metrics-url requires --prefill-cache")
    if args.kv_page_size <= 0:
        parser.error("--kv-page-size must be positive")
    if args.only_input_len is not None and args.only_input_len <= 0:
        parser.error("--only-input-len must be positive")
    if args.decode_kv_pool_tokens <= 0:
        parser.error("--decode-kv-pool-tokens must be positive")
    if args.decode_kv_pool_tokens != 994816 and not args.decode_only:
        parser.error("--decode-kv-pool-tokens requires --decode-only")
    if args.decode_only and not (
        args.dataset_name.startswith("random")
        or args.dataset_name == "generated-shared-prefix"
    ):
        parser.error("--decode-only currently requires a random* dataset or --dataset-name generated-shared-prefix")
    if args.decode_only and args.dataset_name.startswith("random"):
        logging.info(
            "Decode-only mode uses dataset=generated-shared-prefix; "
            f"overriding dataset={args.dataset_name} because this bench_serving "
            "version does not support --random-prefix-len."
        )
        args.dataset_name = "generated-shared-prefix"
    if args.prefill_cache:
        if args.dataset_name != "generated-shared-prefix":
            logging.info(
                "Prefill-cache mode uses dataset=generated-shared-prefix; "
                f"overriding dataset={args.dataset_name}."
            )
        args.dataset_name = "generated-shared-prefix"

    # Define test cases from the plan
    test_cases = [
        {
            "input_len": 8000,
            "output_len": 1500,
            "slo_ttft": 5000,
            "slo_tpot": 16,
            "dp": 8,
        },
        {
            "input_len": 16000,
            "output_len": 1500,
            "slo_ttft": 10000,
            "slo_tpot": 16,
            "dp": 8,
        },
        {
            "input_len": 32000,
            "output_len": 1500,
            "slo_ttft": 15000,
            "slo_tpot": 16,
            "dp": 8,
        },
        {
            "input_len": 128000,
            "output_len": 1500,
            "slo_ttft": 15000,
            "slo_tpot": 16,
            "dp": 8,
        },
        {
            "input_len": 1024000,
            "output_len": 1500,
            "slo_ttft": 15000,
            "slo_tpot": 16,
            "dp": 8,
        }
    ]
    if args.only_input_len is not None:
        test_cases = [
            tc for tc in test_cases if int(tc["input_len"]) == int(args.only_input_len)
        ]
        if not test_cases:
            parser.error(
                f"--only-input-len {args.only_input_len} does not match any test case"
            )
    
    results = []
    try:
        for tc in test_cases:
            benchmark_output_len = (
                1 if (args.prefill_only or args.prefill_cache) else tc["output_len"]
            )
            if args.max_kv_pool_size is not None:
                minimum_kv_tokens = 10 * (
                    int(tc["input_len"]) + int(benchmark_output_len)
                )
                usable_kv_tokens = _prefill_cache_usable_kv_tokens(
                    max_kv_pool_size=args.max_kv_pool_size,
                    context_len=tc["input_len"],
                    page_size=args.kv_page_size,
                )
                if usable_kv_tokens < minimum_kv_tokens:
                    logging.warning(
                        f"Skipping input_len={tc['input_len']}: "
                        f"max_kv_pool_size={args.max_kv_pool_size} leaves only "
                        f"{usable_kv_tokens} usable KV tokens and cannot hold "
                        f"the minimum 10 estimated requests "
                        f"({minimum_kv_tokens} KV tokens required)."
                    )
                    continue

            if args.decode_only:
                logging.info(
                    f"Decode profile for {tc['input_len']}/{tc['output_len']}: "
                    f"prefix={tc['input_len'] - DECODE_SUFFIX_LEN}, "
                    f"suffix={DECODE_SUFFIX_LEN}, "
                    f"warmup_prompts={DECODE_WARMUP_PROMPTS}"
                )
            elif args.prefill_only:
                logging.info(
                    f"Prefill-only profile for {tc['input_len']}/{tc['output_len']}: "
                    "benchmark output_len=1"
                )
            elif args.prefill_cache:
                pool_detail = (
                    f", max_kv_pool_size={args.max_kv_pool_size} tokens"
                    if args.max_kv_pool_size is not None
                    else ""
                )
                logging.info(
                    f"Prefill-cache profile for {tc['input_len']}/{tc['output_len']}: "
                    "benchmark output_len=1, system_prompt_len=0"
                    f"{pool_detail}"
                )

            res = find_optimal_throughput(
                input_len=tc["input_len"],
                output_len=benchmark_output_len,
                slo_ttft=tc["slo_ttft"],
                slo_tpot=tc["slo_tpot"],
                dp=tc.get("dp", 1),
                decode_only=args.decode_only,
                prefill_only=args.prefill_only,
                prefill_cache=args.prefill_cache,
                decode_suffix_len=DECODE_SUFFIX_LEN,
                warmup_prompts=DECODE_WARMUP_PROMPTS,
                dataset_name=args.dataset_name,
                max_kv_pool_size=args.max_kv_pool_size,
                prefill_cache_metrics_url=args.prefill_cache_metrics_url,
                kv_page_size=args.kv_page_size,
                decode_kv_pool_tokens=args.decode_kv_pool_tokens,
            )
            if res:
                results.append(res)
                if res.get("interrupted"):
                    logging.info("Stopping remaining test cases due to user interrupt.")
                    break
    except KeyboardInterrupt:
        logging.info("\nUser interrupted between test cases. Printing results collected so far...")

    if args.prefill_only or args.prefill_cache:
        logging.info(
            f"Prefill mode does not affect MMLU; probe uses max_tokens={MMLU_MAX_TOKENS}."
        )
    mmlu_result = run_mmlu_probe()

    # Print detailed history table for each test case, then the summary
    def _fmt(val, fmt=".2f"):
        return f"{val:{fmt}}" if val is not None else "-"

    def _fmt_rr(val):
        if val is None:
            return "-"
        if isinstance(val, str):
            return val
        return f"{val:.2f}"

    def _fmt_pct(val):
        return f"{val:.2f}%" if val is not None else "-"
    
    for r in results:
        io_str = f"{r['input_len']}/{r['output_len']}"
        history = r.get("history", [])
        slo_cfg = (r["slo"]["best_req_rate"], r["slo"]["best_max_conn"]) if r.get("slo") else None
        slo_near_miss = r["slo"].get("near_miss", False) if r.get("slo") else False
        max_cfg = (r["max"]["best_req_rate"], r["max"]["best_max_conn"]) if r.get("max") else None
        
        decode_detail_mode = r.get("mode") == "decode-only"
        prefill_cache_detail_mode = r.get("mode") == "prefill-cache"
        H_WIDTH = 140 if decode_detail_mode else (214 if prefill_cache_detail_mode else 178)
        print("\n" + "=" * H_WIDTH)
        mode_str = r.get("mode", "standard")
        dataset_str = r.get("dataset_name", "random")
        print(f"EXPLORATION DETAIL — {io_str} [{mode_str}, dataset={dataset_str}]")
        print("=" * H_WIDTH)
        if decode_detail_mode:
            hdr = (f"{'Iter':<5} | {'Ph':<3} | {'RR':<6} | {'MC':<5} | {'TTFT(ms)':<10} | {'TPOT(ms)':<10} | {'E2E(ms)':<11} "
                   f"| {'OutThr(t/s)':<13} | {'SLO':<8} | {'Optimal'}")
        elif prefill_cache_detail_mode:
            hdr = (f"{'Iter':<5} | {'Ph':<3} | {'RR':<6} | {'MC':<5} | {'Unique':<7} | {'Rep':<4} | {'Requests':<8} | {'TTFT(ms)':<10} | {'TPOT(ms)':<10} | {'E2E(ms)':<11} "
                   f"| {'MissIn(t/s)':<13} | {'HitIn(t/s)':<13} | {'HitRate':<9} | {'SLO':<8} | {'Optimal'}")
        else:
            hdr = (f"{'Iter':<5} | {'Ph':<3} | {'RR':<6} | {'MC':<5} | {'TTFT(ms)':<10} | {'TPOT(ms)':<10} | {'E2E(ms)':<11} "
                   f"| {'InThr(t/s)':<12} | {'OutThr(t/s)':<13} | {'TotalThr(t/s)':<15} | {'SLO':<8} | {'Optimal'}")
        print(hdr)
        print("-" * H_WIDTH)
        for h in history:
            slo_str = "-" if h["slo_ok"] is None else ("OK" if h["slo_ok"] else "VIOLATE")
            tags = []
            cfg = (h["req_rate"], h["max_conn"])
            if slo_cfg and cfg == slo_cfg:
                if slo_near_miss:
                    tags.append("★NearMiss")
                elif h["slo_ok"]:
                    tags.append("★SLO-Best")
            if max_cfg and cfg == max_cfg:
                tags.append("★CacheHit-Best" if prefill_cache_detail_mode else "★Max-Best")
            tag_str = " ".join(tags) if tags else ""
            ph = h.get("phase", 1)
            if decode_detail_mode:
                row = (f"{h['iter']:<5} | {ph:<3} | {_fmt_rr(h['req_rate']):<6} | {h['max_conn']:<5} | {_fmt(h['ttft']):<10} | {_fmt(h['tpot']):<10} | {_fmt(h['e2e']):<11} "
                       f"| {_fmt(h.get('out_throughput')):<13} | {slo_str:<8} | {tag_str}")
            elif prefill_cache_detail_mode:
                row = (f"{h['iter']:<5} | {ph:<3} | {_fmt_rr(h['req_rate']):<6} | {h['max_conn']:<5} | {_fmt(h.get('unique_prompts', h.get('num_prompts')), 'd'):<7} | {_fmt(h.get('request_repeat_count', 1), 'd'):<4} | {_fmt(h.get('total_requests', h.get('num_prompts')), 'd'):<8} | {_fmt(h['ttft']):<10} | {_fmt(h['tpot']):<10} | {_fmt(h['e2e']):<11} "
                       f"| {_fmt(h.get('cache_miss_input_throughput')):<13} | {_fmt(h.get('cache_hit_input_throughput')):<13} "
                       f"| {_fmt_pct(h.get('cache_hit_rate')):<9} | {slo_str:<8} | {tag_str}")
            else:
                row = (f"{h['iter']:<5} | {ph:<3} | {_fmt_rr(h['req_rate']):<6} | {h['max_conn']:<5} | {_fmt(h['ttft']):<10} | {_fmt(h['tpot']):<10} | {_fmt(h['e2e']):<11} "
                       f"| {_fmt(h.get('in_throughput')):<12} | {_fmt(h.get('out_throughput')):<13} | {_fmt(h['throughput']):<15} | {slo_str:<8} | {tag_str}")
            print(row)
        print("=" * H_WIDTH)
    
    # Print final summary table
    S_WIDTH = 130 if args.decode_only else (166 if args.prefill_cache else 170)
    print("\n" + "=" * S_WIDTH)
    summary_mode = (
        "Decode-Only"
        if args.decode_only
        else ("Prefill-Cache" if args.prefill_cache else ("Prefill-Only" if args.prefill_only else "Standard"))
    )
    summary_objective = (
        "Output Throughput"
        if args.decode_only
        else ("Cache-Hit Input Throughput" if args.prefill_cache else ("Prefill Throughput" if args.prefill_only else "Max Throughput"))
    )
    print(f"FINAL SUMMARY [{summary_mode}, dataset={args.dataset_name}] (Two Tracks: SLO-Compliant Best vs {summary_objective})")
    print("=" * S_WIDTH)
    if args.decode_only:
        hdr = (f"{'Input/Output':<14} | {'Track':<14} | {'RR':<6} | {'MC':<5} | {'TTFT(ms)':<10} | {'TPOT(ms)':<10} | {'E2E(ms)':<11} "
               f"| {'OutThr(t/s)':<13}")
    elif args.prefill_cache:
        hdr = (f"{'Input/Output':<14} | {'Track':<18} | {'RR':<6} | {'MC':<5} | {'TTFT(ms)':<10} | {'TPOT(ms)':<10} | {'E2E(ms)':<11} "
               f"| {'MissIn(t/s)':<13} | {'HitIn(t/s)':<13} | {'HitRate':<9}")
    else:
        hdr = (f"{'Input/Output':<14} | {'Track':<14} | {'RR':<6} | {'MC':<5} | {'TTFT(ms)':<10} | {'TPOT(ms)':<10} | {'E2E(ms)':<11} "
               f"| {'InThr(t/s)':<12} | {'OutThr(t/s)':<13} | {'TotalThr(t/s)':<15}")
    print(hdr)
    print("-" * S_WIDTH)
    for r in results:
        io_str = f"{r['input_len']}/{r['output_len']}"
        tracks = [
            ("SLO-Compliant", "slo"),
            ("Best-Cache-Hit" if args.prefill_cache else "Max-Throughput", "max"),
        ]
        for track_label, key in tracks:
            d = r.get(key)
            if d is None:
                if args.decode_only:
                    print(f"{io_str:<14} | {track_label:<14} | {'-':<6} | {'-':<5} | {'-':<10} | {'-':<10} | {'-':<11} | {'-':<13}")
                elif args.prefill_cache:
                    print(f"{io_str:<14} | {track_label:<18} | {'-':<6} | {'-':<5} | {'-':<10} | {'-':<10} | {'-':<11} | {'-':<13} | {'-':<13} | {'-':<9}")
                else:
                    print(f"{io_str:<14} | {track_label:<14} | {'-':<6} | {'-':<5} | {'-':<10} | {'-':<10} | {'-':<11} | {'-':<12} | {'-':<13} | {'-':<15}")
            else:
                suf = ""
                if d.get("near_miss"):
                    suf = " (NEAR-MISS)"
                elif "slo_compliant" in d:
                    suf = f" ({'OK' if d['slo_compliant'] else 'VIOLATE'})"
                label = "SLO-NearMiss" if d.get("near_miss") else track_label
                if args.decode_only:
                    row = (f"{io_str:<14} | {label:<14} | {_fmt_rr(d['best_req_rate']):<6} | {d['best_max_conn']:<5} "
                           f"| {_fmt(d['final_ttft']):<10} | {_fmt(d['final_tpot']):<10} | {_fmt(d['final_e2e']):<11} "
                           f"| {_fmt(d.get('final_out_throughput')) + suf:<13}")
                elif args.prefill_cache:
                    row = (f"{io_str:<14} | {label:<18} | {_fmt_rr(d['best_req_rate']):<6} | {d['best_max_conn']:<5} "
                           f"| {_fmt(d['final_ttft']):<10} | {_fmt(d['final_tpot']):<10} | {_fmt(d['final_e2e']):<11} "
                           f"| {_fmt(d.get('final_cache_miss_input_throughput')):<13} "
                           f"| {_fmt(d.get('final_cache_hit_input_throughput')):<13} "
                           f"| {_fmt_pct(d.get('final_cache_hit_rate')) + suf:<9}")
                else:
                    row = (f"{io_str:<14} | {label:<14} | {_fmt_rr(d['best_req_rate']):<6} | {d['best_max_conn']:<5} "
                           f"| {_fmt(d['final_ttft']):<10} | {_fmt(d['final_tpot']):<10} | {_fmt(d['final_e2e']):<11} "
                           f"| {_fmt(d.get('final_in_throughput')):<12} | {_fmt(d.get('final_out_throughput')):<13} | {_fmt(d['final_throughput']) + suf:<15}")
                print(row)
        print("-" * S_WIDTH)
    print("=" * S_WIDTH)

    # Print MMLU accuracy probe summary
    print("\n" + "=" * S_WIDTH)
    print("ACCURACY SUMMARY (MMLU Probe)")
    print("=" * S_WIDTH)
    if mmlu_result["status"] == "ok":
        acc = mmlu_result["accuracy"]
        verdict = "PASS" if mmlu_result["passed"] else "FAIL"
        print(
            f"Source: {mmlu_result['source']} | "
            f"Accuracy: {mmlu_result['correct']}/{mmlu_result['total']} ({acc:.2%}) | "
            f"Floor: {mmlu_result['accuracy_floor']:.0%} | {verdict}"
        )
        print(f"Details: {mmlu_result['log_path']}")
    else:
        print("MMLU probe unavailable; accuracy was not validated for this run.")
        print(f"Details: {mmlu_result['log_path']}")
    print("=" * S_WIDTH)
