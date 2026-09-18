"""Process-local EIC RPC and hit-accounting stats.

A single aggregated log line every EIC_STATS_INTERVAL_S seconds is enough to
answer the three tuning questions from the cc8 benchmarks: which KV writes
error, what fraction of a prefix hits, and why the missed part was missed.
Kept dependency-free (no Prometheus): every scheduler rank already lands its
logs in the collected pod logs, so the line is greppable there.
"""

import logging
import os
import threading
import time
from collections import defaultdict, deque

logger = logging.getLogger(__name__)

_INTERVAL_S = float(os.environ.get("EIC_STATS_INTERVAL_S", "60"))
_LATENCY_KEEP = 8192


def _percentile(sorted_values, q):
    if not sorted_values:
        return 0.0
    idx = min(len(sorted_values) - 1, int(q * len(sorted_values)))
    return sorted_values[idx]


class EicStats:
    def __init__(self):
        self._lock = threading.Lock()
        self._t0 = time.monotonic()
        self._lat = defaultdict(lambda: deque(maxlen=_LATENCY_KEEP))
        self._c = defaultdict(int)
        self._rank = os.environ.get("RANK", "?")
        self._started = False

    def start_periodic_dump(self):
        with self._lock:
            if self._started:
                return
            self._started = True
        thread = threading.Thread(
            target=self._dump_loop, name="eic-stats", daemon=True
        )
        thread.start()

    def record_rpc(self, op, seconds, status_name, keys=0, failed=0):
        with self._lock:
            self._c[f"{op}.calls"] += 1
            self._c[f"{op}.{status_name}"] += 1
            self._c[f"{op}.keys"] += keys
            self._c[f"{op}.keys_failed"] += failed
            self._lat[op].append(seconds)

    def incr(self, name, n=1):
        with self._lock:
            self._c[name] += n

    def observe_admit(self, device_hit, eic_expected, admitted):
        # One request's final prefix accounting, in tokens.
        with self._lock:
            self._c["admit.reqs"] += 1
            self._c["admit.device_tokens"] += device_hit
            self._c["eic.expected_tokens"] += eic_expected
            self._c["eic.admitted_tokens"] += max(0, admitted - device_hit)
            self._c["eic.missing_tokens"] += max(0, device_hit + eic_expected - admitted)

    def observe_load(self, wait_s, service_s):
        # Splits a load op's wall time into queue wait vs service. Separate
        # counters so the periodic line can say which one dominates.
        with self._lock:
            self._c["load.ops"] += 1
            self._lat["load.wait"].append(wait_s)
            self._lat["load.service"].append(service_s)

    def observe_lat(self, name, seconds):
        with self._lock:
            self._lat[name].append(seconds)

    def observe_slow(self, name, seconds, threshold_s):
        # Cumulative count of samples over a threshold. The latency deques are
        # sliding windows, so their p99/max repeats one outlier for many dumps;
        # only this counter differences into a real event rate.
        with self._lock:
            self._c[f"{name}.slow{int(threshold_s)}"] += seconds > threshold_s

    def observe_max(self, name, value):
        # Gauge sampled by the caller; the dump reports the peak seen.
        with self._lock:
            if value > self._c[name]:
                self._c[name] = value

    def _dump_loop(self):
        while True:
            time.sleep(_INTERVAL_S)
            try:
                self.dump()
            except Exception:
                logger.exception("eic stats dump failed")

    def dump(self):
        with self._lock:
            c = dict(self._c)
            lat = {k: sorted(v) for k, v in self._lat.items()}
            uptime = time.monotonic() - self._t0

        def lat_str(op):
            v = lat.get(op, [])
            if not v:
                return f"{op}:none"
            return (
                f"{op}[n={len(v)} p50={_percentile(v, .5) * 1e3:.0f}ms "
                f"p95={_percentile(v, .95) * 1e3:.0f} "
                f"p99={_percentile(v, .99) * 1e3:.0f} "
                f"max={v[-1] * 1e3:.0f}ms]"
            )

        mset_calls = c.get("mset.calls", 0)
        mget_calls = c.get("mget.calls", 0)
        mex_calls = c.get("mexist.calls", 0)
        mset_fail = c.get("mset.PARTIAL_FAILED", 0) + c.get("mset.FAILED", 0)
        mget_fail = c.get("mget.PARTIAL_FAILED", 0) + c.get("mget.FAILED", 0)
        mex_fail = c.get("mexist.PARTIAL_FAILED", 0) + c.get("mexist.FAILED", 0)

        def pct(n, d):
            return f"{100.0 * n / d:.1f}%" if d else "-"

        eic_exp = c.get("eic.expected_tokens", 0)
        eic_got = c.get("eic.admitted_tokens", 0)
        logger.info(
            "EIC_STATS rank=%s uptime=%.0fs | "
            "write nodes=%d fail=%d(%s) | "
            "mset calls=%d fail=%s keys=%d keys_fail=%d(%s) retry=%d | "
            "mget calls=%d partial=%s keys=%d keys_fail=%d(%s) refetch=%d recovered=%d | "
            "mexist calls=%d fail=%s reprobe=%d | "
            "admit reqs=%d device_hit=%d eic_expected=%d eic_got=%d eic_missing=%d eic_hit%%=%s | "
            "miss: cold_or_probe_fail=%d headroom=%d below_threshold=%d dma_incomplete=%d | "
            "load ops=%d qsize_max=%d | "
            "slow>1s: mget=%d unpack=%d | "
            "%s %s %s %s %s %s %s %s %s %s %s",
            self._rank,
            uptime,
            c.get("write.nodes", 0),
            c.get("write.fail", 0),
            pct(c.get("write.fail", 0), c.get("write.nodes", 0)),
            mset_calls,
            pct(mset_fail, mset_calls),
            c.get("mset.keys", 0),
            c.get("mset.keys_failed", 0),
            pct(c.get("mset.keys_failed", 0), c.get("mset.keys", 0)),
            c.get("mset.retry", 0),
            mget_calls,
            pct(mget_fail, mget_calls),
            c.get("mget.keys", 0),
            c.get("mget.keys_failed", 0),
            pct(c.get("mget.keys_failed", 0), c.get("mget.keys", 0)),
            c.get("mget.refetch", 0),
            c.get("mget.recovered", 0),
            mex_calls,
            pct(mex_fail, mex_calls),
            c.get("mexist.reprobe", 0),
            c.get("admit.reqs", 0),
            c.get("admit.device_tokens", 0),
            eic_exp,
            eic_got,
            c.get("eic.missing_tokens", 0),
            pct(eic_got, eic_exp),
            c.get("miss.cold_or_probe_fail", 0),
            c.get("miss.headroom", 0),
            c.get("miss.below_threshold", 0),
            c.get("miss.dma_incomplete", 0),
            c.get("load.ops", 0),
            c.get("load.qsize", 0),
            c.get("mget.slow1", 0),
            c.get("unpack.slow1", 0),
            lat_str("mset"),
            lat_str("mget"),
            lat_str("mexist"),
            lat_str("load.wait"),
            lat_str("load.service"),
            lat_str("load.unpack"),
            lat_str("load.unpack.cat"),
            lat_str("load.unpack.sync"),
            lat_str("load.unpack.h2d"),
            lat_str("mget.first"),
            lat_str("mget.refetch_s"),
        )


stats = EicStats()
