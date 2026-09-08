from dataclasses import dataclass
from enum import Enum


class SidpPrefetchPolicy(str, Enum):
    """How one SiDP cycle orders its remote layer fills."""

    COMPUTE = "compute"
    STATIC_PEAK = "static_peak"
    DYNAMIC_OWNER = "dynamic_owner"


class SidpCopyBackend(str, Enum):
    """Data-movement implementation used by the cycle pipeline."""

    DMA = "dma"
    SM = "sm"


class SidpSlotSync(str, Enum):
    """RAW/WAR protocol for the two-cycle receive slots."""

    EVENT = "event"
    FLAG = "flag"


class SidpDynamicClaimOrder(str, Enum):
    """Priority used while dynamic_owner scans currently eligible owners."""

    ROTATING = "rotating"
    COMPUTE_PRIORITY = "compute_priority"


@dataclass
class SidpConfig:
    # ``dp_size``/``dp_rank`` are the SiDP world size / member rank. In the
    # native single-service DP mode they mirror ``ps.dp_size``/``ps.dp_rank``.
    # In external-worker mode (N independent dp_size=1 services grouped into one
    # SiDP world) they are injected from ``sidp_world_size``/``sidp_member_rank``
    # so the manager's owner/IPC/rendezvous logic stays byte-for-byte identical.
    dp_size: int
    dp_rank: int
    k: int = 1
    cache_cycles: int = 2
    rdzv_port: int = 0
    rdzv_host: str = "127.0.0.1"
    # When True the SiDP group is formed by independent services (Direction A
    # coordinated_static topology). Only affects rendezvous master election and
    # setup-time validation; the fetch/copy data path is unchanged.
    external_mode: bool = False
    # Direction A Phase 2: coordinated_static GPU data plane. When True, a
    # cross-member device barrier (sense-reversing, host-free) is inserted every
    # ``barrier_interval_cycles`` prefetch cycles on the comm stream to correct
    # launch-skew phase drift. Requires external_mode. Default off preserves the
    # unchanged (Phase 1) GPU runner.
    coord_mode: bool = False
    barrier_interval_cycles: int = 4
    # Number of members expected at each device barrier. 0 means "all members"
    # (dp_size). A smaller M lets a fixed subset of members run the coordinated
    # decode barrier while the other (dp_size - M) members stay idle -- used to
    # validate partial participation before dynamic nptr exists. The idle members
    # must not receive decode requests, or they would be expected at the barrier.
    barrier_nptr: int = 0
    # Dynamic nptr: when True, the live barrier participant count is decided by a
    # per-forward host rendezvous (members vote decode/idle) instead of the fixed
    # barrier_nptr. Lets members exit/join at forward boundaries without deadlock.
    coord_dynamic_nptr: bool = False
    # Schedule-consistency observability (read-only). When True, each member
    # appends one jsonl row per rendezvous round recording its scheduling
    # decision (mode/running/waiting/kv/retract/eos) keyed by the global round id,
    # so an offline tool can align all members per round and find prefill/decode
    # divergence. Pure logging: does not change any scheduling or barrier behavior.
    coord_observe: bool = False
    coord_observe_dir: str = "check_logs/sidp_sched_trace"
    # Unified scheduling (Direction A): when True, all members agree on a single
    # forward mode (all-decode or all-prefill) each round via the rendezvous, so
    # prefill/decode never mix across ranks (which would stall the fast decoders
    # at the host round barrier). Policy: decode-wins (defer prefill while any
    # rank has running to decode); force a prefill window when a rank has no
    # running left (would idle otherwise) or drops below the low watermark.
    # Requires coord_mode + coord_dynamic_nptr (needs the rendezvous channel).
    coord_unified_schedule: bool = False
    # running low-watermark (fraction of max_running_requests) below which a rank
    # that wants prefill forces a group prefill window (soft anti-starvation). 0
    # means only force prefill when running is fully drained.
    coord_prefill_low_watermark: float = 0.15
    # Ablation: when True, coord_mode keeps the host-side rendezvous (per-forward
    # vote + peak-shifting phase alignment) but SKIPS launching the cross-member
    # device barrier. Used to measure whether host-only sync is enough or the
    # device barrier is actually pulling weight. Default False = normal barrier.
    coord_disable_device_barrier: bool = False
    num_layers: int = 0
    transfer_dtype: str = "same"
    enable_cycle_overlap: bool = False
    prefetch_policy: str = SidpPrefetchPolicy.COMPUTE.value
    copy_backend: str = SidpCopyBackend.DMA.value
    # ServerArgs resolves ``auto`` before constructing SidpConfig. The default
    # fixed compute-order DMA path uses generation flags; Event remains an
    # explicit compatibility/diagnostic mode.
    slot_sync: str = SidpSlotSync.FLAG.value
    # Fixed DMA can split every encoded component into an arbitrary positive
    # number of graph-stable byte ranges. Groups partition the cycle's ordered
    # remote layers before applying slice-major submission.
    dma_slices: int = 1
    dma_slice_groups: int = 1
    # ``rotating`` preserves the original fairness-oriented probe cursor.
    # ``compute_priority`` retries the earliest not-yet-filled layer first,
    # then falls through to later free owners rather than waiting in place.
    dynamic_claim_order: str = SidpDynamicClaimOrder.ROTATING.value
    # Effective decode mode after per-phase Graph resolution, not the legacy
    # CLI bool. Dynamic DMA supports Full decode Graph and eager prefill/fallback.
    disable_cuda_graph: bool = False
    # Validation-only control: keep the fixed-order SM copy backend but use
    # the legacy per-slot CUDA Event RAW/WAR protocol. This isolates the copy
    # backend cost from the generation-flag protocol used by normal SM modes.
    sm_use_event_sync: bool = False
    # Total CTAs launched by each SM-copy component kernel. Zero preserves the
    # bandwidth-oriented default (4 * device SM count); a positive value
    # throttles the copy kernel so model kernels retain more scheduler/SM room.
    sm_copy_ctas: int = 0
    # Compatibility input for out-of-tree callers. Runtime code uses the
    # resolved ``prefetch_policy`` above.
    enable_peak_shifting: bool = False
    enable_debug_logging: bool = False
    enable_graph_profiling: bool = False
    profile_dummy_compute: bool = False
    peak_sync_strategy: str = "none"
    peak_sync_min_raw_bs: int = 64
    peak_sync_max_replays: int = 0
    peak_sync_timeout_s: float = 30.0
    profile_sample_interval: int = 20
    profile_warmup_replays: int = 20
    profile_output_dir: str = "/tmp/sidp_profile"
