"""POC adaptation of #48950: host capacity, allocation budget and KV scale IDs."""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.srt.arg_groups import hicache_hook
from sglang.srt.arg_groups.overrides import resolution_result
from sglang.srt.mem_cache import hicache_auto_size as sizing
from sglang.srt.mem_cache import kv_cache_configurator as configurator
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.deepseek_v4_memory_pool import DeepSeekV4TokenToKVPool
from sglang.srt.mem_cache.hybrid_cache import hybrid_pool_assembler as assembler
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.mem_cache.memory_pool_host import (
    DeepSeekV4PagedHostPool,
    DeepSeekV4StateHostPool,
)
from sglang.srt.mem_cache.pool_host import base
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.runtime_context import get_context, get_memory, get_schedule
from sglang.srt.server_args import ServerArgs, prepare_server_args
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _mha(size, width, *, rows=None, host_bytes=None):
    rows = size if rows is None else rows
    return SimpleNamespace(
        size=size,
        k_buffer=[torch.empty(rows, 1)],
        get_kv_size_bytes=lambda: (width * rows // 2, width * rows // 2),
        host_capacity_bytes=host_bytes,
    )


@pytest.mark.parametrize("ratio", [0, 0.05, 0.1])
def test_stock_split_is_unchanged(ratio):
    assert (
        assembler._swa_host_shares(110, _mha(1000, 100), _mha(100, 50), ratio) is None
    )


@pytest.mark.parametrize("rows,size", [(1000, 1000), (1000, 500)])
def test_fixed_budget_and_post_capture_rows(rows, size):
    full = _mha(size, 100, rows=rows)
    shares = assembler._swa_host_shares(110, full, _mha(size // 10, 50), 0.2)
    assert shares == pytest.approx((100, 10))
    assert sum(shares) == pytest.approx(110)


def test_mla_split_and_explicit_host_capacity():
    def mla(size, width):
        return SimpleNamespace(
            size=size,
            page_size=256,
            host_capacity_bytes=None,
            get_kv_size_bytes=lambda: width * (size + 256),
        )

    assert assembler._swa_host_shares(
        110, mla(1000, 100), mla(100, 50), 0.2
    ) == pytest.approx((100, 10))
    assert (
        assembler._swa_host_shares(
            110, _mha(1000, 100), _mha(100, 50, host_bytes=10), 0.2
        )
        is None
    )


def test_fixed_split_includes_packed_draft_bytes():
    # The draft is packed into the SWA host allocation, so its bytes affect the
    # split even when its device ring serves fewer slots than the target.
    shares = assembler._swa_host_shares(
        115,
        _mha(1000, 100),
        _mha(100, 50),
        0.2,
        mtp_swa_device_pools=(_mha(20, 25),),
    )
    assert shares == pytest.approx((100, 15))
    assert (shares[1] / 75) / (shares[0] / 100) == pytest.approx(0.2)


def test_generic_stack_uses_host_split_and_keeps_packed_draft_mapping():
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams

    draft = _mha(20, 25)
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        page_size=256,
        token_to_kv_pool_allocator=SimpleNamespace(swa_attn_allocator=object()),
        mtp_draft_device_pools=(SimpleNamespace(swa_kv_pool=draft),),
    )
    memory = SimpleNamespace(
        hicache_size=115,
        hicache_swa_host_ratio=0.2,
        enable_unified_memory=False,
        hicache_write_policy="write_through",
        hicache_io_backend="direct",
        hicache_host_memory_mode="cache",
    )
    with (
        patch.object(assembler, "get_memory", return_value=memory),
        patch.object(assembler, "build_hybrid_swa_group") as group,
        patch.object(assembler, "HybridCacheController"),
    ):
        assembler.build_hybrid_swa_stack(
            params=params,
            full_kv_pool=_mha(1000, 100),
            swa_kv_pool=_mha(100, 50),
            full_layer_mapping={0: 0},
            swa_layer_mapping={1: 0},
            load_cache_event=None,
            storage_backend=None,
            use_mla=False,
        )
    assert group.call_args.kwargs["kv_host_size"] == pytest.approx(100)
    assert group.call_args.kwargs["swa_host_size"] == pytest.approx(15)
    assert group.call_args.kwargs["mtp_swa_device_pools"] == (draft,)


@pytest.mark.parametrize(
    "ratio,expected", [(0, 5), (0.01, 5), (0.0625, 5), (0.2, 16), (1, 80)]
)
@pytest.mark.parametrize("swa_page", [128, 256])
def test_dsv4_host_ratio_preserves_full_and_device(ratio, expected, swa_page):
    pool = SimpleNamespace(size=4096, swa_size=256)
    allocator = SimpleNamespace(size_full=4096)
    params = SimpleNamespace(token_to_kv_pool_allocator=allocator)
    memory = SimpleNamespace(
        hicache_ratio=5, hicache_size=0, hicache_swa_host_ratio=ratio
    )
    before = vars(pool).copy(), vars(allocator).copy()
    with patch.object(assembler, "get_memory", return_value=memory):
        full, swa, c128 = assembler._deepseek_v4_num_host_pages(
            params=params,
            kvcache=pool,
            page_size=256,
            swa_page_size=swa_page,
        )
    assert full == c128 == 80
    assert swa == expected * (256 // swa_page)
    assert (vars(pool), vars(allocator)) == before


def test_dsv4_independent_c128_capacity_is_unchanged():
    allocator = SimpleNamespace(
        size_full=4096, c128_attn_allocator=SimpleNamespace(num_pages=3)
    )
    with patch.object(
        assembler,
        "get_memory",
        return_value=SimpleNamespace(
            hicache_ratio=5, hicache_size=0, hicache_swa_host_ratio=0.2
        ),
    ):
        assert assembler._deepseek_v4_num_host_pages(
            params=SimpleNamespace(token_to_kv_pool_allocator=allocator),
            kvcache=SimpleNamespace(size=4096, swa_size=256),
            page_size=256,
            swa_page_size=256,
        ) == (80, 16, 15)


def test_dsv4_fixed_size_still_rejected():
    with patch.object(
        assembler, "get_memory", return_value=SimpleNamespace(hicache_size=1)
    ):
        with pytest.raises(ValueError, match="does not support --hicache-size"):
            assembler._deepseek_v4_num_host_pages(
                params=SimpleNamespace(
                    token_to_kv_pool_allocator=SimpleNamespace(size_full=4096)
                ),
                kvcache=SimpleNamespace(size=4096, swa_size=256),
                page_size=256,
                swa_page_size=256,
            )


def test_fixed_snapshot_precedes_all_pins_and_books_each_pool():
    order = []
    free = base.HICACHE_HOST_MEMORY_RESERVE_BYTES + 400
    memory = SimpleNamespace(hicache_host_memory_fraction=None)
    with (
        patch.object(sizing, "get_memory", return_value=memory),
        patch.object(sizing.torch.distributed, "is_initialized", return_value=True),
        patch.object(
            sizing.torch.distributed,
            "barrier",
            side_effect=lambda **_: order.append("barrier"),
        ) as barrier,
        patch.object(
            sizing,
            "get_parallel",
            return_value=SimpleNamespace(
                world_group=SimpleNamespace(cpu_group="world-cpu")
            ),
        ),
        patch.object(
            base,
            "available_host_memory_bytes",
            side_effect=lambda **_: order.append("sample") or free,
        ) as sample,
        patch.object(base, "ranks_per_host", return_value=4),
    ):
        with sizing.auto_size_hicache(None, None, enabled=True):
            assert order == ["barrier", "sample", "barrier"]
            assert base.host_memory_budget_bytes(30) == 100
            assert base.host_memory_budget_bytes(50) == 70
            assert base.host_memory_budget_bytes() == 20
        assert base._host_memory_budget.get() is None
        sample.assert_called_once()
        assert barrier.call_count == 2
        assert all(c.kwargs == {"group": "world-cpu"} for c in barrier.call_args_list)


def test_fixed_snapshot_cleanup_propagates_errors_and_disabled_does_not_sample():
    with (
        patch.object(
            sizing,
            "get_memory",
            return_value=SimpleNamespace(hicache_host_memory_fraction=None),
        ),
        patch.object(sizing.torch.distributed, "is_initialized", return_value=False),
        patch.object(sizing, "host_memory_budget_bytes", return_value=100) as sample,
    ):
        with pytest.raises(RuntimeError, match="allocation failed"):
            with sizing.auto_size_hicache(None, None, enabled=True):
                raise RuntimeError("allocation failed")
        assert base._host_memory_budget.get() is None
        sample.assert_called_once()
        sample.reset_mock()
        with sizing.auto_size_hicache(None, None, enabled=False):
            assert base._host_memory_budget.get() is None
        sample.assert_not_called()


def test_dsv4_paged_and_state_mirrors_book_cumulative_budget():
    # Budgeting is independent of the Linux prefault/pinning implementation.
    state = SimpleNamespace(
        ring_size=2,
        kv_score_buffer=SimpleNamespace(kv_score=torch.zeros(8, 2, dtype=torch.uint8)),
    )
    with (
        patch(
            "sglang.srt.mem_cache.pool_host.common.alloc_mmap",
            side_effect=lambda dims, dtype: torch.empty(dims, dtype=dtype),
        ),
        base.host_memory_budget_scope(100),
    ):
        DeepSeekV4PagedHostPool(
            "swa", [torch.empty(8, 8, dtype=torch.uint8)], 8, 4, 256, pin_memory=False
        )
        assert base.host_memory_budget_bytes() == 68
        DeepSeekV4StateHostPool("state", [state], 3, 256, pin_memory=False)
        assert base.host_memory_budget_bytes() == 56
        with pytest.raises(
            ValueError, match="Not enough host memory for V4 paged pool"
        ):
            DeepSeekV4PagedHostPool(
                "draft",
                [torch.empty(8, 8, dtype=torch.uint8)],
                8,
                8,
                256,
                pin_memory=False,
            )
        assert base.host_memory_budget_bytes() == 56


@pytest.mark.parametrize(
    "full_ids,swa_ids", [([0, 2, 4], [1, 3, 5]), ([12, 14], [13, 15])]
)
def test_quantized_subpool_receives_global_and_local_layer_ids(full_ids, swa_ids):
    pool = SWAKVPool.__new__(SWAKVPool)
    pool.full_kv_pool, pool.swa_kv_pool = (
        Mock(spec=["set_kv_buffer"]),
        Mock(spec=["set_kv_buffer"]),
    )
    pool.layers_mapping = {g: (i, False) for i, g in enumerate(full_ids)}
    pool.layers_mapping.update({g: (i, True) for i, g in enumerate(swa_ids)})
    ids = torch.arange(4)
    loc = KVWriteLoc(ids, swa_loc=ids + 4, physical=True)
    for gid in full_ids + swa_ids:
        pool.set_kv_buffer(
            SimpleNamespace(layer_id=gid), loc, torch.zeros(4), torch.ones(4)
        )
    for subpool, globals_, locations in (
        (pool.full_kv_pool, full_ids, ids),
        (pool.swa_kv_pool, swa_ids, ids + 4),
    ):
        assert len(subpool.set_kv_buffer.call_args_list) == len(globals_)
        for local, (gid, call) in enumerate(
            zip(globals_, subpool.set_kv_buffer.call_args_list)
        ):
            assert call.args[0].layer_id == gid
            assert call.kwargs["layer_id_override"] == local
            torch.testing.assert_close(call.args[1], locations)


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("draft_full_capacity", [False, True])
def test_hybrid_builder_connects_quant_method_without_resizing_draft_contract(
    quantized, draft_full_capacity
):
    quant = object() if quantized else None
    builder = SimpleNamespace(
        _build_mha_quant_method=Mock(return_value=quant),
        layer_info=SimpleNamespace(
            num_effective_layers=6,
            swa_attention_layer_ids=[1, 3, 5],
            full_attention_layer_ids=[0, 2, 4],
        ),
        is_hybrid_swa_compress=False,
        kv_cache_dtype=torch.bfloat16,
        kv_cache_dtype_str="bf16",
        draft_swa_full_capacity=draft_full_capacity,
        pool_page_size=256,
        post_capture_kv_active=True,
        device="cpu",
        model_config=SimpleNamespace(get_num_kv_heads=lambda *_: 8, head_dim=128),
    )
    with (
        patch.object(
            configurator,
            "get_parallel",
            return_value=SimpleNamespace(attn_tp_size=4, attn_dcp_size=1),
        ),
        patch.object(
            configurator,
            "get_spec",
            return_value=SimpleNamespace(speculative_algorithm="DSPARK"),
        ),
        patch.object(
            configurator,
            "get_exec",
            return_value=SimpleNamespace(
                features=SimpleNamespace(enable_memory_saver=False)
            ),
        ),
        patch.object(configurator, "SWAKVPool") as pool,
    ):
        configurator.KVCacheConfigurator._build_hybrid_swa_kv_pool(
            builder,
            full_max_total_num_tokens=4096,
            swa_max_total_num_tokens=256,
            mha_pool_class=object,
        )
    builder._build_mha_quant_method.assert_called_once_with(num_layers=6)
    kwargs = pool.call_args.kwargs
    assert kwargs["quant_method"] is quant
    assert kwargs["post_capture_active"] is not quantized
    assert kwargs["size"] == 4096
    assert kwargs["size_swa"] == (4096 if draft_full_capacity else 256)


def test_hybrid_constructor_forwards_quant_method_to_both_subpools():
    quant = object()
    seen = []

    class Pool:
        def __init__(self, **kwargs):
            seen.append(kwargs)
            self.post_capture_active = False

        def get_kv_size_bytes(self):
            return (8, 8)

    pool = SWAKVPool(
        size=64,
        size_swa=16,
        page_size=1,
        dtype=torch.bfloat16,
        head_num=1,
        head_dim=8,
        swa_attention_layer_ids=[1],
        full_attention_layer_ids=[0],
        device="cpu",
        token_to_kv_pool_class=Pool,
        quant_method=quant,
    )
    assert pool.full_kv_pool is not pool.swa_kv_pool
    assert [kwargs["quant_method"] for kwargs in seen] == [quant, quant]
    assert [kwargs["size"] for kwargs in seen] == [64, 16]


@pytest.mark.parametrize("local_layers", [12, 28])
def test_dsv4_pp_stage_swa_state_and_packed_draft_share_host_capacity(local_layers):
    from sglang.srt.mem_cache.cache_init_params import CacheInitParams
    from sglang.srt.mem_cache.hicache_storage import PoolName

    swa_buffers = [torch.empty(2, 32, dtype=torch.uint8) for _ in range(local_layers)]
    draft_buffer = torch.empty(2, 32, dtype=torch.uint8)
    pool = SimpleNamespace(
        size=4096,
        swa_size=256,
        swa_page_size=256,
        swa_kv_pool=SimpleNamespace(
            kv_buffer=swa_buffers, page_size=256, bytes_per_page_padded=32
        ),
        c4_kv_pool=SimpleNamespace(
            kv_buffer=[torch.empty(2, 16, dtype=torch.uint8)], bytes_per_page_padded=16
        ),
        compress_state_pools={12: object()},
        indexer_compress_state_pools={12: object()},
    )
    allocator = SimpleNamespace(
        size_full=4096, swa_attn_allocator=SimpleNamespace(alloc=Mock(), free=Mock())
    )
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=allocator,
        page_size=256,
        tp_cache_group="tp",
        attn_cp_cache_group="cp4",
        pp_cache_group="pp2",
        mtp_draft_device_pools=(
            SimpleNamespace(
                swa_kv_pool=SimpleNamespace(
                    kv_buffer=[draft_buffer], page_size=256, bytes_per_page_padded=32
                )
            ),
        ),
    )
    mapping = assembler._DeepSeekV4LayerMappings(
        transfer_layer_id_max=local_layers,
        full={0: 0},
        swa={i: i for i in range(local_layers)},
        c4={0: 0},
        c128={},
        c4_state={0: 0},
        c4_state_global_layers=[12],
    )
    memory = SimpleNamespace(
        hicache_ratio=5,
        hicache_size=0,
        hicache_swa_host_ratio=0.2,
        hicache_mem_layout="page_first_direct",
        hicache_write_policy="write_through",
        hicache_io_backend="direct",
        hicache_host_memory_mode="cache",
    )
    with (
        patch.object(assembler, "get_memory", return_value=memory),
        patch.object(assembler, "_get_allocator_type", return_value="default"),
        patch.object(assembler, "LogicalHostPool"),
        patch.object(assembler, "DeepSeekV4PagedHostPool") as paged,
        patch.object(assembler, "DeepSeekV4StateHostPool") as state,
        patch.object(assembler, "build_pool_entry") as entry,
        patch.object(assembler, "HostPoolGroup"),
        patch.object(assembler, "HybridCacheController") as controller,
        patch.object(assembler, "_dsv4_low_ratio_entries", return_value=[]),
        patch.object(assembler, "_dsv4_indexer_regions", return_value=[]),
        patch.object(assembler, "_build_dsv4_rope_entry", return_value=None),
    ):
        assembler.build_deepseek_v4_hicache_stack(
            params=params,
            kvcache=pool,
            load_cache_event=None,
            storage_backend=None,
            layer_mappings=mapping,
        )
    swa_call = next(
        c.kwargs
        for c in paged.call_args_list
        if c.kwargs["pool_name"] == str(PoolName.SWA)
    )
    assert swa_call["num_host_pages"] == 16
    assert len(swa_call["device_buffers"]) == local_layers + 1
    assert swa_call["device_buffers"][-1] is draft_buffer
    assert [c.kwargs["num_host_pages"] for c in state.call_args_list] == [16, 16]
    swa_entry = next(
        c.kwargs for c in entry.call_args_list if c.kwargs["name"] == PoolName.SWA
    )
    assert swa_entry["layer_mapping"][local_layers] == local_layers
    assert swa_entry["transfer_layer_id_max"] == local_layers + 1
    assert controller.call_args.kwargs["attn_cp_group"] == "cp4"
    assert controller.call_args.kwargs["pp_group"] == "pp2"


def _ratio_args(**kwargs):
    args = ServerArgs(model_path="dummy", enable_hierarchical_cache=True, **kwargs)
    hicache_hook.handle_hicache_ratio_default(args)
    return args


@pytest.mark.parametrize("ratio", [-1, 1.01, float("nan"), float("inf")])
def test_bad_host_ratio_is_rejected_at_configuration_boundary(ratio):
    with pytest.raises(ValueError, match="must be in"):
        hicache_hook._validate_hicache_swa_host_ratio(
            _ratio_args(hicache_swa_host_ratio=ratio)
        )


def test_cli_and_dsv4_ratio_mode():
    args = prepare_server_args(
        ["--model-path", "dummy", "--hicache-swa-host-ratio", "0.2"]
    )
    assert args.hicache_swa_host_ratio == 0.2
    assert ServerArgs(model_path="dummy").hicache_swa_host_ratio == 0
    with (
        patch.object(
            hicache_hook,
            "model_config_of",
            return_value=SimpleNamespace(is_deepseek_v4_arch=True),
        ),
        patch("sglang.srt.configs.hybrid_arch.mambaish_config", return_value=None),
    ):
        args = _ratio_args(hicache_ratio=5, hicache_swa_host_ratio=0.2)
        hicache_hook.handle_hicache(args)
        assert resolution_result(args, "hicache_ratio") == 5
        assert resolution_result(args, "hicache_host_memory_fraction") is None


@pytest.mark.parametrize("host_ratio", [0, 0.2])
def test_dsv4_auto_size_uses_existing_pool_boundary(host_ratio):
    args = _ratio_args(hicache_swa_host_ratio=host_ratio)
    with (
        patch.object(
            hicache_hook,
            "model_config_of",
            return_value=SimpleNamespace(is_deepseek_v4_arch=True),
        ),
        patch("sglang.srt.configs.hybrid_arch.mambaish_config", return_value=None),
    ):
        hicache_hook.handle_hicache(args)
    fraction = resolution_result(args, "hicache_host_memory_fraction")
    assert fraction == 0.8
    # The existing estimator rejects the actual DSV4 pool type before reading
    # its buffers, so no device allocation or replacement estimator is needed.
    pool = object.__new__(DeepSeekV4TokenToKVPool)
    params = CacheInitParams(
        disable=False,
        req_to_token_pool=None,
        token_to_kv_pool_allocator=Mock(get_kvcache=Mock(return_value=pool)),
        page_size=256,
    )
    with get_context().override_server_args(
        hicache_swa_host_ratio=host_ratio, hicache_host_memory_fraction=fraction
    ):
        with pytest.raises(
            ValueError,
            match="HiCache auto-sizing does not support DeepSeekV4TokenToKVPool",
        ):
            with sizing.auto_size_hicache(params, None, enabled=True):
                pytest.fail("unsupported DSV4 auto-sizing reached host allocation")


def test_runtime_memory_bag_carries_new_flag_without_resizing_device_ratio():
    with get_context().override_server_args(
        hicache_swa_host_ratio=0.2,
        hicache_ratio=5,
        swa_full_tokens_ratio=0.05,
    ):
        assert get_memory().hicache_swa_host_ratio == 0.2
        assert get_memory().hicache_ratio == 5
        assert get_schedule().swa_full_tokens_ratio == 0.05


def test_generic_ratio_requires_fixed_budget_and_rejects_unsupported_layouts():
    with (
        patch.object(
            hicache_hook,
            "model_config_of",
            return_value=SimpleNamespace(is_deepseek_v4_arch=False),
        ),
        patch("sglang.srt.configs.hybrid_arch.mambaish_config", return_value=None),
    ):
        with pytest.raises(ValueError, match="needs --hicache-size"):
            hicache_hook.handle_hicache(
                _ratio_args(hicache_ratio=5, hicache_swa_host_ratio=0.2)
            )
        hicache_hook.handle_hicache(
            _ratio_args(hicache_size=100, hicache_swa_host_ratio=0.2)
        )
        with pytest.raises(
            ValueError, match="not supported with --enable-unified-memory"
        ):
            hicache_hook.handle_hicache(
                _ratio_args(
                    hicache_size=100,
                    hicache_swa_host_ratio=0.2,
                    enable_unified_memory=True,
                )
            )
        with patch(
            "sglang.srt.configs.hybrid_arch.mambaish_config", return_value=object()
        ):
            with pytest.raises(ValueError, match="hybrid linear-attention"):
                hicache_hook.handle_hicache(
                    _ratio_args(hicache_size=100, hicache_swa_host_ratio=0.2)
                )
