"""CPU HF3FS regressions; optionally use a real scratch mount via HF3FS_TEST_DIRECTORY."""

import atexit
import concurrent.futures
import json
import os
from pathlib import Path
import signal
import tempfile
import threading
import unittest
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.storage.hf3fs.mini_3fs_metadata_server import (
    Hf3fsLocalMetadataClient,
)
from sglang.srt.mem_cache.storage.hf3fs.storage_hf3fs import HiCacheHF3FS


class Pool:
    layout = "page_first"
    page_size = 2

    def __init__(self, width=64):
        self.buffer = torch.zeros((16, width), dtype=torch.uint8)

    def get_ksize_per_token(self):
        # Reproduces a paged component whose reported row is not a token size.
        return 3

    def get_dummy_flat_data_page(self):
        return torch.empty(self.buffer.shape[1], dtype=torch.uint8)

    def get_data_page(self, index, flat=True):
        return self.buffer[int(index) // self.page_size]

    def set_from_flat_data_page(self, index, data):
        self.get_data_page(index).copy_(data)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.getenv("HF3FS_TEST_DIRECTORY"))
        self.stores = []
        self.signals = {
            sig: signal.getsignal(sig)
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGQUIT)
        }

    def tearDown(self):
        for store in self.stores:
            store.close()
            atexit.unregister(store.close)
        for sig, handler in self.signals.items():
            signal.signal(sig, handler)
        self.temp.cleanup()

    def backend(self, page_bytes=64, **kwargs):
        real = bool(os.getenv("HF3FS_TEST_DIRECTORY"))
        store = HiCacheHF3FS(
            0,
            str(Path(self.temp.name) / f"pages-{len(self.stores)}"),
            1 << 20,
            2,
            page_bytes,
            2,
            5,
            torch.uint8,
            Hf3fsLocalMetadataClient(),
            is_mla_model=True,
            use_mock_client=not real,
            io_backend="native" if real else "python",
            **kwargs,
        )
        self.stores.append(store)
        pool = Pool(page_bytes)
        store.register_mem_pool_host(pool)
        return store, pool

    def test_mixed_misses_keep_destinations(self):
        store, pool = self.backend(read_entries=4, write_entries=2)
        pool.buffer[0].fill_(11)
        pool.buffer[1].fill_(29)
        self.assertEqual(store.batch_set_v1(["a", "b"], torch.arange(4)), [True, True])
        pool.buffer.fill_(123)
        self.assertEqual(
            store.batch_get_v1(["missing", "b", "missing2", "a"], torch.arange(8)),
            [False, True, False, True],
        )
        self.assertTrue(torch.all(pool.buffer[0] == 123))
        self.assertTrue(torch.all(pool.buffer[1] == 29))
        self.assertTrue(torch.all(pool.buffer[2] == 123))
        self.assertTrue(torch.all(pool.buffer[3] == 11))

    def test_read_and_write_batch_depths(self):
        store, pool = self.backend(read_entries=4, write_entries=2)
        counts = {"batch_read": [], "batch_write": []}
        for client in store.clients:
            for method in counts:
                original = getattr(client, method)

                def record(offsets, tensors, fn=original, direction=method):
                    counts[direction].append(len(offsets))
                    return fn(offsets, tensors)

                setattr(client, method, record)
        keys = [str(i) for i in range(7)]
        self.assertTrue(all(store.batch_set_v1(keys, torch.arange(14))))
        self.assertTrue(all(store.batch_get_v1(keys, torch.arange(14))))
        self.assertEqual(sorted(counts["batch_read"]), [3, 4])
        self.assertEqual(sorted(counts["batch_write"]), [1, 2, 2, 2])
        fallback, _ = self.backend(write_entries=3)
        self.assertEqual((fallback.read_entries, fallback.write_entries), (2, 3))

    def test_anchor_and_component_geometry(self):
        store, _ = self.backend(page_bytes=0)
        self.assertEqual(store.bytes_per_page, 4096)
        keys = ["p0", "p1", "p2"]
        indices = torch.arange(6)
        self.assertEqual(store.batch_set_v1(keys, indices), [True] * 3)
        self.assertEqual(store.batch_get_v1(keys, indices), [True] * 3)
        component = Pool(192)
        component.buffer[:3] = torch.arange(3).reshape(-1, 1) + 20
        expected = component.buffer.clone()
        store.register_mem_host_pool_v2(component, PoolName.SWA)
        self.assertEqual(store._pool_storage_ctx[PoolName.SWA].bytes_per_page, 192)
        transfer = PoolTransfer(PoolName.SWA, host_indices=indices, keys=keys)
        # Anchor alone never proves that the component exists.
        self.assertEqual(store.batch_exists_v2(keys, [transfer]).kv_hit_pages, 0)
        self.assertEqual(store.batch_set_v2([transfer])[PoolName.SWA], [True] * 3)
        component.buffer.zero_()
        self.assertEqual(store.batch_get_v2([transfer])[PoolName.SWA], [True] * 3)
        self.assertTrue(torch.equal(component.buffer, expected))
        self.assertEqual(store.batch_exists_v2(keys, [transfer]).kv_hit_pages, 3)

    @unittest.skipUnless(os.getenv("HF3FS_TEST_DIRECTORY"), "real native binding required")
    def test_aligned_c2_component_and_legacy_isolation(self):
        store, _ = self.backend(page_bytes=0, aligned_slot_page_sizes=[224640])
        pool = Pool(224640)
        pool.buffer[:3] = torch.arange(3).reshape(-1, 1) + 47
        expected = pool.buffer.clone()
        store.register_mem_host_pool_v2(pool, PoolName.SWA)
        ctx = store._pool_storage_ctx[PoolName.SWA]
        self.assertIn("slot256-4k-v1-p224640", ctx.namespace)
        self.assertEqual(ctx.bytes_per_page, 224640)
        keys, indices = ["c0", "c1", "c2"], torch.arange(6)
        transfer = PoolTransfer(PoolName.SWA, host_indices=indices, keys=keys)
        self.assertEqual(store.batch_set_v1(keys, indices), [True] * 3)
        self.assertEqual(store.batch_set_v2([transfer])[PoolName.SWA], [True] * 3)
        pool.buffer.zero_()
        self.assertEqual(store.batch_get_v2([transfer])[PoolName.SWA], [True] * 3)
        self.assertTrue(torch.equal(pool.buffer, expected))
        self.assertEqual(store.batch_exists_v2(keys, [transfer]).kv_hit_pages, 3)
        # Disabling the option reopens the legacy layout and its separate index.
        old_path = Path(f"{store.file_path}.{PoolName.SWA}")
        self.assertFalse(old_path.exists())
        store.aligned_slot_page_sizes = frozenset()
        for client in ctx.clients:
            client.close()
        store.register_mem_host_pool_v2(pool, PoolName.SWA)
        self.assertEqual(store.batch_exists_v2(keys, [transfer]).kv_hit_pages, 0)
        pool.buffer.fill_(81)
        self.assertEqual(store.batch_set_v2([transfer])[PoolName.SWA], [True] * 3)
        self.assertTrue(old_path.exists())
        # Reopen the new format; neither old writes nor metadata overwrite it.
        for client in store._pool_storage_ctx[PoolName.SWA].clients:
            client.close()
        store.aligned_slot_page_sizes = frozenset([224640])
        store.register_mem_host_pool_v2(pool, PoolName.SWA)
        pool.buffer.zero_()
        self.assertEqual(store.batch_get_v2([transfer])[PoolName.SWA], [True] * 3)
        self.assertTrue(torch.equal(pool.buffer, expected))

    def test_worker_failure_drains_siblings_before_return(self):
        store, _ = self.backend()
        entered, release, finished = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

        def fail(*args):
            raise RuntimeError("injected")

        def block(*args):
            entered.set()
            if not release.wait(10):
                raise TimeoutError("test release")
            finished.set()
            return [64]

        store.clients[0].batch_read = fail
        store.clients[1].batch_read = block
        with concurrent.futures.ThreadPoolExecutor(1) as executor:
            future = executor.submit(
                store._run_io,
                store.clients,
                "batch_read",
                [0, 64],
                [torch.empty(64, dtype=torch.uint8)] * 2,
                1,
            )
            self.assertTrue(entered.wait(5))
            try:
                self.assertFalse(future.done())
            finally:
                release.set()
            with self.assertRaisesRegex(RuntimeError, "injected"):
                future.result(timeout=10)
        self.assertTrue(finished.is_set())

    def test_close_waits_for_pending_worker(self):
        store, _ = self.backend()
        entered, release = threading.Event(), threading.Event()

        def pending():
            entered.set()
            self.assertTrue(release.wait(10))

        store.executor.submit(pending)
        self.assertTrue(entered.wait(5))
        with patch.object(
            store.clients[0], "close", wraps=store.clients[0].close
        ) as close:
            with concurrent.futures.ThreadPoolExecutor(1) as executor:
                future = executor.submit(store.close)
                try:
                    with self.assertRaises(concurrent.futures.TimeoutError):
                        future.result(timeout=0.1)
                    close.assert_not_called()
                finally:
                    release.set()
                future.result(timeout=10)
            close.assert_called_once()

    def test_config_and_validation(self):
        path = Path(self.temp.name) / "config.json"
        path.write_text(
            json.dumps(
                dict(
                    file_path_prefix=str(Path(self.temp.name) / "configured"),
                    file_size=1 << 20,
                    numjobs=2,
                    entries=2,
                    read_entries=4,
                    write_entries=1,
                    io_backend="native",
                )
            )
        )
        config = HiCacheStorageConfig(
            tp_rank=0,
            tp_size=1,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            is_mla_model=False,
            enable_storage_metrics=False,
            is_page_first_layout=True,
            model_name=None,
            extra_config={"use_mock_hf3fs_client": True},
        )
        with patch.dict(os.environ, {HiCacheHF3FS.default_env_var: str(path)}):
            store = HiCacheHF3FS.from_env_config(64, torch.uint8, config)
        self.stores.append(store)
        self.assertEqual(
            (store.read_entries, store.write_entries, store.io_backend),
            (4, 1, "native"),
        )
        with self.assertRaises(ValueError):
            self.backend(read_entries=0)
        with self.assertRaises(ValueError):
            self.backend(write_entries=4097)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
