# Native HF3FS batch client

The optional `native` backend uses 3FS's `hf3fs_py_batch` extension. Build the
`src/lib/py/batch` CMake target from a 3FS revision providing this module and
make the extension and API library visible to the SGLang Python environment.
No runtime source-patching installer or Torch C++ extension is needed.

In the JSON selected by `SGLANG_HICACHE_HF3FS_CONFIG_PATH`, add:

```json
{
  "file_path_prefix": "/mnt/3fs/hicache/pages",
  "file_size": 68719476736,
  "numjobs": 16,
  "entries": 8,
  "read_entries": 32,
  "write_entries": 8,
  "io_backend": "native",
  "metadata_server_url": "http://METADATA_HOST:PORT"
}
```

Create the containing directory first. `io_backend` defaults to `python` for
compatibility. Each unspecified directional depth falls back to `entries`.
Mock clients remain available for CPU tests. Native readers and writers have
independent IOR/Iov allocations; the same `numjobs` executor is shared between
both directions, so their capacities are not a guarantee of simultaneous I/O.
The legacy Python client allocates both rings at the larger requested depth.

The native path retains CPU buffer exports, releases the GIL during batch I/O,
and waits for completion before returning. It still copies between registered
3FS staging memory and the SGLang host pool. It is not GPU-direct or zero-copy.
`client_timeout` remains a legacy Python-client setting: native I/O does not
return early on timeout while storage could still use the buffer. An unavailable
backend may therefore block a call or shutdown; an unrecoverable ring error
requires process recovery.

DeepSeek logical KV pools use a 4 KiB marker per logical page with an anchor
file capped at 4 GiB. Real KV payload remains in the separate component files.
Component page sizes come from the actual exported flat page, including its
dtype and all layers. Existing joint component hit policies still apply: an
anchor hit alone does not prove that all required KV components are present.
Use a new storage namespace when changing page geometry; existing files and
metadata are not migrated automatically.

For deployments affected by a burst of Mooncake registrations,
`SGLANG_MOONCAKE_REGISTER_SERIAL=1` opts into checked, sequential registration.
The default remains batch registration with its return status checked. This is
a startup mitigation, not a demonstrated fix for kernel/RDMA deadlocks.

## CPU verification

```sh
PYTHONPATH=python python test/registered/unit/mem_cache/test_hf3fs_batch_storage.py
```

Setting `HF3FS_TEST_DIRECTORY` to a disposable directory on a real 3FS mount
runs the same storage tests with native I/O instead of local mock files. The
test creates and deletes only its own temporary files. Tests cover mixed misses,
read/write depth separation, logical anchors and component geometry, sibling
I/O draining on failure, and shutdown ordering. These tests do not establish
model numerical equivalence or measure end-to-end inference throughput.

The HTTP metadata service and its existing eviction/availability guarantees
are unchanged. Native batching does not make the index highly available or
solve concurrent page reuse between index lookup and I/O.

## Optional aligned component slots

For a 224,640-byte C2 component page, add `"aligned_slot_page_sizes": [224640]`
to the native-backend JSON. This option defaults to an empty list and applies
only to v2 component pools with an exact matching exported payload size; the
logical anchor, v1 KV pool, and all other components keep their existing layout.
It requires a native binding supporting the optional `payload_size` argument.

Selected pages occupy 256 KiB slots. I/O length is rounded up to 4 KiB, so C2
writes/read requests transfer 225,280 bytes (640 extra bytes, about 0.285%) but
still report 224,640 logical bytes. The CPU staging buffer zeros write padding;
user tensors and model geometry are unchanged. With 1 MiB storage chunks, a
selected page no longer crosses a chunk boundary. Slot/file address space grows
by about 16.70% for C2; this is a capacity tradeoff, not free sparse capacity.
Other selected sizes must be in (0, 256 KiB]; measure their tradeoffs separately.

Both the component filename and metadata namespace gain the suffix
`slot256-4k-v1-p<PAYLOAD_BYTES>`. Existing packed files and indices remain intact.
Enabling the option starts a cold component index; existing packed hits are not
implicitly migrated. Disabling it selects the old file/index again, and keys
written only to the new layout miss safely. All clients sharing a cache must use
a consistent layout configuration. Do not delete either layout while readers
can still reference it. Publication follows successful full I/O completion;
short padded completions cannot publish a key as durable.

This changes the CPU storage path only. It does not enable GPUDirect, change
replication/checksums, or imply an inference-throughput improvement. Validate
the model path before enabling it in a running engine.
