"""SiDP Direction A admission-control layer (host-side load balancing).

This package forms *cohorts* of shape-similar requests so the coordinated device
barrier does not stall on a slow member (barrier idle time = fast members waiting
on slow members). It runs entirely on the host, in front of the SiDP worker
services, and is independent of the GPU correctness path: dynamic nptr (Phase 2b)
already guarantees correctness for ANY arrival pattern, so this layer only
improves performance by making each cohort's per-step compute shapes match.

Modules:
  signature -- compute a request's shape_signature without tokenizing.
  cohort    -- assemble same-signature requests into cohorts (+ duplicate fill).
  proxy     -- standalone single-entry HTTP proxy: run via
               ``python -m sglang.srt.layers.sidp.admission.proxy``.

It lives under the sglang tree (not the repo root) so the remote-exec wrapper,
which force-syncs only ``sglang/``, ships it to the GPU host. It has no torch/CUDA
dependency and can be unit-tested locally (stdlib + PIL only).
"""
