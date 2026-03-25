"""
Unit tests for TurboQuant core algorithm.

Tests the Hadamard transform, quantize/dequantize roundtrip,
and KV cache wrappers on GPU.
"""

import importlib.util
import os
import sys

import torch

# Direct import to avoid sglang's full package init (which needs sgl_kernel).
# Resolve path relative to this file: ../../srt/layers/quantization/turboquant_kernels.py
_kernels_path = os.path.join(
    os.path.dirname(__file__),
    "..", "srt", "layers", "quantization", "turboquant_kernels.py",
)
_spec = importlib.util.spec_from_file_location(
    "turboquant_kernels", os.path.abspath(_kernels_path)
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

CENTROIDS_1BIT = _mod.CENTROIDS_1BIT
CENTROIDS_2BIT = _mod.CENTROIDS_2BIT
CENTROIDS_3BIT = _mod.CENTROIDS_3BIT
CENTROIDS_4BIT = _mod.CENTROIDS_4BIT
HadamardTransform = _mod.HadamardTransform
_next_power_of_2 = _mod._next_power_of_2
compute_packed_dim = _mod.compute_packed_dim
compute_compression_ratio = _mod.compute_compression_ratio
pack_indices = _mod.pack_indices
unpack_indices = _mod.unpack_indices
turboquant_dequantize = _mod.turboquant_dequantize
turboquant_dequantize_kv_cache = _mod.turboquant_dequantize_kv_cache
turboquant_quantize = _mod.turboquant_quantize
turboquant_quantize_kv_cache = _mod.turboquant_quantize_kv_cache

DEVICE = torch.device("cuda")


def test_next_power_of_2():
    assert _next_power_of_2(1) == 1
    assert _next_power_of_2(2) == 2
    assert _next_power_of_2(3) == 4
    assert _next_power_of_2(127) == 128
    assert _next_power_of_2(128) == 128
    assert _next_power_of_2(129) == 256
    print("PASS: test_next_power_of_2")


def test_centroids_symmetry():
    """Centroids should be symmetric around 0."""
    for name, centroids in [
        ("1bit", CENTROIDS_1BIT),
        ("2bit", CENTROIDS_2BIT),
        ("3bit", CENTROIDS_3BIT),
        ("4bit", CENTROIDS_4BIT),
    ]:
        n = len(centroids)
        assert n == 2 ** int(name[0]), f"{name}: expected {2**int(name[0])} centroids, got {n}"
        for i in range(n // 2):
            assert abs(centroids[i] + centroids[n - 1 - i]) < 1e-4, (
                f"{name}: centroids[{i}]={centroids[i]} != -centroids[{n-1-i}]={centroids[n-1-i]}"
            )
    print("PASS: test_centroids_symmetry")


def test_hadamard_roundtrip():
    """forward -> inverse should recover original vector."""
    for dim in [64, 128, 256]:
        h = HadamardTransform(dim, seed=42, device=DEVICE)
        x = torch.randn(32, dim, device=DEVICE)
        y = h.forward(x)
        x_recon = h.inverse(y)
        err = (x.float() - x_recon.float()).norm() / x.float().norm()
        assert err < 1e-5, f"dim={dim}: Hadamard roundtrip error {err:.6e}"
    print("PASS: test_hadamard_roundtrip")


def test_hadamard_norm_preservation():
    """Orthogonal transform should preserve L2 norms."""
    h = HadamardTransform(128, seed=42, device=DEVICE)
    x = torch.randn(64, 128, device=DEVICE)
    y = h.forward(x)
    # y is padded_dim which == 128 here, so norm should be preserved
    x_norms = x.float().norm(dim=-1)
    y_norms = y.float().norm(dim=-1)
    rel_err = ((x_norms - y_norms).abs() / (x_norms + 1e-10)).max()
    assert rel_err < 1e-4, f"Norm preservation error: {rel_err:.6e}"
    print("PASS: test_hadamard_norm_preservation")


def test_hadamard_determinism():
    """Same seed should produce same transform."""
    h1 = HadamardTransform(128, seed=42, device=DEVICE)
    h2 = HadamardTransform(128, seed=42, device=DEVICE)
    x = torch.randn(16, 128, device=DEVICE)
    y1 = h1.forward(x)
    y2 = h2.forward(x)
    assert torch.allclose(y1, y2, atol=1e-6), "Same seed produces different results"
    print("PASS: test_hadamard_determinism")


def test_hadamard_different_seeds():
    """Different seeds should produce different transforms."""
    h1 = HadamardTransform(128, seed=42, device=DEVICE)
    h2 = HadamardTransform(128, seed=99, device=DEVICE)
    x = torch.randn(16, 128, device=DEVICE)
    y1 = h1.forward(x)
    y2 = h2.forward(x)
    assert not torch.allclose(y1, y2, atol=1e-3), "Different seeds produce same results"
    print("PASS: test_hadamard_different_seeds")


def test_pack_unpack_roundtrip():
    """Verify pack -> unpack is lossless for all bit widths."""
    padded_dim = 128
    n = 64
    for bits in [1, 2, 3, 4]:
        max_val = (1 << bits) - 1
        indices = torch.randint(0, max_val + 1, (n, padded_dim), dtype=torch.uint8, device=DEVICE)
        packed = pack_indices(indices, bits)
        expected_packed_dim = compute_packed_dim(padded_dim, bits)
        assert packed.shape == (n, expected_packed_dim), (
            f"bits={bits}: packed shape {packed.shape}, expected (64, {expected_packed_dim})"
        )
        unpacked = unpack_indices(packed, bits, padded_dim)
        assert torch.equal(indices, unpacked), f"bits={bits}: pack/unpack roundtrip failed"
    print("PASS: test_pack_unpack_roundtrip")


def test_compression_ratios():
    """Verify compression ratios match expectations."""
    head_dim = 128
    for bits, expected_min in [(4, 3.5), (3, 4.5), (2, 7.0), (1, 12.0)]:
        ratio = compute_compression_ratio(head_dim, bits, mode="mse")
        print(f"  {bits}-bit compression ratio: {ratio:.2f}x")
        assert ratio >= expected_min, f"bits={bits}: ratio {ratio:.2f}x < expected {expected_min}x"
    print("PASS: test_compression_ratios")


def test_quantize_dequantize_shapes():
    """Check output shapes of quantize/dequantize with bit-packing."""
    dim = 128
    n_tokens = 32
    h = HadamardTransform(dim, seed=42, device=DEVICE)
    x = torch.randn(n_tokens, dim, device=DEVICE)

    for bits in [1, 2, 3, 4]:
        q = turboquant_quantize(x, h, bits=bits, mode="mse")
        expected_packed = compute_packed_dim(128, bits)
        assert q["packed_indices"].shape == (n_tokens, expected_packed), (
            f"bits={bits}: packed shape {q['packed_indices'].shape}, expected ({n_tokens}, {expected_packed})"
        )
        assert q["packed_indices"].dtype == torch.uint8
        assert q["norms"].shape == (n_tokens,)
        assert q["padded_dim"] == 128
        assert "qjl_signs" not in q, "MSE mode should not have QJL"

        recon = turboquant_dequantize(q, h, bits=bits, mode="mse")
        assert recon.shape == (n_tokens, dim), f"bits={bits}: recon shape {recon.shape}"

    print("PASS: test_quantize_dequantize_shapes")


def test_quantize_dequantize_quality():
    """Roundtrip quality should improve with more bits."""
    dim = 128
    n_tokens = 256
    h = HadamardTransform(dim, seed=42, device=DEVICE)
    x = torch.randn(n_tokens, dim, device=DEVICE)

    errors = {}
    for bits in [1, 2, 3, 4]:
        q = turboquant_quantize(x, h, bits=bits, mode="mse")
        recon = turboquant_dequantize(q, h, bits=bits, mode="mse", output_dtype=torch.float32)
        # Relative MSE
        mse = ((x.float() - recon.float()) ** 2).mean()
        signal = (x.float() ** 2).mean()
        rel_mse = (mse / signal).item()
        errors[bits] = rel_mse
        print(f"  {bits}-bit relative MSE: {rel_mse:.6f}")

    # More bits should give lower error
    assert errors[4] < errors[3] < errors[2] < errors[1], (
        f"Error should decrease with bits: {errors}"
    )
    # 4-bit should be quite good (< 5% relative MSE)
    assert errors[4] < 0.05, f"4-bit relative MSE too high: {errors[4]:.6f}"
    print("PASS: test_quantize_dequantize_quality")


def test_quantize_prod_mode():
    """Test prod mode (QJL) produces expected packed outputs."""
    dim = 128
    n_tokens = 32
    h = HadamardTransform(dim, seed=42, device=DEVICE)
    x = torch.randn(n_tokens, dim, device=DEVICE)

    q = turboquant_quantize(x, h, bits=4, mode="prod")
    assert "qjl_signs" in q, "Prod mode should have QJL signs"
    assert "residual_norms" in q, "Prod mode should have residual norms"
    # QJL signs are packed at 1 bit: 128 coords / 8 = 16 bytes
    assert q["qjl_signs"].shape == (n_tokens, 16), f"QJL shape: {q['qjl_signs'].shape}"
    assert q["qjl_signs"].dtype == torch.uint8
    # MSE indices use bits-1=3 for prod mode: packed_dim = 128*3/8 = 48
    assert q["packed_indices"].shape == (n_tokens, 48), f"Packed shape: {q['packed_indices'].shape}"

    recon = turboquant_dequantize(q, h, bits=4, mode="prod", output_dtype=torch.float32)
    assert recon.shape == (n_tokens, dim)
    print("PASS: test_quantize_prod_mode")


def test_kv_cache_roundtrip():
    """Test the KV cache specific wrappers."""
    num_tokens = 16
    num_heads = 8
    head_dim = 128

    k = torch.randn(num_tokens, num_heads, head_dim, device=DEVICE)
    v = torch.randn(num_tokens, num_heads, head_dim, device=DEVICE)

    k_h = HadamardTransform(head_dim, seed=42, device=DEVICE)
    v_h = HadamardTransform(head_dim, seed=137, device=DEVICE)

    k_q, v_q = turboquant_quantize_kv_cache(k, v, k_h, v_h, bits=4, mode="mse")
    k_r, v_r = turboquant_dequantize_kv_cache(k_q, v_q, k_h, v_h, num_heads, bits=4, mode="mse")

    assert k_r.shape == k.shape, f"K shape mismatch: {k_r.shape} vs {k.shape}"
    assert v_r.shape == v.shape, f"V shape mismatch: {v_r.shape} vs {v.shape}"

    k_mse = ((k.float() - k_r.float()) ** 2).mean() / (k.float() ** 2).mean()
    v_mse = ((v.float() - v_r.float()) ** 2).mean() / (v.float() ** 2).mean()
    print(f"  KV cache K relative MSE: {k_mse:.6f}")
    print(f"  KV cache V relative MSE: {v_mse:.6f}")
    assert k_mse < 0.05, f"K relative MSE too high: {k_mse:.6f}"
    assert v_mse < 0.05, f"V relative MSE too high: {v_mse:.6f}"
    print("PASS: test_kv_cache_roundtrip")


def test_cosine_similarity():
    """Check cosine similarity between original and reconstructed vectors."""
    dim = 128
    n_tokens = 128
    h = HadamardTransform(dim, seed=42, device=DEVICE)
    x = torch.randn(n_tokens, dim, device=DEVICE)

    for bits in [1, 2, 3, 4]:
        q = turboquant_quantize(x, h, bits=bits, mode="mse")
        recon = turboquant_dequantize(q, h, bits=bits, mode="mse", output_dtype=torch.float32)

        cos_sim = torch.nn.functional.cosine_similarity(x.float(), recon.float(), dim=-1)
        mean_cos = cos_sim.mean().item()
        min_cos = cos_sim.min().item()
        print(f"  {bits}-bit cosine sim: mean={mean_cos:.6f}, min={min_cos:.6f}")
        if bits >= 3:
            assert mean_cos > 0.95, f"{bits}-bit mean cosine sim too low: {mean_cos:.6f}"

    print("PASS: test_cosine_similarity")


def test_zero_vectors():
    """TurboQuant should handle zero/near-zero vectors gracefully."""
    dim = 128
    h = HadamardTransform(dim, seed=42, device=DEVICE)
    x = torch.zeros(4, dim, device=DEVICE)
    x[1] = 1e-10  # near-zero

    q = turboquant_quantize(x, h, bits=4, mode="mse")
    recon = turboquant_dequantize(q, h, bits=4, mode="mse", output_dtype=torch.float32)
    # Zero input should give near-zero output
    assert recon[0].abs().max() < 1e-6, f"Zero vector reconstruction too large: {recon[0].abs().max()}"
    print("PASS: test_zero_vectors")


def test_index_range():
    """Packed indices should unpack to valid centroid range."""
    dim = 128
    n_tokens = 64
    h = HadamardTransform(dim, seed=42, device=DEVICE)
    x = torch.randn(n_tokens, dim, device=DEVICE)

    for bits in [1, 2, 3, 4]:
        q = turboquant_quantize(x, h, bits=bits, mode="mse")
        # Unpack to verify index range
        unpacked = unpack_indices(q["packed_indices"], bits, 128)
        max_idx = unpacked.max().item()
        num_centroids = 2 ** bits
        assert max_idx < num_centroids, (
            f"bits={bits}: max index {max_idx} >= num_centroids {num_centroids}"
        )
    print("PASS: test_index_range")


# ---------------------------------------------------------------------------
# Model-based E2E tests
# ---------------------------------------------------------------------------

MODELS_TO_TEST = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen2.5-1.5B-Instruct",
]

PROMPTS = [
    "The quick brown fox jumps over the lazy dog. " * 5,
    "Explain the theory of relativity in simple terms. Albert Einstein proposed "
    "two interrelated theories: special relativity and general relativity.",
]


def _run_model_test(model_id):
    """Load a model, compress its KV cache with TurboQuant, measure quality."""
    import gc
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda", trust_remote_code=True,
    )
    model.eval()

    config = model.config
    num_kv_heads = config.num_key_value_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    num_layers = config.num_hidden_layers

    k_h = HadamardTransform(head_dim, seed=42, device=DEVICE)
    v_h = HadamardTransform(head_dim, seed=137, device=DEVICE)

    print(f"  Model: {config.architectures[0]} | {num_layers}L, {num_kv_heads} KV heads, d={head_dim}")

    all_top1_ok = 0
    all_top1_total = 0

    for prompt in PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
        seq_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = model(**inputs, use_cache=True)
            past_kv = out.past_key_values

        # Extract KV list
        kv_list = []
        for layer_kv in past_kv:
            k, v = layer_kv[0], layer_kv[1]
            if k is not None and v is not None:
                kv_list.append((k, v))

        for bits in [3, 4]:
            # Compress all layers
            compressed = []
            for k, v in kv_list:
                b, h, s, d = k.shape
                k_flat = k.permute(0, 2, 1, 3).reshape(-1, d)
                v_flat = v.permute(0, 2, 1, 3).reshape(-1, d)
                k_q = turboquant_quantize(k_flat, k_h, bits, "mse")
                v_q = turboquant_quantize(v_flat, v_h, bits, "mse")
                k_q["_shape"] = (b, h, s, d)
                v_q["_shape"] = (b, h, s, d)
                compressed.append((k_q, v_q))

            # Decompress
            recon_kv = []
            for k_q, v_q in compressed:
                b, h, s, d = k_q["_shape"]
                k_r = turboquant_dequantize(k_q, k_h, bits, "mse", torch.bfloat16)
                v_r = turboquant_dequantize(v_q, v_h, bits, "mse", torch.bfloat16)
                k_r = k_r[:, :d].reshape(b, s, h, d).permute(0, 2, 1, 3)
                v_r = v_r[:, :d].reshape(b, s, h, d).permute(0, 2, 1, 3)
                recon_kv.append((k_r, v_r))

            # KV quality
            mse_vals = []
            for (ok, ov), (rk, rv) in zip(kv_list, recon_kv):
                for o, r in [(ok, rk), (ov, rv)]:
                    of = o.float().reshape(-1, o.shape[-1])
                    rf = r.float().reshape(-1, r.shape[-1])
                    mse_vals.append((((of - rf) ** 2).mean() / (of ** 2).mean()).item())
            mean_mse = sum(mse_vals) / len(mse_vals)

            # Next-token comparison
            recon_cache = DynamicCache()
            for li, (kt, vt) in enumerate(recon_kv):
                recon_cache.update(kt.contiguous(), vt.contiguous(), li)

            next_input = inputs["input_ids"][:, -1:]
            with torch.no_grad():
                orig_logits = model(next_input, past_key_values=past_kv, use_cache=False).logits[0, -1]
                recon_logits = model(next_input, past_key_values=recon_cache, use_cache=False).logits[0, -1]

            top1_match = orig_logits.argmax().item() == recon_logits.argmax().item()
            orig_top5 = set(orig_logits.topk(5).indices.tolist())
            recon_top5 = set(recon_logits.topk(5).indices.tolist())
            top5_overlap = len(orig_top5 & recon_top5)
            logit_cos = F.cosine_similarity(
                orig_logits.float().unsqueeze(0), recon_logits.float().unsqueeze(0)
            ).item()

            all_top1_total += 1
            if top1_match:
                all_top1_ok += 1

            print(f"    {bits}b | {seq_len}tok | MSE={mean_mse:.4f} | "
                  f"logit_cos={logit_cos:.3f} | top1={'OK' if top1_match else 'NO'} | top5={top5_overlap}/5")

            del compressed, recon_kv, recon_cache

    # At 4-bit, MSE should match paper bound
    assert mean_mse < 0.015, f"4-bit MSE too high: {mean_mse:.4f}"
    # At least some top-1 matches expected
    assert all_top1_ok > 0, f"No top-1 matches across any prompt/bits combo"

    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()


def test_model_qwen3_0_6b():
    """E2E test on Qwen3-0.6B (Qwen3ForCausalLM, 8 KV heads)."""
    _run_model_test("Qwen/Qwen3-0.6B")
    print("PASS: test_model_qwen3_0_6b")


def test_model_qwen2_5_1_5b():
    """E2E test on Qwen2.5-1.5B-Instruct (Qwen2ForCausalLM, 2 KV heads)."""
    _run_model_test("Qwen/Qwen2.5-1.5B-Instruct")
    print("PASS: test_model_qwen2_5_1_5b")


if __name__ == "__main__":
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print()

    unit_tests = [
        test_next_power_of_2,
        test_centroids_symmetry,
        test_hadamard_roundtrip,
        test_hadamard_norm_preservation,
        test_hadamard_determinism,
        test_hadamard_different_seeds,
        test_pack_unpack_roundtrip,
        test_compression_ratios,
        test_quantize_dequantize_shapes,
        test_quantize_dequantize_quality,
        test_quantize_prod_mode,
        test_kv_cache_roundtrip,
        test_cosine_similarity,
        test_zero_vectors,
        test_index_range,
    ]

    model_tests = [
        test_model_qwen3_0_6b,
        test_model_qwen2_5_1_5b,
    ]

    all_tests = unit_tests + model_tests

    passed = 0
    failed = 0
    for test in all_tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"FAIL: {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()

    print(f"{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(all_tests)}")
    if failed == 0:
        print("All tests passed!")
    else:
        sys.exit(1)
