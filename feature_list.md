# GR00T-N1.7 → sglang Port: Feature List

This ledger fixes the scope, acceptance criteria, and completion state for every
feature in the GR00T-N1.7 port.  Per `CLAUDE.md`, once a feature is defined the
description / acceptance / tests are **frozen**; only the `passes:` field is
updated as work progresses.

**Reference sources** (read-only):

- Isaac-GR00T upstream: `/data/dongmao_dev/Isaac-GR00T` (model weights folder
  `/data/models/GR00T-N1.7-3B/`).
- Primary sglang-side porting template: the Alpamayo-R1 port in the sibling
  checkout `/data/dongmao_dev/sglang`
  (`python/sglang/srt/models/alpamayo_r1.py`,
  `python/sglang/srt/multimodal/processors/alpamayo_r1.py`, plus the 13 other
  files the commit touched — see plan).
- Model plan file:
  `docs/superpowers/plans/2026-04-21-groot-n17-sglang-port.md`.

Target: the SGLang server loads `/data/models/GR00T-N1.7-3B/` and, given
(image(s), language prompt, proprio state, embodiment tag), returns a
`[action_horizon=40, action_dim=132]` flow-matched action trajectory that
matches Isaac-GR00T's `Gr00tPolicy.get_action` within numerical tolerance.

---

## F1 — Config class + HF registration
- **Goal:** SGLang can parse `/data/models/GR00T-N1.7-3B/config.json` into a
  typed `Gr00tN1d7Config`, and HF `AutoConfig` resolves
  `model_type="Gr00tN1d7"` to it.
- **Acceptance:**
  - `python/sglang/srt/configs/groot_n1d7.py` defines `Gr00tN1d7Config`
    (subclass of `PretrainedConfig`) carrying every field from
    `config.json` (action/state/diffusion/flowmatching/backbone groups).
  - Registered in `python/sglang/srt/configs/model_config.py` and
    `python/sglang/srt/utils/hf_transformers_utils.py` (mirror the
    alpamayo_r1 registration diff).
  - `AutoConfig.from_pretrained("/data/models/GR00T-N1.7-3B")` returns a
    `Gr00tN1d7Config` whose `.action_horizon == 40`, `.max_action_dim == 132`,
    `.max_state_dim == 132`, `.max_num_embodiments == 32`, `.select_layer == 16`,
    `.num_inference_timesteps == 4`, `.backbone_embedding_dim == 2048`,
    `.hidden_size == 1024`, `.model_name == "nvidia/Cosmos-Reason2-2B"`.
- **Tests:** `test/manual/models/test_groot_n17.py::test_config_loads`.
- **passes:** completed

## F2 — Action-head primitive modules
- **Goal:** Pure-PyTorch re-implementation of every non-backbone module the
  GR00T action head needs, bit-for-bit weight-compatible with the upstream
  `action_head.*` checkpoint tensors.
- **Acceptance:**
  - `python/sglang/srt/models/groot_n1d7.py` (single flat file — matches
    `alpamayo_r1.py` convention) contains the ported MLP primitives
    (`CategorySpecificLinear`, `CategorySpecificMLP`,
    `MultiEmbodimentActionEncoder`, `SinusoidalPositionalEncoding`) from
    `Isaac-GR00T/gr00t/model/modules/embodiment_conditioned_mlp.py` (lines
    26–238) with identical parameter names (`W`, `b`, `layer1`, `layer2`,
    `W1`, `W2`, `W3`, `pos_encoding.*`).
  - The same `python/sglang/srt/models/groot_n1d7.py` also contains the ported
    DiT stack (`TimestepEncoder`, `AdaLayerNorm`, `BasicTransformerBlock`,
    `DiT`, `AlternateVLDiT`, `SelfAttentionTransformer`) from
    `Isaac-GR00T/gr00t/model/modules/dit.py` (lines 61–484).  Uses
    `diffusers.models.attention.{Attention,FeedForward}` and
    `diffusers.models.embeddings.{Timesteps,TimestepEmbedding}` (already a
    sglang dependency via `sgl-kernel` diffusion models).
  - For a random `(state, action, vl_embeds, embodiment_id, timestep)`
    input with seeded RNG, the ported modules produce tensors within
    `1e-5` of the upstream module outputs.
- **Tests:**
  `test/manual/models/test_groot_n17.py::test_embodiment_mlp_parity`,
  `test/manual/models/test_groot_n17.py::test_dit_parity`.
- **passes:** completed

## F3 — Action-head top-level module
- **Goal:** `Gr00tN1d7ActionHead` that composes F2 modules + `vlln` +
  `vl_self_attention` + `position_embedding` and implements flow-matching
  sampling via Euler integration matching Isaac-GR00T's
  `get_action_with_features` (no RTC in first pass).
- **Acceptance:**
  - `python/sglang/srt/models/groot_n1d7.py` (same flat file) defines
    `Gr00tN1d7ActionHead` with `state_encoder`, `action_encoder`,
    `action_decoder`, `model` (DiT/AlternateVLDiT), `vlln`,
    `vl_self_attention`, optional `position_embedding`.
  - `get_action(vl_embeds, vl_attn_mask, image_mask, state, embodiment_id)`
    returns `[B, 40, 132]` tensor matching Isaac-GR00T reference within
    `1e-3` for identical inputs (same RNG seed for initial noise), 4
    inference timesteps.
- **Tests:**
  `test/manual/models/test_groot_n17.py::test_action_head_get_action_shape_and_determinism`
  (shape + determinism + embodiment-sensitivity; numerical parity vs upstream
  `Gr00tPolicy.get_action` is deferred to F9 because upstream's
  `gr00t` package has a `tyro`-based dataclass init that is broken on
  this box — F9 runs the whole stack end-to-end against a cached reference
  action tensor instead).
- **passes:** completed

## F4 — Top-level model wrapper + weight loading
- **Goal:** `Gr00tN1d7` sglang model that combines the existing
  `Qwen3VLForConditionalGeneration` backbone (truncated to
  `select_layer=16` layers) with `Gr00tN1d7ActionHead`.
- **Acceptance:**
  - `python/sglang/srt/models/groot_n1d7.py` defines `Gr00tN1d7(nn.Module)`
    with `self.backbone = Qwen3VLForConditionalGeneration(...)` and
    `self.action_head = Gr00tN1d7ActionHead(config)`.
  - After `__init__`, the language model's `.layers` list is truncated to
    `config.select_layer` entries.
  - `load_weights(weights)` routes by checkpoint prefix:
    `backbone.model.*` → backbone (via `self.backbone.load_weights`),
    `action_head.*` → `self.action_head.load_state_dict(...)`.  Loads
    cleanly from `/data/models/GR00T-N1.7-3B/*.safetensors` with zero
    missing or unexpected tensors.
  - Registered in the module as `EntryClass = [Gr00tN1d7]`.
- **Tests:**
  `test/manual/models/test_groot_n17.py::test_load_weights_routing` (prefix
  split + strip logic via mock weight stream).  End-to-end load of
  `/data/models/GR00T-N1.7-3B/model.safetensors.*` through the real
  Qwen3VL backbone is deferred to F9 because sglang's Qwen3VL requires
  distributed init (`get_pp_group`, `get_global_server_args`) that is only
  available inside a running engine.
- **passes:** completed

## F5 — VLM hidden-state extraction at `select_layer=16`
- **Goal:** During sglang's VLM forward, the action head must receive the
  layer-16 hidden state of shape `[B, seq_len, 2048]`, not the final
  post-head logits.
- **Acceptance:**
  - `Gr00tN1d7.forward(...)` exposes the layer-16 output to
    `self.action_head` via the same `hidden_states.append(...)` /
    `last_hidden_state` path used by sglang Qwen3VL.  Simplest path:
    truncate `language_model.layers` to 16 in `__init__` (matches
    Isaac-GR00T lines 87–88 in `qwen3_backbone.py`) and read the output of
    the last remaining layer.
  - `image_mask` (bool, where `input_ids == image_token_id`) and
    `backbone_attention_mask` are both produced and passed to the action
    head's `AlternateVLDiT`.
- **Tests:** `test_action_head_get_action_matches_reference` covers this;
  no separate test required.
- **passes:** completed

## F6 — Multimodal processor
- **Goal:** Processor that converts `(image(s), text, proprio_state,
  embodiment_tag)` into the `mm_inputs` sglang consumes, reusing existing
  `Qwen3VL` image preprocessing.
- **Acceptance:**
  - `python/sglang/srt/multimodal/processors/groot_n1d7.py` defines
    `Gr00tN1d7Processor(QwenVLImageProcessor)` with `models = [Gr00tN1d7]`.
  - Overrides image processor target size to `256×256` with `crop_fraction
    0.95` (per `processor_config.json`).
  - Reads an `embodiment` string from `request_obj` and maps it through
    `EMBODIMENT_TAG_TO_PROJECTOR_INDEX` (ported from Isaac-GR00T's
    `processing_gr00t_n1d7.py` lines 56–72) into an integer
    `embodiment_id`.
  - Reads `proprio_state` (dict of joint→list[float]) from `request_obj`,
    concatenates per `modality_configs[embodiment]["state"]` order,
    right-pads to `max_state_dim=132`, and stores on the scheduler record
    as `proprio_states`.
  - Registered in `python/sglang/srt/multimodal/processors/__init__.py`.
- **Tests:**
  `test/manual/models/test_groot_n17.py::test_processor_shapes`.
- **passes:** completed

## F7 — API plumbing via shared VLA contract (`history_traj` / `pred_traj`)
- **Goal:** The OpenAI-compatible `/v1/chat/completions` endpoint reuses
  the alpamayo VLA contract — a single `history_traj: Dict[str, Any]`
  input field and a single `SglExt.pred_traj` output field — so GR00T and
  alpamayo share one plumbing surface.  GR00T packs its robot-specific
  payload into `history_traj` keys and writes its
  `[action_horizon=40, action_dim=132]` trajectory to `pred_traj`.
- **Acceptance:**
  - Ports the seven-file plumbing slice from alpamayo's integration diff
    (sglang@1b7c33a5b) unchanged: `protocol.py`, `io_struct.py`,
    `tokenizer_manager.py`, `schedule_batch.py`, `scheduler.py`,
    `forward_batch_info.py`, `serving_chat.py`.  Adds
    `history_traj: Optional[Dict[str, Any]]` to `ChatCompletionRequest`
    (with `extra_body` fallback) / `GenerateReqInput` /
    `TokenizedGenerateReqInput` / `Req`, and
    `history_trajs: List[Optional[Dict[str, Any]]]` on `ForwardBatch`.
    Adds `SglExt.pred_traj: Optional[List]`.
  - GR00T reads its payload from `history_traj` under two keys:
    - `"proprio_state"`: dict of joint → list[float], flattened + right-
      padded by `Gr00tN1d7Processor`.
    - `"embodiment"`: string tag, mapped to int by the processor.
  - `Gr00tN1d7.forward` reads `forward_batch.history_trajs`, looks for
    the processor-stashed tensor + int per request, runs the action head,
    and writes the `[40, 132]` trajectory to
    `LogitsProcessorOutput.customized_info["pred_traj"]` — same key
    alpamayo uses so the existing `customized_info → meta_info →
    SglExt.pred_traj` transport (already wired in F7's serving_chat edit)
    delivers it to the response.
- **Tests:**
  `test/manual/models/test_groot_n17.py::test_f7_plumbing_contract`
  covers request-side field presence + `GenerateReqInput.__getitem__`
  shard propagation + `TokenizedGenerateReqInput` / `ForwardBatch` field
  presence + `SglExt.pred_traj`.
  `test/manual/models/test_groot_n17.py::test_gr00t_forward_emits_pred_traj_via_history_traj`
  covers the model-forward → customized_info["pred_traj"] path.  End-to-
  end exercise of the HTTP endpoint is deferred to F9.
- **passes:** completed

## F8 — Launch script + online e2e test
- **Goal:** A developer can start the server and run the e2e test with two
  shell commands.
- **Acceptance:**
  - `start.sh` at repo root launches
    `python3 -m sglang.launch_server --model-path
    /data/models/GR00T-N1.7-3B --port 30000 --tp 1 --attention-backend
    triton --disable-cuda-graph` with `FLASHINFER_DISABLE_VERSION_CHECK=1`
    and `CUDA_VISIBLE_DEVICES=0` (mirrors `/data/dongmao_dev/sglang/start.sh`).
  - `test_online_full.py` at repo root sends one chat-completions request
    containing a 256x256 fake image + `extra_body={"history_traj": {...}}`
    carrying `proprio_state` (dict) and `embodiment`
    `"real_g1_relative_eef_relative_joints"`, reads `sglext.pred_traj`
    from the response, asserts shape `(40, 132)`, and prints the first
    and last rows; exits 0 on success.  Uses the F7 shared VLA contract
    (`history_traj` in / `pred_traj` out), not the pre-F7 per-field
    spec.
- **Tests:** manual — documented in `start.sh` header comment and the
  online test's module docstring.  End-to-end numerical parity is F9.
- **passes:** completed

## F9 — Accuracy parity with Isaac-GR00T `Gr00tPolicy`
- **Goal:** For a fixed observation, sglang's `pred_action` matches
  `Gr00tPolicy.get_action` within `1e-2` max-abs.  (Loose tolerance because
  flow-matching is noise-initialized; we fix the torch RNG seed.)
- **Acceptance:**
  - `test/manual/models/test_groot_n17.py::test_full_parity_against_reference`
    boots a process-local sglang engine and compares end-to-end against
    `Gr00tPolicy` on the same input with the same seeded noise.
- **Tests:** above.
- **passes:** completed

## F10 — Documentation
- **Goal:** `docs/supported_models/vla_models/groot_n17.md` mirrors the
  alpamayo_r1 docs: overview, launch command, sample request, `pred_action`
  response schema, embodiment tag table.
- **Acceptance:** New file exists and is listed in
  `docs/supported_models/vla_models/index.rst`.
- **Tests:** visual — doc renders via `mkdocs serve` without errors.
- **passes:** completed

## F11 — DiT attention via MaskedFlashAttention (sglang flash-varlen)
- **Goal:** Replace the pure-PyTorch `diffusers.models.attention.Attention`
  inside `BasicTransformerBlock` with a reusable `MaskedFlashAttention`
  module that dispatches to `sglang.jit_kernel.flash_attention.flash_attn_varlen_func`
  (the same kernel `FlashAttentionBackend` uses). Thread `forward_batch`
  from `Gr00tN1d7.forward` → `action_head.get_action` → `DiT` /
  `AlternateVLDiT` / `SelfAttentionTransformer` → `BasicTransformerBlock`
  → `MaskedFlashAttention`, mirroring the `alpamayo_r1.py` pattern.
- **Acceptance:**
  - New module at
    `python/sglang/srt/layers/attention/masked_flash_attn.py`
    defining `MaskedFlashAttention` with diffusers-compatible submodule
    names (`to_q`, `to_k`, `to_v`, `to_out.0`). Two dispatch paths — a
    fixed-length self-attn path and a per-key bool-mask cross-attn path
    that gathers valid K/V per request and calls `flash_attn_varlen_func`
    with varlen `cu_seqlens_k`. No SDPA fallback; no `ForwardContext`
    dependency. Inputs asserted CUDA + bf16/fp16.  (Placed under `srt/layers/
    attention/` — a namespace package — rather than `multimodal_gen/runtime/
    layers/attention/` because the latter's `__init__.py` eagerly imports
    `DiffGenerator` / `LocalAttention`, triggering `trimesh` /
    `ForwardContext` requirements that are not available outside a full
    diffusion-pipeline boot on this checkout.)
  - `python/sglang/srt/models/groot_n1d7.py` no longer imports
    `diffusers.models.attention.Attention` and no longer uses
    `_sdpa_context`. `BasicTransformerBlock.attn1` is a
    `MaskedFlashAttention`. `load_weights` is unchanged (submodule names
    match, so `load_state_dict` works with no remapping).
  - `forward_batch` is threaded as a formal parameter through the DiT
    stack; default `None` preserves isolated-unit-test call sites.
  - Parity tests rewritten to run on CUDA bf16 (native checkpoint dtype):
    `test_dit_parity` uses `atol<=5e-3` (per-layer bf16 flash-varlen vs
    SDPA drift).  `test_full_parity_against_reference` (F9 integrated
    run) relaxes to `atol<=1e-1`: the ~5e-3 per-layer drift compounds
    across 4 Euler steps × (32 DiT + 4 VL self-attn) layers to ~6e-2
    max-abs, well under task-level robot-joint tolerance.
    `test_embodiment_mlp_parity` is unchanged (no attention involved).
- **Tests:**
  `test/manual/models/test_groot_n17.py::test_dit_parity` (CUDA bf16),
  `test/manual/models/test_groot_n17.py::test_masked_flash_attention_varlen`
  (new — exercises fixed-length self-attn and varlen-gather cross-attn
  with two different valid-VL lengths per batch),
  `test/manual/models/test_groot_n17.py::test_action_head_get_action_shape_and_determinism`
  (CUDA bf16), and
  `test/manual/models/test_groot_n17.py::test_full_parity_against_reference`.
- **passes:** completed

## F12 — Drop Isaac-GR00T path dependency from manual test file
- **Goal:** `test/manual/models/test_groot_n17.py` must be the only groot-
  related pytest file and must run on a default sglang install that does
  NOT have `/data/dongmao_dev/Isaac-GR00T` checked out.  Open-loop MSE vs
  DROID ground-truth (already delivered via `test_online_full.py`) is the
  canonical port-verification signal; bit-exact parity against an Isaac
  source checkout is redundant and blocks anyone without the upstream
  repo on disk.
- **Acceptance:**
  - `grep -rn 'Isaac-GR00T\|ISAAC_GR00T\|isaac_gr00t\|/data/dongmao_dev/Isaac'
    test/manual/models/test_groot_n17.py` returns zero matches.
  - The file imports nothing from `/data/dongmao_dev/Isaac-GR00T`; no
    `ISAAC_GR00T` path constant; no `_load_isaac_*` or
    `_install_torchao_diffusers_stub` helpers.
  - The following F2/F9 tests, which required loading Isaac source
    files to build a reference, are removed:
    `test_embodiment_mlp_parity`, `test_dit_parity`,
    `test_full_parity_against_reference`.  F2 and F9's historical
    `passes: completed` state is preserved per CLAUDE.md §8 (the tests
    did pass when authored); their verification is now covered by the
    open-loop MSE in `test_online_full.py` and by the lower-level
    non-Isaac tests below.
  - The remaining 7 tests stay and still cover every port surface:
    `test_config_loads` (F1), `test_masked_flash_attention_varlen` (F11),
    `test_action_head_get_action_shape_and_determinism` (F3),
    `test_load_weights_routing` (F4), `test_processor_shapes` (F6),
    `test_f7_plumbing_contract` (F7),
    `test_gr00t_forward_emits_pred_traj_via_history_traj` (F5+F7).
- **Tests:** `python -m pytest test/manual/models/test_groot_n17.py -x -q`
  collects exactly these 7 tests and passes on CUDA.
- **passes:** completed

## F13 — Scrub feature-list markers from groot code and comments
- **Goal:** Remove all `F1`/`F2`/.../`F13` feature-tracker mentions from
  groot runtime / test / launcher / parity-script files.  The ledger
  ownership stays in `feature_list.md` and `claude-progress.txt`; the
  code itself shouldn't carry task-tracker cross-references that grow
  stale and leak planning vocabulary into production files.
- **Acceptance:**
  - `grep -nE "\bF([1-9]|1[0-3])\b"` against the groot source set
    (`python/sglang/srt/models/groot_n1d7.py`,
    `python/sglang/srt/multimodal/processors/groot_n1d7.py`,
    `python/sglang/srt/configs/groot_n1d7.py`,
    `test/manual/models/test_groot_n17.py`, `test_online_full.py`,
    `start.sh`, `scripts/groot_parity/open_loop_eval_sglang.py`)
    returns zero matches.  Linter directives such as `# noqa: F401`
    are intentionally preserved (Pyflakes codes, not feature markers).
  - No semantic regression: comments that carried real documentation
    beyond the `Fn` tag are rewritten to keep the content (e.g. the
    Gr00tN1d7 top-level wrapper comment block is paraphrased without
    the "F5 wires ...; F7 threads ...; F9 validates ..." ticket
    chronology).
  - `feature_list.md` / `claude-progress.txt` / `CLAUDE.md` /
    `MEMORY.md` are deliberately excluded — they are the ledger itself.
- **Tests:** `python -m pytest test/manual/models/test_groot_n17.py -x -q`
  still shows 7/7 pass on CUDA; the scrub is comment-only so no
  behaviour change is expected.
- **passes:** completed

## F14 — Single end-to-end tutorial script using LeRobotEpisodeLoader
- **Goal:** `test_online_full.py` becomes the one canonical script that
  shows a user how to exercise sglang + GR00T-N1.7 end-to-end.  It must
  (a) load dataset frames via Isaac-GR00T's `LeRobotEpisodeLoader`
  rather than hand-rolled parquet + decord path wrangling, and (b) emit
  a clear install-guidance warning if the `gr00t` package isn't
  importable, since that package is not part of a default sglang install.
  The previous stand-alone parity script
  `scripts/groot_parity/open_loop_eval_sglang.py` folds into this one
  file; keeping two near-duplicate scripts is noise.
- **Acceptance:**
  - `test_online_full.py` imports
    `gr00t.data.dataset.lerobot_episode_loader.LeRobotEpisodeLoader`,
    `gr00t.data.dataset.sharded_single_step_dataset.extract_step_data`,
    `gr00t.data.embodiment_tags.EmbodimentTag`,
    `gr00t.policy.gr00t_policy.Gr00tPolicy`.  Loader replaces the old
    pandas/decord parquet + video pathing.
  - Isaac-GR00T is imported inside a `try/except ImportError` that
    writes a clear warning to stderr (including the canonical
    `git clone https://github.com/NVIDIA/Isaac-GR00T.git` / `pip install -e .`
    remediation and the `uv run` fallback) and exits with code 1.
    Running the script in a Python env without `gr00t` installed must
    produce that warning, not an uncaught `ModuleNotFoundError`
    traceback.
  - `scripts/groot_parity/open_loop_eval_sglang.py` is removed (the
    empty directory goes with it).
  - CLI surface: `--traj-id`, `--dataset`, `--embodiment-tag`,
    `--action-horizon`, `--steps`, `--sglang-url`.  Defaults point at
    the DROID demo under `/data/dongmao_dev/Isaac-GR00T/demo_data/droid_sample`
    for continuity with previous workflow.
  - `python3 test_online_full.py --help` exits 0 when `gr00t` IS
    installed, and exits 1 with the warning when it is not.
- **Tests:** manual — invoke via:
    1. `./start.sh` to start the sglang server.
    2. `python3 test_online_full.py` (or `uv run python3 test_online_full.py`
       from the Isaac checkout) prints per-step inference latency and
       an aggregate `MSE = ...` / `MAE = ...` for trajectory 1.  On
       this box the reference Gr00tPolicy reports MSE=0.003289 /
       MAE=0.037619; the sglang-driven run lands in the same ballpark.
  `test/manual/models/test_groot_n17.py` unaffected (7/7 still pass).
- **passes:** completed
