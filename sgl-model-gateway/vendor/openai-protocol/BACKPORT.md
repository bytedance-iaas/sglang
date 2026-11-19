# Deployed protocol compatibility backport

This is the unmodified crates.io `openai-protocol 1.0.0` source apart from the
two request-preservation changes below. Keep the gateway's existing API and
locked dependency graph while restoring engine metadata lost on forwarding.

- `StreamOptions.continuous_usage_stats` follows merged
  [smg-project/smg#1460](https://github.com/smg-project/smg/pull/1460), commit
  `20741ed1937af49897a7da56c8024efba68376db`. The field is optional and omitted
  when absent. The compatible construction site uses `StreamOptions::default()`.
- `ChatCompletionRequest.other` follows the flattened engine-extension map in
  [the upstream request schema](https://github.com/smg-project/smg/blob/6669d611f17dca691e93c1113faeeee59cf89ca2/crates/protocols/src/chat.rs).
  Unknown top-level engine fields, including `return_spec_tokens_details`,
  survive deserialize/serialize. Known typed fields keep existing validation.

[sgl-project/sglang#24608](https://github.com/sgl-project/sglang/issues/24608)
documents the same HTTP forwarding failure. Its November 2026 discussion notes
that the bundled gateway still pins `=1.0.0`, and a newer protocol version
requires migration. This backport avoids that migration and preserves the
existing gateway binary's routing, DP selection, and standard request semantics.

The source archive's license and attribution remain unchanged. This modifies
request metadata transport and does not change inference, EOS, model weights,
speculative decoding, CUDA Graph, or GPU topology.
