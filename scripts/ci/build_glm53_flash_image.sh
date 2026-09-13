#!/usr/bin/env bash
set -euo pipefail

repo_root="$(git rev-parse --show-toplevel)"
: "${GLM53_BUILD_DIR:?Set an empty build directory outside the checkout}"
: "${GITHUB_RUN_ID:?A unique GitHub run ID is required}"
: "${GITHUB_RUN_ATTEMPT:?A GitHub run attempt is required}"
[[ "$GITHUB_RUN_ID" =~ ^[0-9]+$ && "$GITHUB_RUN_ATTEMPT" =~ ^[0-9]+$ ]]
mkdir -p "$GLM53_BUILD_DIR"
python3 "$repo_root/scripts/ci/glm53_flash_image.py" prepare \
  --path "$GLM53_BUILD_DIR/context" | tee "$GLM53_BUILD_DIR/identity.json"
source_commit="$(git rev-parse HEAD)"
source_tree="$(git rev-parse 'HEAD^{tree}')"
manifest_sha256="$(sha256sum "$GLM53_BUILD_DIR/context/source.json")"
manifest_sha256="${manifest_sha256%% *}"
image_repo=iaas-gpu-cn-beijing.cr.volces.com/serving/sglang
image_tag="$image_repo:glm53-flash-${source_commit:0:12}-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}"
build_url="https://github.com/bytedance-iaas/sglang/actions/runs/${GITHUB_RUN_ID}"

HTTP_PROXY="${BUILD_DOWNLOAD_PROXY:-}" HTTPS_PROXY="${BUILD_DOWNLOAD_PROXY:-}" \
  http_proxy="${BUILD_DOWNLOAD_PROXY:-}" https_proxy="${BUILD_DOWNLOAD_PROXY:-}" \
  curl --fail --location --retry 3 --retry-all-errors --connect-timeout 30 --max-time 300 \
    https://files.pythonhosted.org/packages/7a/ad/6f9aa43796dc2f028cbe62b43c276dbce8091b955eb918d42a147d97b377/sgl_deep_gemm-0.1.7-py3-none-manylinux2014_x86_64.whl \
    -o "$GLM53_BUILD_DIR/context/deep-gemm.whl"
printf '%s  %s\n' f4e67086dc685ddcfcbb7833cc9770afd850cab23e173e77b9b18c19de0c2836 \
  "$GLM53_BUILD_DIR/context/deep-gemm.whl" | sha256sum --check --status

for attempt in 1 2; do
  if timeout --kill-after=30s 80m docker buildx build \
      --platform linux/amd64 --network host --progress plain --push \
      --provenance=false --sbom=false \
      --tag "$image_tag" \
      --build-arg "SOURCE_COMMIT=$source_commit" \
      --build-arg "SOURCE_TREE=$source_tree" \
      --build-arg "SOURCE_MANIFEST_SHA256=$manifest_sha256" \
      --build-arg "BUILD_URL=$build_url" \
      --build-arg "IMAGE_TAG=$image_tag" \
      --build-arg "HTTP_PROXY=${BUILD_DOWNLOAD_PROXY:-}" \
      --build-arg "HTTPS_PROXY=${BUILD_DOWNLOAD_PROXY:-}" \
      --build-arg "NO_PROXY=${NO_PROXY:-localhost,127.0.0.1,.cr.volces.com}" \
      --build-arg "no_proxy=${NO_PROXY:-localhost,127.0.0.1,.cr.volces.com}" \
      --metadata-file "$GLM53_BUILD_DIR/build-metadata.json" \
      "$GLM53_BUILD_DIR/context"; then
    break
  fi
  if [[ "$attempt" == 2 ]]; then
    exit 1
  fi
  printf 'Build/export attempt %s failed; retrying once with cached layers.\n' "$attempt" >&2
  sleep 5
done

digest="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["containerimage.digest"])' "$GLM53_BUILD_DIR/build-metadata.json")"
[[ "$digest" =~ ^sha256:[0-9a-f]{64}$ ]]
image_ref="$image_repo@$digest"
for attempt in 1 2 3; do
  if timeout 300 docker buildx imagetools inspect "$image_ref" --raw > "$GLM53_BUILD_DIR/registry-manifest.json"; then
    break
  fi
  [[ "$attempt" != 3 ]] || exit 1
  sleep 3
done
python3 - "$GLM53_BUILD_DIR/registry-manifest.json" "$digest" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
raw = Path(sys.argv[1]).read_bytes()
# imagetools appends a newline to the raw registry response.
assert any("sha256:" + hashlib.sha256(data).hexdigest() == sys.argv[2] for data in (raw, raw.rstrip(b"\n")))
manifest = json.loads(raw)
assert manifest["schemaVersion"] == 2 and "config" in manifest
PY
printf 'image_tag=%s\nimage_ref=%s\ncommit=%s\ntree=%s\nmanifest_sha256=%s\nbuild_url=%s\n' \
  "$image_tag" "$image_ref" "$source_commit" "$source_tree" "$manifest_sha256" "$build_url" \
  | tee "$GLM53_BUILD_DIR/result.env"
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  printf 'image_ref=%s\nimage_tag=%s\n' "$image_ref" "$image_tag" >> "$GITHUB_OUTPUT"
fi
if [[ -n "${GITHUB_STEP_SUMMARY:-}" ]]; then
  {
    printf '### GLM-5.3-Flash fixed-environment image\n\n'
    printf -- '- Commit: %s\n- Tree: %s\n- Source manifest: %s\n' "$source_commit" "$source_tree" "$manifest_sha256"
    printf -- '- Tag: %s\n- Immutable image: %s\n' "$image_tag" "$image_ref"
    printf '\nBuild/install validation passed; runtime imports require the registered task Pod. Full H20 PD inference and performance were not tested.\n'
  } >> "$GITHUB_STEP_SUMMARY"
fi
