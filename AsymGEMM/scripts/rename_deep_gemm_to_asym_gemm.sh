#!/usr/bin/env bash
set -euo pipefail

# Bulk rename helper:
#   - Replaces symbol/text in file contents
#   - Optionally renames file/dir paths that contain the old token
#
# Default behavior is DRY RUN.
# Use --apply to perform changes.
#
# Examples:
#   ./scripts/rename_asym_gemm_to_asym_gemm.sh
#   ./scripts/rename_asym_gemm_to_asym_gemm.sh --apply
#   ./scripts/rename_asym_gemm_to_asym_gemm.sh --apply --rename-paths

ROOT="$(pwd)"
FROM="asym_gemm"
TO="asym_gemm"
APPLY=0
RENAME_PATHS=1
INCLUDE_THIRD_PARTY=0

usage() {
  cat <<USAGE
Usage: $0 [options]

Options:
  --apply                 Apply changes (default: dry-run)
  --rename-paths          Rename files/directories containing the token (default: on)
  --no-rename-paths       Disable file/directory rename
  --root <path>           Project root (default: current directory)
  --from <text>           Source token (default: asym_gemm)
  --to <text>             Target token (default: asym_gemm)
  --include-third-party   Include third-party/ in content replacement
  -h, --help              Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply) APPLY=1; shift ;;
    --rename-paths) RENAME_PATHS=1; shift ;;
    --no-rename-paths) RENAME_PATHS=0; shift ;;
    --root) ROOT="$2"; shift 2 ;;
    --from) FROM="$2"; shift 2 ;;
    --to) TO="$2"; shift 2 ;;
    --include-third-party) INCLUDE_THIRD_PARTY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if ! command -v perl >/dev/null 2>&1; then
  echo "Error: perl is required." >&2
  exit 1
fi

if [[ ! -d "$ROOT" ]]; then
  echo "Error: root directory not found: $ROOT" >&2
  exit 1
fi

cd "$ROOT"

if command -v rg >/dev/null 2>&1; then
  RG_GLOBS=(
    -g '!/.git/*'
    -g '!/build/*'
    -g '!/dist/*'
    -g '!/*.ncu-rep'
    -g '!/*.ptx'
    -g '!/*.cubin'
  )
  if [[ "$INCLUDE_THIRD_PARTY" -eq 0 ]]; then
    RG_GLOBS+=( -g '!/third-party/*' )
  fi
  mapfile -t FILES < <(rg -l --hidden -F "$FROM" . "${RG_GLOBS[@]}")
else
  GREP_ARGS=(
    -RIl --binary-files=without-match
    --exclude-dir=.git
    --exclude-dir=build
    --exclude-dir=dist
    --exclude=*.ncu-rep
    --exclude=*.ptx
    --exclude=*.cubin
  )
  if [[ "$INCLUDE_THIRD_PARTY" -eq 0 ]]; then
    GREP_ARGS+=( --exclude-dir=third-party )
  fi
  mapfile -t FILES < <(grep "${GREP_ARGS[@]}" -- "$FROM" . || true)
fi

echo "Root: $ROOT"
echo "Replace: '$FROM' -> '$TO'"
echo "Mode: $([[ "$APPLY" -eq 1 ]] && echo APPLY || echo DRY-RUN)"
echo "Path rename: $([[ "$RENAME_PATHS" -eq 1 ]] && echo enabled || echo disabled)"
echo "Matched files: ${#FILES[@]}"

if [[ ${#FILES[@]} -gt 0 ]]; then
  printf '%s\n' "${FILES[@]}"
fi

if [[ "$APPLY" -eq 1 && ${#FILES[@]} -gt 0 ]]; then
  for f in "${FILES[@]}"; do
    perl -0777 -i -pe "s/\Q$FROM\E/$TO/g" "$f"
  done
  echo "Content replacement complete."
fi

if [[ "$RENAME_PATHS" -eq 1 ]]; then
  mapfile -t PATHS < <(find . -depth -name "*${FROM}*" \
    -not -path './.git/*' \
    -not -path './build/*' \
    -not -path './dist/*' \
    $( [[ "$INCLUDE_THIRD_PARTY" -eq 0 ]] && echo "-not -path './third-party/*'" ))

  echo "Matched paths: ${#PATHS[@]}"
  if [[ ${#PATHS[@]} -gt 0 ]]; then
    printf '%s\n' "${PATHS[@]}"
  fi

  if [[ "$APPLY" -eq 1 && ${#PATHS[@]} -gt 0 ]]; then
    for p in "${PATHS[@]}"; do
      newp="${p//$FROM/$TO}"
      if [[ "$p" != "$newp" ]]; then
        mkdir -p "$(dirname "$newp")"
        mv "$p" "$newp"
      fi
    done
    echo "Path rename complete."
  fi
fi

if [[ "$APPLY" -eq 0 ]]; then
  echo "Dry-run only. Re-run with --apply to make changes."
fi
