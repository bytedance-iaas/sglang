#!/usr/bin/env bash

set -euo pipefail

if command -v tosutil >/dev/null 2>&1; then
  tosutil version
  exit 0
fi

case "$(uname -m)" in
  x86_64) ;;
  *) echo "tosutil bootstrap supports x86_64 runners only" >&2; exit 1 ;;
esac

version=4.1.7
expected_sha256=de97013fbf179a88fcc8da5035617cbed07e1318871e9af6e076348e879d9386
binary=/tmp/tosutil
url=https://tos-tools.tos-cn-beijing.volces.com/linux/amd64/tosutil

curl --fail --show-error --location \
  --connect-timeout 20 --max-time 120 \
  --output "${binary}" "${url}"
printf '%s  %s\n' "${expected_sha256}" "${binary}" | sha256sum --check --strict
chmod +x "${binary}"
sudo install -m 0755 "${binary}" /usr/local/bin/tosutil
rm -f "${binary}"
tosutil version
