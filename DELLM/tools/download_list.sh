#!/usr/bin/env bash
# Linux/macOS: download every URL in a list (one per line), resuming partial files.
# Usage: bash tools/download_list.sh urls.txt data/raw/CICIoT2023
set -euo pipefail
list=${1:-urls.txt}; out=${2:-data/raw/CICIoT2023}
mkdir -p "$out"
while read -r url; do
  [ -z "$url" ] && continue
  wget -c --tries=5 -P "$out" "$url"
done < "$list"
