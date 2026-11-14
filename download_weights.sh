#!/usr/bin/env bash
# Fetch SAM and DINOv2 checkpoints and verify each against weights.sha256.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$ROOT/artifacts/weights"
CHECKSUMS="$ROOT/weights.sha256"
mkdir -p "$DEST"

sha256_of() {
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$1" | awk '{print $1}'
  else
    shasum -a 256 "$1" | awk '{print $1}'
  fi
}

expected_hash() {
  awk -v name="$1" '$2 == name { print $1 }' "$CHECKSUMS"
}

fetch() {
  local url="$1" name="$2" expected actual tmp
  expected="$(expected_hash "$name")"
  if [[ ! "$expected" =~ ^[[:xdigit:]]{64}$ ]]; then
    echo "Missing or invalid SHA-256 entry for $name in weights.sha256" >&2
    return 1
  fi

  if [[ -f "$DEST/$name" ]]; then
    actual="$(sha256_of "$DEST/$name")"
    if [[ "$actual" == "$expected" ]]; then
      echo "[ok  ] $name"
      return
    fi
    echo "Checksum mismatch for existing $DEST/$name" >&2
    return 1
  fi

  tmp="$(mktemp "$DEST/.${name}.XXXXXX")"
  trap 'rm -f "$tmp"' RETURN
  echo "[get ] $name"
  curl --fail --location --retry 3 --progress-bar --output "$tmp" "$url"
  actual="$(sha256_of "$tmp")"
  if [[ "$actual" != "$expected" ]]; then
    echo "Checksum mismatch for downloaded $name (expected $expected, got $actual)" >&2
    return 1
  fi
  mv "$tmp" "$DEST/$name"
  trap - RETURN
}

fetch "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" "sam_vit_b_01ec64.pth"
fetch "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" "sam_vit_h_4b8939.pth"
fetch "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth" "dinov2_vitl14_pretrain.pth"

echo "Verified checkpoints in: $DEST"
