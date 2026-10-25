#!/usr/bin/env bash
# Fetch ANN_SIFT10K ("siftsmall") from the INRIA TEXMEX corpus into data/siftsmall
# and verify pinned SHA-256 checksums of the archive and of every file used.
# Source and license notes: DATA.md.
set -euo pipefail

URL="ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz"
ARCHIVE_SHA256="b8f1e59b20319ac44279d5251706909dd3a5b8ca5ce2a11ddb1e73902252770e"
DEST="${1:-data}"

sha256() {
  if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
  else shasum -a 256 "$1" | cut -d' ' -f1; fi
}

verify_files() {
  local ok=0
  while read -r want name; do
    [ -f "$DEST/siftsmall/$name" ] || return 1
    [ "$(sha256 "$DEST/siftsmall/$name")" = "$want" ] || { echo "checksum mismatch: $name" >&2; ok=1; }
  done <<'SUMS'
1e90414a3254361aba48a0d58e461f7661fb135dabb3da985290e94de8f60fe0 siftsmall_base.fvecs
095af38bda2741be721447ba97e9286fa918ece946be92a9ae133276eddc1a10 siftsmall_query.fvecs
b0717bf276111a166f14dcc2653fd04465b2afd6d34c72be641b622cf99f95ae siftsmall_groundtruth.ivecs
SUMS
  return $ok
}

if verify_files 2>/dev/null; then
  echo "data/siftsmall already present and verified"
  exit 0
fi

mkdir -p "$DEST"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
echo "downloading $URL (5.3 MB)"
curl -fsS --retry 3 --max-time 600 -o "$tmp/siftsmall.tar.gz" "$URL"
got="$(sha256 "$tmp/siftsmall.tar.gz")"
if [ "$got" != "$ARCHIVE_SHA256" ]; then
  echo "archive checksum mismatch: got $got, want $ARCHIVE_SHA256" >&2
  exit 1
fi
tar -xzf "$tmp/siftsmall.tar.gz" -C "$tmp"
mkdir -p "$DEST/siftsmall"
for f in siftsmall_base.fvecs siftsmall_query.fvecs siftsmall_groundtruth.ivecs; do
  mv "$tmp/siftsmall/$f" "$DEST/siftsmall/$f"
done
verify_files
echo "data/siftsmall fetched and verified"
