#!/bin/bash
# Build an isolated restart copy of a decomposed LES case.
# usage: make_isolated_case.sh SRC_CASE DEST_CASE START_TIME
# - processor*/constant: hard links (read-only mesh), processor*/START: real copy
# - system/: copy, constant/: symlink (read-only at solve time)
set -euo pipefail
SRC="$1"; DEST="$2"; T="$3"
[ -d "$SRC/processor0/$T" ] || { echo "missing $SRC/processor0/$T" >&2; exit 1; }
[ ! -e "$DEST" ] || { echo "$DEST exists; refuse" >&2; exit 1; }
mkdir -p "$DEST/logs"
chmod 700 "$DEST"
cp -a "$SRC/system" "$DEST/system"
ln -s "$SRC/constant" "$DEST/constant"
for p in "$SRC"/processor*; do
  b=$(basename "$p")
  mkdir -p "$DEST/$b"
  cp -al "$p/constant" "$DEST/$b/constant"
  cp -a "$p/$T" "$DEST/$b/$T"
done
echo "built $DEST from $SRC t=$T"
