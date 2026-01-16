#!/usr/bin/env bash
set -euo pipefail

HIERFL_DIR="${1:-}"
if [[ -z "${HIERFL_DIR}" || ! -d "${HIERFL_DIR}" ]]; then
  echo "Usage: bash scripts/apply_overlay.sh /path/to/HierFL"
  exit 1
fi

echo "[apply_overlay] Copying DCAS overlay into: ${HIERFL_DIR}"

mkdir -p "${HIERFL_DIR}/src/dcas"
cp -r src/dcas/* "${HIERFL_DIR}/src/dcas/"
cp control_plane.py "${HIERFL_DIR}/"
cp -r scripts "${HIERFL_DIR}/" || true

echo "[apply_overlay] Done."
echo "Next: follow INTEGRATION.md to connect these modules to the training driver."
