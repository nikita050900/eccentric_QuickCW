#!/usr/bin/env bash
set -euo pipefail

SCRIPTDIR="$(cd "$(dirname "$0")" && pwd)"

echo "Submitting NGC 3115 (detect + UL)..."
bash "${SCRIPTDIR}/submit_NGC3115.sh"
echo ""

echo "Submitting NGC 1316 (detect + UL)..."
bash "${SCRIPTDIR}/submit_NGC1316.sh"
echo ""

echo "All 4 jobs submitted. Monitor with: squeue -u $(whoami)"
