#!/usr/bin/env bash
# ==============================================================================
# DVC Convenience Wrapper
# Runs DVC using pt-env and ensures proper cache directory and links.
# Usage:
#   ./code/current/mobilenet/bash_scripts/dvc.sh <dvc commands...>
# Examples:
#   ./code/current/mobilenet/bash_scripts/dvc.sh add <path>
#   ./code/current/mobilenet/bash_scripts/dvc.sh push
#   ./code/current/mobilenet/bash_scripts/dvc.sh pull
#   ./code/current/mobilenet/bash_scripts/dvc.sh status
# ==============================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/paths.sh"

[ -x "${DVC_BIN}" ] || { echo "[ERROR] DVC not found or not executable: ${DVC_BIN}" >&2; exit 1; }

exec "${DVC_BIN}" "$@"
