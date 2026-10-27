#!/usr/bin/env bash
# ==============================================================================
# Ephemeral DVC Runner for Shared Research PC
# Keeps Client ID, Client Secret, and OAuth Tokens strictly in RAM & Temp storage.
# Automatically purges all temporary OAuth tokens and cache files on exit.
#
# Usage:
#   export DVC_GDRIVE_CLIENT_ID="<your_client_id>"
#   export DVC_GDRIVE_CLIENT_SECRET="<your_client_secret>"
#   ./code/current/mobilenet/bash_scripts/dvc_ephemeral.sh push
#
# Or as a one-liner:
#   DVC_GDRIVE_CLIENT_ID="..." DVC_GDRIVE_CLIENT_SECRET="..." \
#     ./code/current/mobilenet/bash_scripts/dvc_ephemeral.sh push
# ==============================================================================
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/paths.sh"

if [ -z "${DVC_GDRIVE_CLIENT_ID:-}" ] || [ -z "${DVC_GDRIVE_CLIENT_SECRET:-}" ]; then
    echo "[ERROR] Missing credentials in environment!"
    echo ""
    echo "Please provide DVC_GDRIVE_CLIENT_ID and DVC_GDRIVE_CLIENT_SECRET:"
    echo "  export DVC_GDRIVE_CLIENT_ID='your_client_id.apps.googleusercontent.com'"
    echo "  export DVC_GDRIVE_CLIENT_SECRET='your_client_secret'"
    echo "  $0 push"
    exit 1
fi

[ -x "${DVC_BIN}" ] || { echo "[ERROR] DVC not found or not executable: ${DVC_BIN}" >&2; exit 1; }

# Create a secure, isolated temporary directory for OAuth token storage
TEMP_CACHE="$(mktemp -d /tmp/dvc_ephemeral_XXXXXX)"
chmod 700 "${TEMP_CACHE}"

cleanup() {
    local exit_code=$?
    echo ""
    echo "[CLEANUP] Destroying ephemeral OAuth token and temp cache: ${TEMP_CACHE}"
    rm -rf "${TEMP_CACHE}"
    unset DVC_GDRIVE_CLIENT_ID DVC_GDRIVE_CLIENT_SECRET GDRIVE_CLIENT_ID GDRIVE_CLIENT_SECRET
    echo "[CLEANUP] Ephemeral session terminated. Zero credentials or tokens saved to disk."
    exit ${exit_code}
}
trap cleanup EXIT INT TERM

# PyDrive2 reads appdirs.user_cache_dir("pydrive2fs"), which respects XDG_CACHE_HOME
export XDG_CACHE_HOME="${TEMP_CACHE}"
export DVC_SITE_CACHE_DIR="${TEMP_CACHE}/dvc_site"
mkdir -p "${DVC_SITE_CACHE_DIR}"

# Provide both naming conventions for compatibility
export GDRIVE_CLIENT_ID="${DVC_GDRIVE_CLIENT_ID}"
export GDRIVE_CLIENT_SECRET="${DVC_GDRIVE_CLIENT_SECRET}"

echo "=================================================================="
echo "  Starting Ephemeral DVC Session"
echo "  Isolated Cache : ${TEMP_CACHE}"
echo "  Client ID      : ${DVC_GDRIVE_CLIENT_ID:0:16}..."
echo "  Action         : ${*:-push}"
echo "=================================================================="

if [ $# -eq 0 ]; then
    "${DVC_BIN}" push
else
    "${DVC_BIN}" "$@"
fi
