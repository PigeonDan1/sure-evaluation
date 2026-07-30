#!/usr/bin/env bash
set -euo pipefail

FUNASR_REPO="${FUNASR_REPO:-git@github.com:modelscope/FunASR.git}"
# Pin to a known-good commit of fun_text_processing.  To update, replace the
# hash and re-run `sure-eval env setup --node normalization/funasr_itn --force`.
FUNASR_COMMIT="${FUNASR_COMMIT:-d3982158b7e54f29923a139637eab9422bbf5369}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}/fun_text_processing"

# --- Clean previous environment ----------------------------------------------
echo "[funasr_itn] cleaning previous environment"
rm -rf "${SCRIPT_DIR}/.venv"
rm -rf "${TARGET_DIR}"

# --- Python virtualenv -------------------------------------------------------
echo "[funasr_itn] creating Python virtualenv"
cd "${SCRIPT_DIR}"
uv venv --python 3.11
uv sync

# --- fun_text_processing -----------------------------------------------------
BUILD_ROOT="$(mktemp -d /tmp/sure-funasr-itn-build.XXXXXX)"
cleanup() { rm -rf "${BUILD_ROOT}"; }
trap cleanup EXIT

echo "[funasr_itn] fetching fun_text_processing from ${FUNASR_REPO} @ ${FUNASR_COMMIT}"

git clone --depth 1 --filter=blob:none --sparse "${FUNASR_REPO}" "${BUILD_ROOT}/FunASR"
cd "${BUILD_ROOT}/FunASR"
git sparse-checkout set fun_text_processing
git checkout "${FUNASR_COMMIT}" 2>/dev/null || true

cp -r fun_text_processing "${TARGET_DIR}"
echo "${FUNASR_COMMIT}" > "${TARGET_DIR}/.git-funasr-commit"

echo "[funasr_itn] fun_text_processing installed at ${TARGET_DIR}"
