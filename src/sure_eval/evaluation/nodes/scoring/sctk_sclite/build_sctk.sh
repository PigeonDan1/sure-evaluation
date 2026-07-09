#!/usr/bin/env bash
set -euo pipefail

SCTK_REPO="${SCTK_REPO:-https://github.com/usnistgov/SCTK.git}"
SCTK_COMMIT="${SCTK_COMMIT:-9688a26882a688132a5e414cadcb4c19b6fffaba}"
SURE_EVAL_CACHE_DIR="${SURE_EVAL_CACHE_DIR:-${HOME}/.cache/sure-eval}"
PREFIX="${SURE_EVAL_CACHE_DIR}/tools/sctk/${SCTK_COMMIT}"
BUILD_ROOT=""

usage() {
  cat <<'USAGE'
Usage:
  build_sctk.sh [--prefix <install-dir>] [--commit <git-commit-or-tag>] [--build-root <dir>]

Builds NIST SCTK and installs sclite outside the main Python environment.

Default prefix:
  ${SURE_EVAL_CACHE_DIR:-${HOME}/.cache/sure-eval}/tools/sctk/<commit>

After installation, point the scorer at the binary with either:
  export SURE_EVAL_SCLITE_BIN=<prefix>/bin/sclite

or let the node discover the default cache path automatically.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --commit) SCTK_COMMIT="$2"; shift 2 ;;
    --build-root) BUILD_ROOT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${BUILD_ROOT}" ]]; then
  BUILD_ROOT="$(mktemp -d /tmp/sure-sctk-build.XXXXXX)"
  CLEAN_BUILD_ROOT=1
else
  mkdir -p "${BUILD_ROOT}"
  CLEAN_BUILD_ROOT=0
fi

cleanup() {
  if [[ "${CLEAN_BUILD_ROOT}" -eq 1 ]]; then
    rm -rf "${BUILD_ROOT}"
  fi
}
trap cleanup EXIT

mkdir -p "${PREFIX}"
echo "[sctk] repo: ${SCTK_REPO}"
echo "[sctk] commit: ${SCTK_COMMIT}"
echo "[sctk] prefix: ${PREFIX}"

git clone "${SCTK_REPO}" "${BUILD_ROOT}/SCTK"
cd "${BUILD_ROOT}/SCTK"
git checkout "${SCTK_COMMIT}"

make config
make all
make install PREFIX="${PREFIX}"

if [[ ! -x "${PREFIX}/bin/sclite" ]]; then
  echo "SCTK build finished, but ${PREFIX}/bin/sclite was not created or is not executable." >&2
  exit 1
fi

echo "[sctk] installed: ${PREFIX}/bin/sclite"
"${PREFIX}/bin/sclite" -v || true
