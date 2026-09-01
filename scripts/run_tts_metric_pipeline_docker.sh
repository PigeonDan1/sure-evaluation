#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CACHE_DIR="${SURE_EVALUATION_TTS_CACHE:-${REPO_ROOT}/runtime/cache/tts-metrics}"
WORK_DIR="${REPO_ROOT}/artifacts/tts_metric_pipeline"
ASR_FUNASR_IMAGE="${SURE_TTS_ASR_FUNASR_IMAGE:-docker.v2.aispeech.com/sjtu/sjtu_yukai-dujunhao-sure_funaudiollm__sensevoicesmall:v1.0}"
ASR_TTS_IMAGE="${SURE_TTS_ASR_TTS_IMAGE:-docker.v2.aispeech.com/sjtu/sjtu_yukai-wenbinhuang-asr-tts:eval-dnsmos}"
UTMOS_IMAGE="${SURE_TTS_UTMOS_IMAGE:-docker.v2.aispeech.com/sjtu/sjtu_yukai-yiweiguo-utmos:v1.0}"
OUTPUT=""
PREDICTION_AUDIO=""
REFERENCE_TEXT=""
REFERENCE_AUDIO=""
LANGUAGE="en"
GPU="0"
DEVICE="cuda:0"
SPEAKER_BACKENDS="wavlm-large,ecapa-tdnn,eres2net"
MOS_BACKENDS="dnsmos,wv-mos,utmos"
SEMANTIC_NORMALIZER=""
SKIP_SEMANTIC=0
PULL_IMAGES=1

usage() {
  cat <<'USAGE'
Usage:
  scripts/run_tts_metric_pipeline_docker.sh \
    --prediction-audio <wav> \
    --reference-text <text> \
    --reference-audio <wav/mp3> \
    --language zh \
    --output <merged.json>

Optional:
  --gpu 0
  --device cuda:0
  --cache-dir <cache-dir>
  --work-dir <partial-report-dir>
  --speaker-backends wavlm-large,ecapa-tdnn,eres2net
  --mos-backends dnsmos,wv-mos,utmos
  --semantic-normalizer wetext:zh_tn
  --skip-semantic
  --no-pull

Environment image overrides:
  SURE_TTS_ASR_FUNASR_IMAGE
  SURE_TTS_ASR_TTS_IMAGE
  SURE_TTS_UTMOS_IMAGE
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prediction-audio) PREDICTION_AUDIO="$2"; shift 2 ;;
    --reference-text) REFERENCE_TEXT="$2"; shift 2 ;;
    --reference-audio) REFERENCE_AUDIO="$2"; shift 2 ;;
    --language) LANGUAGE="$2"; shift 2 ;;
    --gpu) GPU="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --output) OUTPUT="$2"; shift 2 ;;
    --speaker-backends) SPEAKER_BACKENDS="$2"; shift 2 ;;
    --mos-backends) MOS_BACKENDS="$2"; shift 2 ;;
    --semantic-normalizer) SEMANTIC_NORMALIZER="$2"; shift 2 ;;
    --skip-semantic) SKIP_SEMANTIC=1; shift ;;
    --no-pull) PULL_IMAGES=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${PREDICTION_AUDIO}" || -z "${REFERENCE_TEXT}" || -z "${REFERENCE_AUDIO}" || -z "${OUTPUT}" ]]; then
  usage >&2
  exit 2
fi

to_hpc_path() {
  local path="$1"
  if [[ "${path}" == /mnt/cloudstorfs/* ]]; then
    printf '/hpc_stor03/%s\n' "${path#/mnt/cloudstorfs/}"
  else
    printf '%s\n' "${path}"
  fi
}

PREDICTION_AUDIO="$(to_hpc_path "${PREDICTION_AUDIO}")"
REFERENCE_AUDIO="$(to_hpc_path "${REFERENCE_AUDIO}")"
CACHE_DIR="$(to_hpc_path "${CACHE_DIR}")"
WORK_DIR="$(to_hpc_path "${WORK_DIR}")"
OUTPUT="$(to_hpc_path "${OUTPUT}")"

mkdir -p "${WORK_DIR}" "$(dirname "${OUTPUT}")"

contains_csv() {
  local csv="$1"
  local item="$2"
  [[ ",${csv}," == *",${item},"* ]]
}

uses_chinese_asr() {
  local language="${1,,}"
  [[ "${language}" == zh* || "${language}" == cmn* || "${language}" == yue* ]]
}

docker_base=(
  env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u ALL_PROXY -u all_proxy
  docker run --rm --gpus "device=${GPU}"
  -v /hpc_stor03:/hpc_stor03
  -w "${REPO_ROOT}"
  -e PYTHONPATH=src
)

pull_image_once() {
  local image="$1"
  local marker="${WORK_DIR}/.pulled.$(echo "${image}" | tr '/:' '__')"
  if [[ "${PULL_IMAGES}" -eq 0 || -e "${marker}" ]]; then
    return
  fi
  echo "[tts-metric] pull ${image}"
  env -u HTTP_PROXY -u HTTPS_PROXY -u http_proxy -u https_proxy -u ALL_PROXY -u all_proxy docker pull "${image}"
  touch "${marker}"
}

run_inner() {
  local image="$1"
  local output="$2"
  local speaker="$3"
  local mos="$4"
  local no_semantic="$5"
  shift 5
  pull_image_once "${image}"
  local semantic_normalizer_args=()
  if [[ -n "${SEMANTIC_NORMALIZER}" && -z "${no_semantic}" ]]; then
    semantic_normalizer_args=(--semantic-normalizer "${SEMANTIC_NORMALIZER}")
  fi
  "${docker_base[@]}" "$@" "${image}" \
    python scripts/run_tts_metric_pipeline.py \
      --prediction-audio "${PREDICTION_AUDIO}" \
      --reference-text "${REFERENCE_TEXT}" \
      --reference-audio "${REFERENCE_AUDIO}" \
      --language "${LANGUAGE}" \
      --device "${DEVICE}" \
      --cache-dir "${CACHE_DIR}" \
      --speaker-backends "${speaker}" \
      --mos-backends "${mos}" \
      --output "${output}" \
      "${semantic_normalizer_args[@]}" \
      ${no_semantic}
}

PARTS=()

if [[ "${SKIP_SEMANTIC}" -eq 0 ]]; then
  echo "[tts-metric] semantic"
  SEMANTIC_IMAGE="${ASR_TTS_IMAGE}"
  if uses_chinese_asr "${LANGUAGE}"; then
    SEMANTIC_IMAGE="${ASR_FUNASR_IMAGE}"
  fi
  run_inner \
    "${SEMANTIC_IMAGE}" \
    "${WORK_DIR}/semantic.json" "" "" "" \
    -e "MODELSCOPE_CACHE=${CACHE_DIR}/semantic/modelscope" \
    -e "HF_HOME=${CACHE_DIR}/semantic/huggingface" \
    -e "HF_HUB_CACHE=${CACHE_DIR}/semantic/huggingface/hub"
  PARTS+=("${WORK_DIR}/semantic.json")
fi

speaker_fast=""
for backend in wavlm-large ecapa-tdnn; do
  if contains_csv "${SPEAKER_BACKENDS}" "${backend}"; then
    if [[ -n "${speaker_fast}" ]]; then speaker_fast+=","; fi
    speaker_fast+="${backend}"
  fi
done
if [[ -n "${speaker_fast}" ]]; then
  echo "[tts-metric] speaker wavlm/ecapa"
  run_inner \
    "${ASR_TTS_IMAGE}" \
    "${WORK_DIR}/speaker_wavlm_ecapa.json" "${speaker_fast}" "" "--no-semantic" \
    -e "HF_HOME=${CACHE_DIR}/huggingface" \
    -e "HF_HUB_CACHE=${CACHE_DIR}/huggingface/hub" \
    -e "MODELSCOPE_CACHE=${CACHE_DIR}/modelscope" \
    -e TRITON_CACHE_DIR=/tmp/sure-eval-triton
  PARTS+=("${WORK_DIR}/speaker_wavlm_ecapa.json")
fi

if contains_csv "${SPEAKER_BACKENDS}" "eres2net"; then
  echo "[tts-metric] speaker eres2net"
  run_inner \
    "${ASR_FUNASR_IMAGE}" \
    "${WORK_DIR}/speaker_eres2net.json" "eres2net" "" "--no-semantic" \
    -e "MODELSCOPE_CACHE=${CACHE_DIR}/speaker/modelscope" \
    -e LD_LIBRARY_PATH=/usr/lib64:/opt/conda/lib \
    -v /usr/lib64/libsox.so:/usr/lib64/libsox.so:ro \
    -v /usr/lib64/libsox.so.3:/usr/lib64/libsox.so.3:ro \
    -v /usr/lib64/libsox.so.3.0.0:/usr/lib64/libsox.so.3.0.0:ro \
    -v /usr/lib64/libltdl.so.7:/usr/lib64/libltdl.so.7:ro \
    -v /usr/lib64/libltdl.so.7.3.1:/usr/lib64/libltdl.so.7.3.1:ro
  PARTS+=("${WORK_DIR}/speaker_eres2net.json")
fi

mos_fast=""
for backend in dnsmos wv-mos; do
  if contains_csv "${MOS_BACKENDS}" "${backend}"; then
    if [[ -n "${mos_fast}" ]]; then mos_fast+=","; fi
    mos_fast+="${backend}"
  fi
done
if [[ -n "${mos_fast}" ]]; then
  echo "[tts-metric] mos dnsmos/wv-mos"
  run_inner \
    "${ASR_TTS_IMAGE}" \
    "${WORK_DIR}/mos_dnsmos_wvmos.json" "" "${mos_fast}" "--no-semantic" \
    -e "HF_HOME=${CACHE_DIR}/mos/huggingface" \
    -e "HF_HUB_CACHE=${CACHE_DIR}/mos/huggingface/hub" \
    -e TRITON_CACHE_DIR=/tmp/sure-eval-triton
  PARTS+=("${WORK_DIR}/mos_dnsmos_wvmos.json")
fi

if contains_csv "${MOS_BACKENDS}" "utmos"; then
  echo "[tts-metric] mos utmos"
  run_inner \
    "${UTMOS_IMAGE}" \
    "${WORK_DIR}/mos_utmos.json" "" "utmos" "--no-semantic"
  PARTS+=("${WORK_DIR}/mos_utmos.json")
fi

"${REPO_ROOT}/.venv.hostbak/bin/python" - "$OUTPUT" "${PARTS[@]}" <<'PY'
import json
import sys
from pathlib import Path

output = Path(sys.argv[1])
parts = [Path(path) for path in sys.argv[2:]]
merged = {"sample": None, "ok": True, "metrics": {}, "errors": [], "source_reports": [str(p) for p in parts]}
for path in parts:
    data = json.loads(path.read_text(encoding="utf-8"))
    merged["sample"] = merged["sample"] or data.get("sample")
    merged["metrics"].update(data.get("metrics", {}))
    merged["errors"].extend(data.get("errors", []))
sim_metrics = {
    name: metric["score"]
    for name, metric in merged["metrics"].items()
    if name.startswith("sim/") and isinstance(metric, dict) and "score" in metric
}
if sim_metrics:
    merged["metrics"]["sim"] = {
        "metric_name": "sim",
        "score": sum(float(score) for score in sim_metrics.values()) / len(sim_metrics),
        "details": {"num_backends": len(sim_metrics), "backend_metrics": sim_metrics},
    }
merged["ok"] = not merged["errors"]
output.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({k: v["score"] for k, v in merged["metrics"].items()}, ensure_ascii=False, indent=2))
if merged["errors"]:
    print(json.dumps(merged["errors"], ensure_ascii=False, indent=2), file=sys.stderr)
    sys.exit(1)
PY

echo "[tts-metric] merged report: ${OUTPUT}"
