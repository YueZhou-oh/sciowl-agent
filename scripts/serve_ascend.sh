#!/usr/bin/env bash
# Run INSIDE your matched, supported vLLM-Ascend environment. Not tested on NPU here.
set -euo pipefail
: "${SCIOWL_MODEL_PATH:?Set SCIOWL_MODEL_PATH to exported HF checkpoint directory}"
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3}"
args=(serve "$SCIOWL_MODEL_PATH" --served-model-name SciOWL-Instruct
  --host "${SERVE_HOST:-127.0.0.1}" --port "${SERVE_PORT:-8000}"
  --tensor-parallel-size "${TP_SIZE:-4}" --max-model-len "${MAX_MODEL_LEN:-8192}"
  --max-num-seqs "${MAX_NUM_SEQS:-4}" --dtype bfloat16)
if [[ "${NATIVE_TOOL_CALLING:-1}" == 1 ]]; then
  args+=(--enable-auto-tool-choice --tool-call-parser hermes)
fi
if [[ -n "${SCIOWL_API_KEY:-}" ]]; then args+=(--api-key "$SCIOWL_API_KEY"); fi
exec vllm "${args[@]}"

