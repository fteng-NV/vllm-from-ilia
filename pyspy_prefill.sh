#!/usr/bin/env bash
# Profile the GIL on a vLLM PREFILL worker to see whether the KV connector's
# background threads (NIXL poller + write senders / Mooncake sender) steal the
# GIL from the prefill engine thread.
#
# Run ON THE PREFILL NODE while a STEADY-STATE bench is in flight (skip warmup).
#
# Usage:  ./pyspy_prefill.sh <label> [duration_s]
#   label      : tag for the output file, e.g. nixl  or  mc
#   duration_s : capture window (default 30)
set -euo pipefail

LABEL="${1:?need a label, e.g. nixl or mc}"
DUR="${2:-30}"

PID="$(pgrep -f 'VLLM::Worker_TP0' | head -n1 || true)"
[[ -z "$PID" ]] && PID="$(pgrep -f 'VLLM::Worker' | head -n1 || true)"
[[ -z "$PID" ]] && { echo "No VLLM::Worker process found. Candidates:"; pgrep -af 'VLLM::' || true; exit 1; }

echo "Profiling PID=$PID label=$LABEL duration=${DUR}s -> pyspy_${LABEL}_gil.svg"
py-spy record --gil --pid "$PID" --duration "$DUR" --rate 200 --output "pyspy_${LABEL}_gil.svg"
