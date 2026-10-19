#!/bin/sh
# tests/run.sh — this repository's test entry point.
#
# Rule 19 item 4 requires the QA gate to be wired in here, so the default run
# FAILS unless fresh QA evidence exists for the work in flight. Everything else
# in the fast path is this project's own standing check, in the order the handoff
# lists them, cheapest first (handoff section 8, "Progress and staleness check").
#
# Fast path (seconds to a minute):
#   1. the five recorded digests (SHA256SUMS)
#   2. the DQVD container's structure and hashes (gate 1b's input)
#   3. the retrieval scorer's self-test (8 hand-checked cases)
#   4. loader gate 3: 1376 tensors agree with the writer's manifest, 8 tampered
#      copies are refused by name
#   5. gate 22: the vision oracle (phase 8) — the writer is deterministic, the
#      pin's own mtmd accepts the mmproj it writes (211 tensors), and the frozen
#      image reference reproduces for both fixtures
#   6. tests/qa-gate.sh (rule 19)
#
# Deep path (tests/run.sh --deep, about half an hour):
#   7. the artifact gate 1b (byte-identical to the container, metadata read back)
#   8. the Rust suite (cargo test -p ds4-core) in the nested ds4-rs checkout
#   9. gate 4, the CPU reference tower against the frozen anchor (about 20 min)
#
# Usage: tests/run.sh [--deep]
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"

ART=${EG2_ARTIFACT:-out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf}
MANIFEST=${EG2_MANIFEST:-out/artifact-manifest.tsv}
DQVD=${EG2_DQVD:-out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-dqvd.bin}
BASELINE=${EG2_BASELINE:-out/EmbeddingGemma-2-914f7f8-baseline-bf16.gguf}
DEEP=0
[ "${1:-}" = "--deep" ] && DEEP=1

step() { printf '\n== %s ==\n' "$*"; }
for f in "$ART" "$MANIFEST" "$DQVD"; do
    [ -e "$f" ] || { echo "tests/run.sh: missing $f (set EG2_ARTIFACT/EG2_MANIFEST/EG2_DQVD)" >&2; exit 2; }
done

step "1. the recorded digests"
sha256sum -c SHA256SUMS

step "2. the DQVD container's structure"
./bin/eg2_pack_cpu check "$DQVD"

step "3. the retrieval scorer's self-test"
tools/eg2_retrieval_score_selftest.sh

step "4. loader gate 3"
tools/eg2_loader_check.sh "$ART" "$MANIFEST"

step "5. gate 22: the vision oracle (phase 8)"
bash tools/eg2_mmproj_check.sh

step "6. rule 19 QA gate"
bash tests/qa-gate.sh

if [ "$DEEP" -eq 0 ]; then
    printf '\nfast path complete (deep path: tests/run.sh --deep)\n'
    exit 0
fi

step "7. artifact gate 1b"
tools/eg2_artifact_check.sh "$ART" "$MANIFEST" "$DQVD" \
    "${EG2_MODEL_DIR:-.models/EmbeddingGemma-2-914f7f8}" out/artifact-inventory.txt

step "8. the Rust suite (ds4-core)"
(cd ds4-rs && cargo test -q -p ds4-core)

step "9. gate 4: the CPU reference tower against the frozen anchor"
(cd ds4-rs && make ds4_cpu.o tests/parity/eg2vq_embd_c_oracle >/dev/null)
EG2_TOK_REF_LOG=out/tok-anchor-verbose.log tools/eg2_phase4_gate.sh \
    "$ART" "$BASELINE" tests/anchor/anchor-items.tsv \
    "${EG2_MODEL_DIR:-.models/EmbeddingGemma-2-914f7f8}/config_sentence_transformers.json" \
    | tail -25

printf '\ndeep path complete\n'
