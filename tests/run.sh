#!/bin/sh
# tests/run.sh — this repository's test entry point.
#
# Rule 19 item 4 requires the QA gate to be wired in here, so the default run
# FAILS unless fresh QA evidence exists for the work in flight. Everything else
# in the fast path is this project's own standing check, in the order the handoff
# lists them, cheapest first (handoff section 8, "Progress and staleness check").
#
# Fast path (seconds to a minute):
#   1. the fourteen recorded digests (SHA256SUMS)
#   2. the DQVD container's structure and hashes (gate 1b's input)
#   3. the retrieval scorer's self-test (8 hand-checked cases)
#   4. loader gate 3: 1376 tensors agree with the writer's manifest, 8 tampered
#      copies are refused by name
#   5. gate 22: the vision mmproj (phase 8) — the writer is deterministic, and
#      THIS project's own GGUF reader reads the file back: 211 tensors agree
#      with the writer's manifest and the 19 metadata keys are the recorded
#      gemma4v contract. The frozen vision reference (tests/vision/reference/)
#      is intact and its two fixtures differ. Nothing here executes llama.cpp;
#      the pin's own reproduction of the frozen numbers is opt-in
#      (tools/eg2_vision_anchor.sh) and is superseded by gate 23.
#   6. gate 23: the vision TOWER (phase 8, unit 23) — the port's own
#      preprocessing checked before any matmul, then its 88x512 soft tokens for
#      both frozen fixtures against tests/vision/reference, with the bound
#      declared and printed before the run (tools/eg2_vision_gate23.sh)
#   7. gate 24: the vision tower on the DEVICE (phase 8, unit 24) — both legs in
#      one process, the device leg against the CPU leg on the soft tokens and on
#      the three stage buffers, then the device leg against the frozen dumps
#      through gate 23's own comparison, then the four attribution flips
#      (tools/eg2_vision_gate24.sh).  It builds the CUDA object when a device
#      source changed: that is the one step here whose cost depends on the
#      tree's CUDA build being current (about 90 s to rebuild ds4_cuda.o, about
#      40 s to link the oracle; measured warm run 58.9 s, i.e. the two image
#      runs with their kick probes plus the four attribution flips).
#   8. gate 25: the IMAGE SPLICE (phase 8, unit 25) — the port's sequence against
#      the frozen dumps' own header, then four legs: the splice on the BASELINE
#      weights with the reference's soft rows (bounded), the same on the port's
#      soft rows, the artifact's own leg (the 4-bit tier's loss, reported), and the
#      device leg against the CPU one (bounded); then the engine's own
#      probe/encode_memory entries (tools/eg2_splice_gate25.sh)
#   9. gate 26: the embedding WIRE (phase 8, unit 26) — the sequence the route
#      builds checked against the anchor's own 92 positions, then /health's image
#      contract, then an image through POST /v1/embeddings on the device leg
#      (two fixtures, a prefixed item, a text item), then the same requests on
#      the CPU leg and the two legs compared, then six refusals by name, then the
#      frozen-baseline cross-check (tools/eg2_embed_media_gate26.sh; it stops and
#      starts one server per backend, so the port is busy for its run)
#  10. gate 27: the BATCH (unit 27) — a mixed-length set and a 64-small-item set
#      through ONE tower call against the per-item calls (byte-identical), the
#      8192-row budget's boundary and the malformed block tables refused by name
#      at the device entry, then the wire: three items as one request against
#      three single requests on both backends, and four items of exactly 2048
#      tokens (the 8192-row budget) accepted while one item more is a 400
#      (tools/eg2_embed_batch_gate27.sh; it also starts and stops a server per
#      backend, and its 8192-row call allocates the 912 MiB workspace size class)
#  11. tests/qa-gate.sh (rule 19; DEFERRED by the owner's recorded instruction
#      until the port is code-complete — qa-evidence/qa-deferral.txt)
#
# Deep path (tests/run.sh --deep, about half an hour):
#  12. the artifact gate 1b (byte-identical to the container, metadata read back)
#  13. the Rust suite (cargo test -p ds4-core) in the nested ds4-rs checkout
#  14. gate 4, the CPU reference tower against the frozen anchor (about 20 min)
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

step "5. gate 22: the vision mmproj, read back by our own reader (phase 8)"
bash tools/eg2_mmproj_check.sh

step "6. gate 23: the vision tower against the frozen reference (phase 8)"
bash tools/eg2_vision_gate23.sh

step "7. gate 24: the vision tower on the DEVICE against the CPU leg (phase 8)"
bash tools/eg2_vision_gate24.sh

step "8. gate 25: the image splice and the engine entry (phase 8)"
bash tools/eg2_splice_gate25.sh

step "9. gate 26: the embedding wire, an image through /v1/embeddings (phase 8)"
bash tools/eg2_embed_media_gate26.sh

step "10. gate 27: several items in one tower call, the batch (unit 27)"
bash tools/eg2_embed_batch_gate27.sh

step "11. rule 19 QA gate"
bash tests/qa-gate.sh

if [ "$DEEP" -eq 0 ]; then
    printf '\nfast path complete (deep path: tests/run.sh --deep)\n'
    exit 0
fi

step "12. artifact gate 1b"
tools/eg2_artifact_check.sh "$ART" "$MANIFEST" "$DQVD" \
    "${EG2_MODEL_DIR:-.models/EmbeddingGemma-2-914f7f8}" out/artifact-inventory.txt

step "13. the Rust suite (ds4-core)"
(cd ds4-rs && cargo test -q -p ds4-core)

step "14. gate 4: the CPU reference tower against the frozen anchor"
(cd ds4-rs && make ds4_cpu.o tests/parity/eg2vq_embd_c_oracle >/dev/null)
EG2_TOK_REF_LOG=out/tok-anchor-verbose.log tools/eg2_phase4_gate.sh \
    "$ART" "$BASELINE" tests/anchor/anchor-items.tsv \
    "${EG2_MODEL_DIR:-.models/EmbeddingGemma-2-914f7f8}/config_sentence_transformers.json" \
    | tail -25

printf '\ndeep path complete\n'
