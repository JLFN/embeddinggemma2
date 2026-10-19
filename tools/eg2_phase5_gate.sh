#!/bin/bash
# eg2_phase5_gate.sh — gate 5: what the VQ codec costs, in retrieval terms.
#
# Phase 5 of embeddinggemma2-handoff.md. The owner's requirement is that the VQ
# artifact not be "less smart", and the honest way to answer that is to measure
# retrieval with the same engine on both containers rather than to compare weight
# cosines. RECIPE.md section 5 already reports the codec's intrinsic cost (a
# 0.9859 mean weight cosine over the text tower); this script reports what that
# cost does to a real task.
#
# The two things being compared are the SAME code path over two weight sources:
#
#   baseline   out/EmbeddingGemma-2-914f7f8-baseline-bf16.gguf, the unquantized
#              original as llama.cpp wrote it (the file the anchor loads)
#   artifact   out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf, the
#              shipped container: 461 type-42 VQ payloads decoded through the
#              project's own dim-generic decoder
#
# Both legs run the family CPU reference tower (ds4-rs/tests/parity/eg2vq_embd_c_oracle,
# built by `make tests/parity/eg2vq_embd_c_oracle` in ds4-rs) over one ids file,
# so the tokenization is shared and cannot be a difference between the legs. The
# task is the frozen labelled set of tools/eg2_retrieval_set.sh (BEIR SciFact test
# split, 300 queries / 283 documents / 339 labelled pairs, taken whole).
#
# Flags, declared: the oracle is the tree's own -O0 entry, the same binary gate 4
# used, with the tree default thread count (min(12, cpus)); a measured aside says
# the same entry at -O2 runs 3.5x faster and moves the printed vectors by at most
# 3e-07, so a later unit that needs the wall time may take it. The legs run
# concurrently by default (two processes, 12 threads each, on a 28-core host; the
# per-leg secs are therefore contended wall time, not a latency claim): set
# EG2_PHASE5_SERIAL=1 to force them one after the other.
#
# Leg reuse: a leg is re-run unless its raw file exists AND the ids file and the
# oracle binary are the same ones recorded in its .meta file AND the raw has the
# expected number of items. That makes this gate resumable (the legs are ~78
# minutes each at -O0) without letting a stale raw file stand in for a run.
#
# Usage: tools/eg2_phase5_gate.sh <artifact.gguf> <baseline.gguf> \
#          [queries.tsv] [pool.tsv] [qrels.tsv] [prompt-map.json]
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ART=${1:?usage: eg2_phase5_gate.sh <artifact.gguf> <baseline.gguf> [queries.tsv] [pool.tsv] [qrels.tsv] [prompt-map.json]}
BASE=${2:?}
QUERIES=${3:-$ROOT/tests/retrieval/scifact-test-queries.tsv}
POOL=${4:-$ROOT/tests/retrieval/scifact-test-pool.tsv}
QRELS=${5:-$ROOT/tests/retrieval/scifact-test-qrels.tsv}
PROMPTS=${6:-$ROOT/.models/EmbeddingGemma-2-914f7f8/config_sentence_transformers.json}

ORACLE="$ROOT/ds4-rs/tests/parity/eg2vq_embd_c_oracle"
SCORE="$ROOT/tools/eg2_retrieval_score.sh"
TOJSONL="$ROOT/tools/eg2_oracle_to_jsonl.sh"
SETBUILD="$ROOT/tools/eg2_retrieval_set.sh"
BEIR_DIR=${EG2_BEIR_DIR:-$ROOT/out/beir/scifact}
ITEMS="$ROOT/out/phase5-items.tsv"
IDS="$ROOT/out/phase5-ids.tsv"
IDS_FULL="$ROOT/out/phase5-ids-full.txt"
WIDTHS=${EG2_PHASE5_WIDTHS:-"768 512 256 128"}

# The frozen set, pinned. A different set invalidates the numbers this gate
# prints, so a mismatch is a refusal, not a warning: re-pin the digests in the
# same commit that regenerates the set.
QUERIES_SHA=1247b1f4d25a1875404480961095df8266424323eb310d9d821d3231b69a404d
POOL_SHA=caf498d9db8aca3648c7e688edb970226b7bee70ad009a8f05c24c11473b73b0
QRELS_SHA=864141f92383b8b7140ebbaf61cb9dac40ae522796690063ffa1a114f973476c
EXPECT_ITEMS=583
EXPECT_TOKENS=107029

for f in "$ART" "$BASE" "$QUERIES" "$POOL" "$QRELS" "$PROMPTS"; do
    [ -s "$f" ] || { echo "eg2_phase5_gate.sh: missing $f" >&2; exit 2; }
done
[ -x "$ORACLE" ] || { echo "eg2_phase5_gate.sh: no $ORACLE — run: (cd ds4-rs && make tests/parity/eg2vq_embd_c_oracle)" >&2; exit 2; }

check_sha() {
    local file=$1 want=$2 got
    got=$(sha256sum "$file" | cut -d' ' -f1)
    [ "$got" = "$want" ] || { echo "eg2_phase5_gate.sh: $file sha256 $got != the pinned $want" >&2; exit 2; }
    echo "pinned  ${file#$ROOT/}  $got"
}
check_sha "$QUERIES" "$QUERIES_SHA"
check_sha "$POOL" "$POOL_SHA"
check_sha "$QRELS" "$QRELS_SHA"

echo
echo "== the frozen set, and its provenance =="
if [ -d "$BEIR_DIR" ]; then
    rerun=$(mktemp -d "${TMPDIR:-/tmp}/eg2-phase5-set.XXXXXX")
    "$SETBUILD" "$BEIR_DIR" "$rerun" > "$rerun/build.log"
    for f in scifact-test-queries.tsv scifact-test-pool.tsv scifact-test-qrels.tsv; do
        cmp -s "$ROOT/tests/retrieval/$f" "$rerun/$f" \
            || { echo "eg2_phase5_gate.sh: $f is not what the builder produces from $BEIR_DIR" >&2; exit 2; }
    done
    rm -rf "$rerun"
    echo "the builder reproduces all three frozen files byte for byte from ${BEIR_DIR#$ROOT/}"
else
    echo "note: ${BEIR_DIR#$ROOT/} is absent, the build-reproduces-the-set check was skipped"
fi

echo
echo "== the items and their ids =="
cat "$QUERIES" "$POOL" > "$ITEMS"
n_items_file=$(grep -vc '^#' "$ITEMS")
cargo run -q --release -p ds4-core --example gemma_vq_tok --manifest-path "$ROOT/ds4-rs/Cargo.toml" \
    -- "$(readlink -f "$ART")" "$(readlink -f "$ITEMS")" "$(readlink -f "$PROMPTS")" \
    > "$IDS_FULL" 2> "$ROOT/out/phase5-tok.log"
tail -1 "$IDS_FULL"
grep -v '^TOKENS' "$IDS_FULL" > "$IDS"
n_ids=$(wc -l < "$IDS")
[ "$n_ids" = "$EXPECT_ITEMS" ] || { echo "eg2_phase5_gate.sh: $n_ids ids for $n_items_file items, expected $EXPECT_ITEMS" >&2; exit 2; }
recorded_tokens=$(awk -F'[= ]' '/^TOKENS/ { for (i = 1; i <= NF; i++) if ($i == "tokens") print $(i + 1) }' "$IDS_FULL")
[ "$recorded_tokens" = "$EXPECT_TOKENS" ] || { echo "eg2_phase5_gate.sh: $recorded_tokens tokens, expected $EXPECT_TOKENS" >&2; exit 2; }
ids_sha=$(sha256sum "$IDS" | cut -d' ' -f1)
oracle_sha=$(sha256sum "$ORACLE" | cut -d' ' -f1)
echo "items:  ${ITEMS#$ROOT/} sha256 $(sha256sum "$ITEMS" | cut -d' ' -f1)"
echo "ids:    ${IDS#$ROOT/} sha256 $ids_sha ($n_ids items, $recorded_tokens tokens)"
echo "oracle: ${ORACLE#$ROOT/} sha256 $oracle_sha"

run_leg() {
    local leg=$1 gguf=$2
    local raw="$ROOT/out/phase5-$leg-raw.txt" log="$ROOT/out/phase5-$leg.log" meta="$ROOT/out/phase5-$leg.meta"
    if [ -s "$raw" ] && [ -s "$meta" ] \
        && [ "$(awk '$1 == "ids_sha" { print $2 }' "$meta")" = "$ids_sha" ] \
        && [ "$(awk '$1 == "oracle_sha" { print $2 }' "$meta")" = "$oracle_sha" ] \
        && [ "$(wc -l < "$raw")" = "$EXPECT_ITEMS" ]; then
        echo "-- $leg: reused ${raw#$ROOT/} (same ids and same oracle binary as its .meta)"
        return 0
    fi
    "$ORACLE" "$gguf" "$IDS" > "$raw" 2> "$log"
    {
        echo "ids_sha $ids_sha"
        echo "oracle_sha $oracle_sha"
        echo "gguf $gguf"
        echo "gguf_sha $(sha256sum "$gguf" | cut -d' ' -f1)"
    } > "$meta"
}

echo
if [ "${EG2_PHASE5_SERIAL:-0}" = 1 ]; then
    echo "== the two weight legs, one after the other =="
    run_leg baseline "$BASE"
    run_leg artifact "$ART"
else
    echo "== the two weight legs, concurrently (EG2_PHASE5_SERIAL=1 runs them one after the other) =="
    run_leg baseline "$BASE" & p1=$!
    run_leg artifact "$ART" & p2=$!
    wait $p1 || { echo "eg2_phase5_gate.sh: the baseline leg failed" >&2; exit 1; }
    wait $p2 || { echo "eg2_phase5_gate.sh: the artifact leg failed" >&2; exit 1; }
fi
for leg in baseline artifact; do
    raw="$ROOT/out/phase5-$leg-raw.txt"
    echo "-- $leg: $(cat "$ROOT/out/phase5-$leg.log")"
    n=$(wc -l < "$raw")
    [ "$n" = "$EXPECT_ITEMS" ] || { echo "eg2_phase5_gate.sh: $leg produced $n vectors, expected $EXPECT_ITEMS" >&2; exit 1; }
    echo "   ${raw#$ROOT/} sha256 $(sha256sum "$raw" | cut -d' ' -f1)"
done

echo
echo "== the vectors in the anchor shape, both legs =="
for leg in baseline artifact; do
    out="$ROOT/out/phase5-$leg.jsonl"
    "$TOJSONL" "$ROOT/out/phase5-$leg-raw.txt" > "$out"
    echo "${out#$ROOT/} sha256 $(sha256sum "$out" | cut -d' ' -f1)"
done

echo
"$SCORE" "$ROOT/out/phase5-baseline.jsonl" "$ROOT/out/phase5-artifact.jsonl" "$QRELS" $WIDTHS

echo
echo "== the same codec on gate 4's sentence pair (agreement only: that set carries no labels) =="
P4B="$ROOT/out/phase4-baseline.jsonl"
P4A="$ROOT/out/phase4-artifact.jsonl"
if [ -f "$P4B" ] && [ -f "$P4A" ]; then
    echo "-- inputs: ${P4B#$ROOT/} sha256 $(sha256sum "$P4B" | cut -d' ' -f1)"
    echo "           ${P4A#$ROOT/} sha256 $(sha256sum "$P4A" | cut -d' ' -f1)"
    "$SCORE" "$P4B" "$P4A" - $WIDTHS
else
    echo "-- SKIPPED: gate 4's jsonl pair is absent; run tools/eg2_phase4_gate.sh first"
fi

echo
echo "== what the numbers were measured on =="
echo "artifact:  ${ART#$ROOT/}"
echo "  sha256 $(sha256sum "$ART" | cut -d' ' -f1)"
echo "baseline:  ${BASE#$ROOT/}"
echo "  sha256 $(sha256sum "$BASE" | cut -d' ' -f1)"
echo "set:       ${QRELS#$ROOT/} sha256 $QRELS_SHA (300 queries / 283 documents / 339 labelled pairs, BEIR SciFact test split, taken whole)"
echo "ids:       ${IDS#$ROOT/} sha256 $ids_sha"
echo
echo "gate 5: MEASURED (the same CPU reference tower over both weight sources, one ids file,"
echo "  $n_ids items / $recorded_tokens tokens, widths $WIDTHS; the numbers above are the answer and"
echo "  the hashes are what they were measured on; no threshold is declared, by the plan of record)"
