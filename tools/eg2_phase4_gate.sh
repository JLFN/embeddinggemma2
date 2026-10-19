#!/bin/bash
# eg2_phase4_gate.sh — gate 4: the text tower's embeddings, three ways.
#
# Phase 4 of embeddinggemma2-handoff.md. Three legs, none of them this project's
# own arithmetic twice:
#
#   the Rust host   the tokenizer, whose ids must equal llama.cpp's own dump
#                   (tools/eg2_tok_check.sh); those ids are the C leg's input
#   the C oracle    ds4-rs/tests/parity/eg2vq_embd_c_oracle, the family's CPU
#                   reference tower, run over both weight sources of the same
#                   checkpoint
#   the anchor      tests/expected/anchor-embeddinggemma2.jsonl, llama.cpp at the
#                   pin through the BASELINE GGUF (phase 2, frozen)
#
# Why two weight sources, and which one carries the declared gate:
#
#   The anchor was produced by llama.cpp on the baseline BF16 GGUF, because
#   llama.cpp cannot read a type-42 payload at all (docs/artifact-gguf-gate1b.txt)
#   and the baseline is the file it can load (phase-doc X.5: "so the reference's
#   embedding example and our own C reference path load the same file"). So only
#   the baseline leg compares like with like, and the gate declared before the
#   run — C-vs-anchor MEAN COSINE >= 0.9999 at 768 — is measured there.
#   The VQ artifact leg runs the same graph on our own container; its gap to the
#   anchor is the quantization, which RECIPE.md section 5 already measures as a
#   0.9859 text-tower weight cosine, and which phase 5 owns. Reporting it as a
#   failure of the port would blame the tower for the codec.
#
# Usage: tools/eg2_phase4_gate.sh <artifact.gguf> <baseline.gguf> <items.tsv> <prompt-map.json>
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ART=${1:?usage: eg2_phase4_gate.sh <artifact.gguf> <baseline.gguf> <items.tsv> <prompt-map.json>}
BASE=${2:?}
ITEMS=${3:?}
PROMPTS=${4:?}
ANCHOR="$ROOT/tests/expected/anchor-embeddinggemma2.jsonl"
ORACLE="$ROOT/ds4-rs/tests/parity/eg2vq_embd_c_oracle"
IDS="$ROOT/out/phase4-ids.tsv"
IDS_FULL="$ROOT/out/phase4-ids-full.txt"

for f in "$ART" "$BASE" "$ITEMS" "$PROMPTS" "$ANCHOR" "$ORACLE"; do
    [ -e "$f" ] || { echo "eg2_phase4_gate.sh: missing $f" >&2; exit 2; }
done
# The cargo test runs with the package root as its cwd (ds4-rs/crates/ds4-core),
# so every path handed to it is absolute.
ART=$(readlink -f "$ART")
BASE=$(readlink -f "$BASE")
ITEMS=$(readlink -f "$ITEMS")
PROMPTS=$(readlink -f "$PROMPTS")
ANCHOR=$(readlink -f "$ANCHOR")
ORACLE=$(readlink -f "$ORACLE")

# One row per item, in the anchor's own shape: the 768 vector as the engine
# produced it, and 512/256/128 as its leading dimensions re-normalized after the
# cut — the same rule as tools/eg2_anchor.sh, whose frozen jsonl is one side of
# the comparison.
to_jsonl() {
    awk -F'\t' '
    function arr(   i, s) { s = "["; for (i = 1; i <= 768; i++) s = s (i > 1 ? "," : "") sprintf("%.7f", v[i]);
                            return s "]" }
    function trunc(k,   i, s, n2, t) { n2 = 0; for (i = 1; i <= k; i++) n2 += v[i] * v[i];
                            t = sqrt(n2); if (t == 0) t = 1;
                            s = "["; for (i = 1; i <= k; i++) s = s (i > 1 ? "," : "") sprintf("%.7f", v[i] / t);
                            return s "]" }
    {
        n = split($3, v, " ");
        if (n != 768) { print "gate: line " NR " has " n " values" > "/dev/stderr"; exit 1 }
        printf "{\"kind\":\"item\",\"id\":\"%s\",\"ntok\":%s,\"dims\":{\"768\":%s,\"512\":%s,\"256\":%s,\"128\":%s}}\n", \
            $1, $2, arr(), trunc(512), trunc(256), trunc(128)
    }' "$1"
}

echo "== leg 1: the tokenizer (Rust host) against llama.cpp's own ids"
tools/eg2_tok_check.sh "$ART" "$ITEMS" "$PROMPTS" "${EG2_TOK_REF_LOG:-}"

echo
echo "== the host's ids for both legs (the vocabulary is the same in both files)"
cargo run -q --release -p ds4-core --example gemma_vq_tok --manifest-path "$ROOT/ds4-rs/Cargo.toml" \
    -- "$ART" "$ITEMS" "$PROMPTS" > "$IDS_FULL"
tail -1 "$IDS_FULL"
# The oracle reads the four-field item lines only, and refuses anything else;
# the driver's own trailing summary is for the record, not for it.
grep -v '^TOKENS' "$IDS_FULL" > "$IDS"
n_ids=$(wc -l < "$IDS")

for leg in baseline artifact; do
    gguf=$BASE; floor=0.9999
    [ "$leg" = artifact ] && { gguf=$ART; floor=""; }
    raw="$ROOT/out/phase4-$leg-raw.txt"; log="$ROOT/out/phase4-$leg.log"
    eng="$ROOT/out/phase4-$leg.jsonl"

    echo
    echo "== leg 2 ($leg): the C reference tower on $leg weights"
    "$ORACLE" "$gguf" "$IDS" > "$raw" 2> "$log"
    cat "$log"
    n_vec=$(wc -l < "$raw")
    if [ "$n_vec" != "$n_ids" ]; then
        echo "eg2_phase4_gate.sh: $n_ids items but $n_vec vectors" >&2
        exit 1
    fi
    to_jsonl "$raw" > "$eng"

    echo "-- the Rust host's route over the same set (cargo test)"
    DS4_EG2VQ_C_ORACLE="$ORACLE" \
    DS4_EG2VQ_IDENTITY="$ART" \
    DS4_EG2VQ_WEIGHTS="$gguf" \
    DS4_EG2VQ_ITEMS="$ITEMS" \
    DS4_EG2VQ_PROMPTS="$PROMPTS" \
    DS4_EG2VQ_ANCHOR="$ANCHOR" \
    DS4_EG2VQ_MIN_COS="$floor" \
    cargo test --quiet --manifest-path "$ROOT/ds4-rs/Cargo.toml" -p ds4-core --test eg2vq_ref -- --nocapture

    echo "-- the port against the frozen anchor"
    tools/eg2_anchor_cmp.sh "$eng" "$ANCHOR"

    if [ "$leg" = baseline ]; then
        line=$(tools/eg2_anchor_cmp.sh "$eng" "$ANCHOR" | awk '$1 == 768 { print; exit }')
        meancos=$(echo "$line" | awk '{ print $4 }')
        if awk -v c="$meancos" 'BEGIN { exit (c >= 0.9999) ? 0 : 1 }'; then
            echo "gate 4: PASSED (baseline C reference vs anchor, 768 mean cosine $meancos >= 0.9999)"
        else
            echo "gate 4: FAILED (baseline C reference vs anchor, 768 mean cosine $meancos < 0.9999)"
            exit 1
        fi
    fi
    echo "$leg jsonl: ${eng#"$ROOT"/} sha256 $(sha256sum "$eng" | cut -d' ' -f1)"
done

echo
echo "artifact:  $ART"
echo "  sha256 $(sha256sum "$ART" | cut -d' ' -f1)"
echo "baseline:  $BASE"
echo "  sha256 $(sha256sum "$BASE" | cut -d' ' -f1)"
echo "anchor:    ${ANCHOR#"$ROOT"/}"
echo "  sha256 $(sha256sum "$ANCHOR" | cut -d' ' -f1)"
echo "ids:       ${IDS#"$ROOT"/} sha256 $(sha256sum "$IDS" | cut -d' ' -f1)"
