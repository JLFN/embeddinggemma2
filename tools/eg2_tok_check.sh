#!/bin/bash
# eg2_tok_check.sh — gate: the port's gemma4 tokenizer equals llama.cpp's.
#
# Phase 4 of embeddinggemma2-handoff.md, first leg. The port must reproduce the
# reference's own token ids per item before any arithmetic is written, so this
# compares two dumps of the same frozen inputs:
#
#   reference  llama.cpp at the pin, --verbose-prompt over the whole item file,
#              regenerated here under its own tag (out/tok-anchor-*.log); its
#              vectors land in a derived jsonl and the frozen phase-2 evidence
#              is not touched.
#   port       the Rust host's own path: ds4-core's `gemma_vq_tok` example over
#              the artifact (so the vocabulary comes from the artifact's own
#              tokenizer.ggml.* tables) and the checkpoint's prompt map.
#
# Both sides must give the same token count AND the same id sequence, per item.
#
# Authority (all /data/llama.cpp): the `gemma4` pre-tokenizer and its BPE,
# llama-vocab.cpp:528-536 (:2360-2362 selects it, :3652-3654 escapes spaces),
# the special-token cache at :3026-3054 + :3168-3191 partitioned at
# :3401-3500, and the merge queue at :264-278 + :589-745.
#
# Usage:
#   tools/eg2_tok_check.sh <artifact.gguf> <items.tsv> <prompt-map.json>
#                          [reference-verbose.log]
#   The fourth argument reuses an existing reference log instead of running
#   llama.cpp again.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
GGUF=${1:?usage: eg2_tok_check.sh <artifact.gguf> <items.tsv> <prompt-map.json> [verbose.log]}
ITEMS=${2:?}
PROMPTS=${3:?}
LOG=${4:-}

for f in "$GGUF" "$ITEMS" "$PROMPTS"; do
    [ -e "$f" ] || { echo "eg2_tok_check.sh: missing $f" >&2; exit 2; }
done

WORK=$(mktemp -d "${TMPDIR:-/tmp}/eg2-tok.XXXXXX")
trap 'rm -rf "$WORK"' EXIT

if [ -z "$LOG" ]; then
    echo "== regenerating the reference token dump (llama.cpp at the pin)"
    tools/eg2_anchor.sh --batched --tag tok-anchor \
        --gguf "$ROOT/out/EmbeddingGemma-2-914f7f8-baseline-bf16.gguf" \
        --items "$ITEMS" --out "$WORK/derived.jsonl" >/dev/null
    LOG="$ROOT/out/tok-anchor-verbose.log"
fi
[ -s "$LOG" ] || { echo "eg2_tok_check.sh: no reference log at $LOG" >&2; exit 2; }

echo "== the port (Rust host, vocabulary from the artifact)"
cargo run -q --release -p ds4-core --example gemma_vq_tok --manifest-path "$ROOT/ds4-rs/Cargo.toml" \
    -- "$GGUF" "$ITEMS" "$PROMPTS" > "$WORK/port.tsv"

# Reference side: "main: prompt N: '...'" opens an item, "number of tokens in
# prompt = M" states the count, and each following "  <id> -> '<piece>'" is one
# token. Pieces are not compared (a piece can hold an escaped newline); the ids
# are the tokenizer's whole contract.
awk '
    /main: prompt [0-9]+: /   { n++; have[n] = 0; next }
    /number of tokens in prompt = / { want[n] = $NF; next }
    /^[[:space:]]*[0-9]+ -> /  { if (est[n]) { est[n] = est[n] "," $1 } else { est[n] = $1 }
                                 have[n]++; next }
    END {
        for (i = 1; i <= n; i++) printf "%d\t%d\t%s\n", i, want[i], est[i]
    }' "$LOG" > "$WORK/ref.tsv"

# Neither side may be empty, and the counts must agree per item before the ids.
ref_n=$(wc -l < "$WORK/ref.tsv")
port_n=$(grep -vc '^TOKENS' "$WORK/port.tsv" || true)
if [ "$ref_n" != "$port_n" ]; then
    echo "eg2_tok_check.sh: reference has $ref_n items, the port has $port_n" >&2
    exit 1
fi

# port.tsv is <item> <prompt_name> <ntok> <ids>; join by position.
paste -d'\t' "$WORK/ref.tsv" <(grep -v '^TOKENS' "$WORK/port.tsv") |
awk -F'\t' '
    {
        idx = $1; want = $2; ref = $3; item = $4; pname = $5; ntok = $6; ids = $7
        if (want != ntok) { bad_count++; printf "MISMATCH count %s (%s): reference %s, port %s\n", item, pname, want, ntok }
        if (ref != ids)   { bad_ids++;   printf "MISMATCH ids %s (%s): reference %s tokens, port %s\n", item, pname, want, ntok }
        n++
    }
    END {
        printf "tokenizer: %d items, %d count mismatches, %d id mismatches\n", n, bad_count, bad_ids
        exit (bad_count || bad_ids) ? 1 : 0
    }'

tail -1 "$WORK/port.tsv"
echo "reference log: ${LOG#"$ROOT"/}"
