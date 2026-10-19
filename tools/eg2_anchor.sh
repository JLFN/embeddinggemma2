#!/bin/bash
# eg2_anchor.sh — produce the frozen EmbeddingGemma-2 anchor vectors (phase 2).
#
# The anchor is the llama.cpp reference's own output over a frozen input set,
# with the baseline GGUF from phase 1a. It is a second opinion beside this
# project's own C reference path, never it (handoff section 3, phase 2).
#
# What the model is given, per item:  prompts[prompt_name] + text
#   - prompts is the checkpoint's config_sentence_transformers.json "prompts"
#     map, taken verbatim (the table below is the whole map this project uses,
#     copied from that file; an unknown name is a hard error, never a silent
#     empty prefix);
#   - the checkpoint's Pooling module has include_prompt true
#     (1_Pooling/config.json), so the prefix is part of the text that is pooled;
#   - the prefix-plus-text form is what the model card documents ("task
#     instruction prefixes prepended to text inputs", README.md:128-151).
#
# Output: one JSON object per line.
#   line 1  {"kind":"provenance", ...}     revision, command line, digests
#   then    {"kind":"item","id":...,"prompt_name":...,"prompt":...,"input":...,
#            "ntok":N,"input_sha256":...,"dims":{"768":[...],"512":[...],
#            "256":[...],"128":[...]}}
# The 768 vector is the reference's own normalized output (--embd-normalize 2,
# euclidean, which is the 2_Normalize module of the checkpoint). The three
# shorter vectors are that vector truncated to its leading dimensions and
# L2-renormalized, which is Matryoshka truncation as the model card requires
# ("Re-normalize after truncating", README.md:166-185). Skipping the
# renormalization would produce plausible-looking but wrong cosines.
#
# One item per model invocation by default. Reason, measured: the reference's
# own output for a given input depends on the batch it sits in (a 113-item batch
# versus the same item alone differs by 3.4e-4 per element, cosine 0.9999962),
# because a different ubatch shape changes the reduction order of every matrix
# multiply. Encoding one item at a time makes each vector a function of its own
# input only, so two identical inputs are bit-identical, and a later edit to the
# item file cannot move the other vectors. --batched does the whole file in one
# invocation (13 s); it is bit-reproducible on this machine, but every vector
# then depends on the set.
#
# Usage:
#   tools/eg2_anchor.sh                       # the anchor (one item per run)
#   tools/eg2_anchor.sh --batched             # the same, in a single invocation
#   tools/eg2_anchor.sh --bare                # same items, no prompt prefix
#   tools/eg2_anchor.sh --eos-literal         # no trailing <eos> in the file,
#                                             # the string "<eos>" appended
#                                             # instead (a cross-check leg)
#   tools/eg2_anchor.sh --gguf F --out F --items F --bin F --tag T
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ITEMS="$ROOT/tests/anchor/anchor-items.tsv"
GGUF="$ROOT/out/EmbeddingGemma-2-914f7f8-baseline-bf16.gguf"
OUT="$ROOT/tests/expected/anchor-embeddinggemma2.jsonl"
BIN=${EG2_ANCHOR_BIN:-/data/llama-anchor-build/bin/llama-embedding}
BARE=0
EOS_LITERAL=0
PER_ITEM=1
TAG="anchor"
while [ $# -gt 0 ]; do
    case "$1" in
        --items) ITEMS=$2; shift 2 ;;
        --gguf)  GGUF=$2; shift 2 ;;
        --out)   OUT=$2; shift 2 ;;
        --bin)   BIN=$2; shift 2 ;;
        --tag)   TAG=$2; shift 2 ;;
        --bare)  BARE=1; shift ;;
        --per-item) PER_ITEM=1; shift ;;
        --batched)  PER_ITEM=0; shift ;;
        --eos-literal) EOS_LITERAL=1; shift ;;
        -h|--help) sed -n '2,45p' "$0"; exit 0 ;;
        *) echo "eg2_anchor.sh: unknown argument '$1'" >&2; exit 2 ;;
    esac
done
for f in "$ITEMS" "$GGUF" "$BIN"; do
    [ -e "$f" ] || { echo "eg2_anchor.sh: missing $f" >&2; exit 2; }
done
mkdir -p "$(dirname "$OUT")" "$ROOT/out"

# ---- the checkpoint's own prompt table (config_sentence_transformers.json) ----
# Verbatim, including the trailing space of every entry.
prompt_for() {
    case "$1" in
        Raw)                  printf '%s' '' ;;
        BitextMining)         printf '%s' 'task: search result | query: ' ;;
        Classification)       printf '%s' 'task: classification | query: ' ;;
        Clustering)           printf '%s' 'task: clustering | query: ' ;;
        CodeRetrieval)        printf '%s' 'task: code retrieval | query: ' ;;
        Document)             printf '%s' 'title: none | text: ' ;;
        FactChecking)         printf '%s' 'task: fact checking | query: ' ;;
        InstructionRetrieval) printf '%s' 'task: code retrieval | query: ' ;;
        MultilabelClassification) printf '%s' 'task: classification | query: ' ;;
        PairClassification)   printf '%s' 'task: sentence similarity | query: ' ;;
        QuestionAnswering)    printf '%s' 'task: question answering | query: ' ;;
        Reranking)            printf '%s' 'task: search result | query: ' ;;
        Retrieval)            printf '%s' 'task: search result | query: ' ;;
        Retrieval-document)   printf '%s' 'title: none | text: ' ;;
        Retrieval-query)      printf '%s' 'task: search result | query: ' ;;
        STS)                  printf '%s' 'task: sentence similarity | query: ' ;;
        SearchQuery)          printf '%s' 'task: search result | query: ' ;;
        SentenceSimilarity)   printf '%s' 'task: sentence similarity | query: ' ;;
        Summarization)        printf '%s' 'task: sentence similarity | query: ' ;;
        document)             printf '%s' 'title: none | text: ' ;;
        query)                printf '%s' 'task: search result | query: ' ;;
        *) return 1 ;;
    esac
}

WORK=$(mktemp -d "${TMPDIR:-/tmp}/eg2-anchor.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
PROMPTS="$WORK/prompts.txt"
SIDECAR="$WORK/sidecar.tsv"
mkdir -p "$WORK/in"
: > "$PROMPTS"; : > "$SIDECAR"

n_items=0
while IFS=$'\t' read -r id pname text; do
    case "$id" in ''|'#'*) continue ;; esac
    [ -n "$pname" ] || { echo "eg2_anchor.sh: $id has no prompt_name" >&2; exit 1; }
    if ! pre=$(prompt_for "$pname"); then
        echo "eg2_anchor.sh: prompt_name '$pname' is not in the checkpoint's map" >&2
        exit 1
    fi
    [ "$BARE" = 1 ] && pre=''
    input="$pre$text"
    [ "$EOS_LITERAL" = 1 ] && input="$input<eos>"
    if [ "$n_items" -gt 0 ]; then printf '\n' >> "$PROMPTS"; fi
    printf '%s' "$input" >> "$PROMPTS"
    printf '%s' "$input" > "$WORK/in/$(printf '%04d' "$n_items").txt"
    esc() { printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'; }
    printf '%s\t%s\t%s\t%s\t%s\n' "$id" "$pname" "$(esc "$pre")" "$(esc "$input")" \
        "$(printf '%s' "$input" | sha256sum | cut -d' ' -f1)" >> "$SIDECAR"
    n_items=$((n_items + 1))
done < "$ITEMS"
[ "$n_items" -gt 0 ] || { echo "eg2_anchor.sh: no items read from $ITEMS" >&2; exit 1; }

BINDIR=$(cd "$(dirname "$BIN")" && pwd)
export LD_LIBRARY_PATH="$BINDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
VER=$("$BIN" --version 2>&1 | tr '\n' ' ' | sed 's/  */ /g' | sed 's/ *$//')

# ---- run 1: the vectors -------------------------------------------------------
# --embd-output-format raw prints one whitespace-separated embedding per line;
# --embd-normalize 2 is the checkpoint's own Normalize module (2_Normalize).
# A vector line is 768 whitespace-separated %1.7f fields and nothing else, which
# is what the parser keys on; no other line of the log has that shape.
RAW="$ROOT/out/$TAG-raw.log"
: > "$RAW"
: > "$WORK/vectors.txt"
if [ "$PER_ITEM" = 1 ]; then
    for f in "$WORK"/in/*.txt; do
        "$BIN" -m "$GGUF" -f "$f" --pooling mean --embd-normalize 2 \
               --embd-output-format raw --no-warmup > "$WORK/one.log" 2>&1
        awk 'NF==768' "$WORK/one.log" >> "$WORK/vectors.txt"
        printf '== %s\n' "$(basename "$f")" >> "$RAW"
        cat "$WORK/one.log" >> "$RAW"
    done
else
    "$BIN" -m "$GGUF" -f "$PROMPTS" --pooling mean --embd-normalize 2 \
           --embd-output-format raw --no-warmup > "$RAW" 2>&1
    awk 'NF==768' "$RAW" > "$WORK/vectors.txt"
fi
nvec=$(wc -l < "$WORK/vectors.txt")
if [ "$nvec" != "$n_items" ]; then
    echo "eg2_anchor.sh: $nvec embedding lines for $n_items items (see $RAW)" >&2
    exit 1
fi

# ---- run 2: the reference's own token count per item -------------------------
# --verbose-prompt prints "prompt <i>" then "number of tokens in prompt = N".
VERB="$ROOT/out/$TAG-verbose.log"
"$BIN" -m "$GGUF" -f "$PROMPTS" --pooling mean --no-warmup --verbose-prompt \
       > "$VERB" 2>&1
awk '/number of tokens in prompt = /{print $NF}' "$VERB" > "$WORK/ntok.txt"
ntok_n=$(wc -l < "$WORK/ntok.txt")
if [ "$ntok_n" != "$n_items" ]; then
    echo "eg2_anchor.sh: $ntok_n token counts for $n_items items (see $VERB)" >&2
    exit 1
fi

# ---- assemble ----------------------------------------------------------------
paste -d$'\t' "$SIDECAR" "$WORK/ntok.txt" > "$WORK/meta.tsv"
REV=$("$BIN" --version 2>&1 | grep -o '[0-9a-f]\{7,\}' | head -1 || true)
GGUF_SHA=$(sha256sum "$GGUF" | cut -d' ' -f1)
ITEMS_SHA=$(sha256sum "$ITEMS" | cut -d' ' -f1)
GGUF_REL=${GGUF#"$ROOT"/}
ITEMS_REL=${ITEMS#"$ROOT"/}
OUT_REL=${OUT#"$ROOT"/}
{
printf '{"kind":"provenance","producer":"%s","llama_cpp_revision":"%s","llama_embedding_version":"%s","model":"%s","model_sha256":"%s","items":"%s","items_sha256":"%s","items_count":%d,"vector_mode":"%s","prompt_mode":"%s","eos":"%s","normalize":"l2","mrl_widths":[768,512,256,128],"mrl_renormalized_after_truncation":true,"command":"%s -m %s -f <one input per invocation> --pooling mean --embd-normalize 2 --embd-output-format raw --no-warmup","note":"reference output over the frozen set; a second opinion beside this project own C reference path, never it"}\n' \
    "tools/eg2_anchor.sh" "$REV" "$VER" "$GGUF_REL" "$GGUF_SHA" "$ITEMS_REL" "$ITEMS_SHA" \
    "$n_items" "$([ "$PER_ITEM" = 1 ] && echo per-item || echo batched)" \
    "$([ "$BARE" = 1 ] && echo bare || echo prefixed)" \
    "$([ "$EOS_LITERAL" = 1 ] && echo literal-in-text || echo tokenizer-flag)" "$BIN" "$GGUF_REL"
awk -F'\t' '
function arr(   i, s) { s = "["; for (i = 1; i <= 768; i++) s = s (i > 1 ? "," : "") sprintf("%.7f", v[i]);
                        return s "]" }
function trunc(k,   i, s, n2, t) { n2 = 0; for (i = 1; i <= k; i++) n2 += v[i] * v[i];
                        t = sqrt(n2); if (t == 0) t = 1;
                        s = "["; for (i = 1; i <= k; i++) s = s (i > 1 ? "," : "") sprintf("%.7f", v[i] / t);
                        return s "]" }
NR == FNR { id[FNR] = $1; pn[FNR] = $2; pv[FNR] = $3; iv[FNR] = $4; hs[FNR] = $5; tk[FNR] = $6; next }
{
    n = split($0, v, " ");
    if (n != 768) { print "eg2_anchor.sh: vector line " FNR " has " n " values" > "/dev/stderr"; exit 1 }
    printf "{\"kind\":\"item\",\"id\":\"%s\",\"prompt_name\":\"%s\",\"prompt\":\"%s\",\"input\":\"%s\",\"ntok\":%s,\"input_sha256\":\"%s\",\"dims\":{\"768\":%s,\"512\":%s,\"256\":%s,\"128\":%s}}\n", \
        id[FNR], pn[FNR], pv[FNR], iv[FNR], tk[FNR], hs[FNR], arr(), trunc(512), trunc(256), trunc(128)
}' "$WORK/meta.tsv" "$WORK/vectors.txt"
} > "$OUT"

echo "eg2_anchor.sh: $n_items items -> $OUT_REL"
echo "eg2_anchor.sh: $GGUF_REL sha256 $GGUF_SHA"
echo "eg2_anchor.sh: jsonl sha256 $(sha256sum "$OUT" | cut -d' ' -f1)"
