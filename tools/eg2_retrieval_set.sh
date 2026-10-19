#!/bin/bash
# eg2_retrieval_set.sh — build the phase-5 frozen labelled retrieval set.
#
# Phase 5 of embeddinggemma2-handoff.md answers the owner's "not less smart" with
# a retrieval metric, and a retrieval metric needs labels this project did not
# write itself. The labels are BEIR's SciFact test split, taken WHOLE and
# unmodified: every query the split labels (300) and every document it marks
# relevant (283), 339 labelled pairs. There is no sampling rule and no seed, so
# there is nothing here for a later unit to re-roll.
#
# What is frozen is the selection and the input formatting, not the text:
#
#   source     https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip
#              (BEIR's own distribution; the paper is Thakur et al., NeurIPS 2021
#              Datasets and Benchmarks). The zip's sha256 is pinned below and the
#              builder refuses another one.
#   selection  qrels/test.tsv rows with score >= 1. Every row in this split scores
#              1, so the whole file is the label set: 339 pairs over 300 queries
#              (277 queries have one relevant document, 14 have two, 4 have three,
#              3 have four, 2 have five) and 283 distinct documents.
#   query side `prompt_name = SearchQuery`, the checkpoint's retrieval-query prompt
#              (`task: search result | query: ` — config_sentence_transformers.json
#              keys Retrieval/Retrieval-query/Reranking/SearchQuery are this same
#              string).
#   doc side   `prompt_name = Raw` (no prefix) with the text carrying the README's
#              own titled-document formatting, `title: {title} | text: {content}`
#              (checkpoint README, the prompt table and the note under it: "titled
#              documents must still be formatted manually"). The config's `Document`
#              key is the untitled form (`title: none | text: `); every SciFact
#              document has a title, so the title slot is filled.
#
# Item ids are the dataset's own numeric ids, unprefixed: query ids run 1..1395
# and document ids 33370..195689316, disjoint by range, which the builder asserts
# rather than assumes.
#
# Usage: tools/eg2_retrieval_set.sh <beir-scifact-dir> <out-dir>
#   e.g. tools/eg2_retrieval_set.sh out/beir/scifact tests/retrieval
set -euo pipefail

SRC=${1:?usage: eg2_retrieval_set.sh <beir-scifact-dir> <out-dir>}
OUT=${2:?usage: eg2_retrieval_set.sh <beir-scifact-dir> <out-dir>}
ZIP_SHA=536e14446a0ba56ed1398ab1055f39fe852686ecad24a6306c80c490fa8e0165
ZIP_URL=https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip

command -v jq >/dev/null || { echo "eg2_retrieval_set.sh: jq is required" >&2; exit 2; }
for f in "$SRC/corpus.jsonl" "$SRC/queries.jsonl" "$SRC/qrels/test.tsv"; do
    [ -s "$f" ] || { echo "eg2_retrieval_set.sh: missing $f" >&2; exit 2; }
done
# The zip is only checked when it is still around; the extracted files are what
# this script reads, and the frozen outputs are hashed by the gate either way.
if [ -f "$SRC/../scifact.zip" ]; then
    got=$(sha256sum "$SRC/../scifact.zip" | cut -d' ' -f1)
    [ "$got" = "$ZIP_SHA" ] || { echo "eg2_retrieval_set.sh: scifact.zip sha256 $got != $ZIP_SHA" >&2; exit 2; }
fi

mkdir -p "$OUT"
tmp=$(mktemp -d "${TMPDIR:-/tmp}/eg2-retrieval-set.XXXXXX"); trap 'rm -rf "$tmp"' EXIT

# The split's own qrels, in numeric order, header dropped. jq joins on a literal
# TAB rather than @tsv so the text is transported verbatim: the builder asserts
# below that no title or text carries a TAB or newline, which is what makes the
# tab-separated items format lossless for this corpus (checked, not hoped: 0 of
# 5183 documents and 0 of 1109 queries contain one).
tail -n +2 "$SRC/qrels/test.tsv" | awk -F'\t' '$3 >= 1 { print $1 "\t" $2 "\t" $3 }' \
    | sort -t$'\t' -k1,1n -k2,2n > "$tmp/qrels.tsv"
[ -s "$tmp/qrels.tsv" ] || { echo "eg2_retrieval_set.sh: no qrels rows with score >= 1" >&2; exit 2; }

jq -r '[._id, .title, .text] | join("\t")' "$SRC/corpus.jsonl" > "$tmp/corpus.tsv"
jq -r '[._id, .text] | join("\t")' "$SRC/queries.jsonl" > "$tmp/queries.tsv"
awk -F'\t' 'NF != 3 { print "eg2_retrieval_set.sh: corpus.jsonl line " NR " has " NF " fields" > "/dev/stderr"; exit 1 }' "$tmp/corpus.tsv"
awk -F'\t' 'NF != 2 { print "eg2_retrieval_set.sh: queries.jsonl line " NR " has " NF " fields" > "/dev/stderr"; exit 1 }' "$tmp/queries.tsv"

# Query ids and document ids must be disjoint, or one item id in the ids file
# would name two things. Ranges, measured rather than assumed.
awk -F'\t' '{ if ($1 < qmin || qn == 0) qmin = $1; if ($1 > qmax) qmax = $1; qn++ } END { print qmin, qmax, qn }' \
    "$tmp/queries.tsv" > "$tmp/qrange"
awk -F'\t' '{ if ($1 < dmin || dn == 0) dmin = $1; if ($1 > dmax) dmax = $1; dn++ } END { print dmin, dmax, dn }' \
    "$tmp/corpus.tsv" > "$tmp/drange"
read -r qmin qmax _ < "$tmp/qrange"
read -r dmin dmax _ < "$tmp/drange"
if [ "$qmax" -ge "$dmin" ]; then
    echo "eg2_retrieval_set.sh: query id range 1..$qmax overlaps document id range $dmin..$dmax" >&2
    exit 2
fi

n_pairs=$(wc -l < "$tmp/qrels.tsv")
n_q=$(cut -f1 "$tmp/qrels.tsv" | sort -u | wc -l)
n_d=$(cut -f2 "$tmp/qrels.tsv" | sort -u | wc -l)
# ---- the three frozen files, each with its provenance in a comment header ----

{
    printf '# Frozen labelled retrieval set for phase 5 (the VQ codec'"'"'s cost), query side.\n'
    printf '# Format: TAB-separated  id <TAB> prompt_name <TAB> text\n'
    printf '# Source: %s (sha256 %s), the test split, taken whole.\n' "$ZIP_URL" "$ZIP_SHA"
    printf '# Selection: every query id in qrels/test.tsv with score >= 1 (n=%s). Ids are the\n' "$n_q"
    printf '# dataset'"'"'s own, unprefixed. prompt_name SearchQuery is the checkpoint'"'"'s retrieval-query\n'
    printf '# prompt, "task: search result | query: " (config_sentence_transformers.json).\n'
    printf '# Built by tools/eg2_retrieval_set.sh (in git; no sampling rule, no seed, no\n'
    printf '# timestamp: a re-run of the builder is byte-identical).\n'
    awk -F'\t' 'NR == FNR { keep[$1] = 1; next }
                { if (!($1 in keep)) next; print $1 "\tSearchQuery\t" $2 }' \
        <(cut -f1 "$tmp/qrels.tsv" | sort -u) "$tmp/queries.tsv" | sort -n
} > "$OUT/scifact-test-queries.tsv"

{
    printf '# Frozen labelled retrieval set for phase 5 (the VQ codec'"'"'s cost), document side.\n'
    printf '# Format: TAB-separated  id <TAB> prompt_name <TAB> text\n'
    printf '# Source: %s (sha256 %s), the test split, taken whole.\n' "$ZIP_URL" "$ZIP_SHA"
    printf '# Selection: every corpus id qrels/test.tsv marks relevant, score >= 1 (n=%s). This is a\n' "$n_d"
    printf '# HARD pool by construction: every document in it is the gold evidence for some other\n'
    printf '# claim in the same split, so no distractor is filler text.\n'
    printf '# prompt_name Raw (no prefix) with the README'"'"'s titled-document formatting in the text\n'
    printf '# itself, "title: {title} | text: {content}" (checkpoint README, the prompt table and the\n'
    printf '# note beneath it). Raw is the untitled-prefix case, so the whole input is that string.\n'
    printf '# Built by tools/eg2_retrieval_set.sh (in git; a re-run is byte-identical).\n'
    awk -F'\t' 'NR == FNR { keep[$1] = 1; next }
                { if (!($1 in keep)) next; print $1 "\tRaw\ttitle: " $2 " | text: " $3 }' \
        <(cut -f2 "$tmp/qrels.tsv" | sort -u) "$tmp/corpus.tsv" | sort -n
} > "$OUT/scifact-test-pool.tsv"

{
    printf '# Frozen labelled retrieval set for phase 5 (the VQ codec'"'"'s cost), labels.\n'
    printf '# Format: TAB-separated  query_id <TAB> corpus_id <TAB> score, ids as the dataset ships them.\n'
    printf '# Source: %s (sha256 %s), qrels/test.tsv rows with score >= 1, unmodified.\n' "$ZIP_URL" "$ZIP_SHA"
    printf '# %s pairs over %s queries and %s documents.\n' "$n_pairs" "$n_q" "$n_d"
    cat "$tmp/qrels.tsv"
} > "$OUT/scifact-test-qrels.tsv"

# The two items files must not have lost or mangled a line on the way out.
for f in "$OUT/scifact-test-queries.tsv" "$OUT/scifact-test-pool.tsv"; do
    got=$(grep -vc '^#' "$f")
    want=$([ "$f" = "$OUT/scifact-test-queries.tsv" ] && echo "$n_q" || echo "$n_d")
    [ "$got" = "$want" ] || { echo "eg2_retrieval_set.sh: $f has $got items, expected $want" >&2; exit 2; }
done
[ "$(grep -vc '^#' "$OUT/scifact-test-qrels.tsv")" = "$n_pairs" ] \
    || { echo "eg2_retrieval_set.sh: qrels file lost rows" >&2; exit 2; }

# The summary prints the SELECTED ids' ranges (the disjointness above was argued
# on the source files' full ranges, which is the stronger statement).
sqmin=$(cut -f1 "$tmp/qrels.tsv" | sort -n | head -1)
sqmax=$(cut -f1 "$tmp/qrels.tsv" | sort -n | tail -1)
sdmin=$(cut -f2 "$tmp/qrels.tsv" | sort -n | head -1)
sdmax=$(cut -f2 "$tmp/qrels.tsv" | sort -n | tail -1)
echo "eg2_retrieval_set.sh: queries $n_q (selected ids $sqmin..$sqmax; source file $qmin..$qmax)," \
     "documents $n_d (selected ids $sdmin..$sdmax; source file $dmin..$dmax), pairs $n_pairs"
for f in "$OUT/scifact-test-queries.tsv" "$OUT/scifact-test-pool.tsv" "$OUT/scifact-test-qrels.tsv"; do
    echo "  ${f}  $(sha256sum "$f" | cut -d' ' -f1)"
done
