#!/bin/bash
# eg2_retrieval_score_selftest.sh — check tools/eg2_retrieval_score.sh against
# cases whose answers are worked out by hand, so that the phase-5 numbers rest on
# a scorer that has been shown to count correctly.
#
# Two fixtures, each small enough to verify on paper:
#
#   case A (string ids)   two queries, three documents, four-dimensional vectors
#                         in exact arithmetic. It pins the two-sided mean, the
#                         percentiles' inputs, the tail counts, the tie break
#                         (documents d2 and d3 are identical for query q2, so the
#                         smaller id must take the top spot), recall@k with one
#                         relevant document per query, MRR@10, and the top-1
#                         change count of a deliberate rotation.
#   case B (numeric ids)  one query, two identical documents with ids 9 and 10.
#                         The documented rule is "ties broken by document id
#                         ascending", and by VALUE 9 precedes 10, so the relevant
#                         document 10 must land at rank 2 (MRR 0.5). Byte order
#                         would put "10" first and report MRR 1.0, so this case is
#                         what distinguishes the two rules.
#
# Usage: tools/eg2_retrieval_score_selftest.sh
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SCORE="$ROOT/tools/eg2_retrieval_score.sh"
tmp=$(mktemp -d "${TMPDIR:-/tmp}/eg2-score-selftest.XXXXXX"); trap 'rm -rf "$tmp"' EXIT
fails=0
oks=0

check() {  # check <label> <expected-substring> <file>
    if grep -qF -- "$2" "$3"; then
        printf 'ok   %s\n' "$1"
        oks=$((oks + 1))
    else
        printf 'FAIL %s: no line containing %s in the scorer output\n' "$1" "$2"
        fails=$((fails + 1))
    fi
}

# ---- case A: string ids, exact arithmetic, a deliberate rotation ------------
cat > "$tmp/a-base.jsonl" <<'EOF'
{"kind":"item","id":"q1","ntok":3,"dims":{"4":[1.0000000,0.0000000,0.0000000,0.0000000]}}
{"kind":"item","id":"q2","ntok":3,"dims":{"4":[0.0000000,1.0000000,0.0000000,0.0000000]}}
{"kind":"item","id":"d1","ntok":5,"dims":{"4":[1.0000000,0.0000000,0.0000000,0.0000000]}}
{"kind":"item","id":"d2","ntok":5,"dims":{"4":[0.0000000,1.0000000,0.0000000,0.0000000]}}
{"kind":"item","id":"d3","ntok":5,"dims":{"4":[0.0000000,1.0000000,0.0000000,0.0000000]}}
EOF
cat > "$tmp/a-vq.jsonl" <<'EOF'
{"kind":"item","id":"q1","ntok":3,"dims":{"4":[0.0000000,1.0000000,0.0000000,0.0000000]}}
{"kind":"item","id":"q2","ntok":3,"dims":{"4":[0.0000000,0.0000000,1.0000000,0.0000000]}}
{"kind":"item","id":"d1","ntok":5,"dims":{"4":[1.0000000,0.0000000,0.0000000,0.0000000]}}
{"kind":"item","id":"d2","ntok":5,"dims":{"4":[0.0000000,1.0000000,0.0000000,0.0000000]}}
{"kind":"item","id":"d3","ntok":5,"dims":{"4":[0.0000000,1.0000000,0.0000000,0.0000000]}}
EOF
printf 'q1\td1\t1\nq2\td3\t1\n' > "$tmp/a-qrels.tsv"
"$SCORE" "$tmp/a-base.jsonl" "$tmp/a-vq.jsonl" "$tmp/a-qrels.tsv" 4 > "$tmp/a.out"

# worked out by hand:
#   agreement  q1 0, q2 0, d1 1, d2 1, d3 1 -> mean 0.6, min 0; query side 0, document side 1
#   baseline   q1: d1 first, margin 1. q2: d2 and d3 tie at cosine 1, so d2 takes the
#              top spot and d3 is rank 2 -> recall@1 (1 + 0)/2 = 0.5, MRR (1 + 1/2)/2 = 0.75
#   vq         q1: top spot moves to d2, d1 falls to rank 3 -> recall@1 0
#              q2: all three scores are 0, so d1 is the top spot and d3 is rank 3
#              -> recall@1 0, MRR (1/3 + 1/3)/2 = 1/3; both queries changed their top document
check "A agreement mean 0.600000000"  "  0.600000000" "$tmp/a.out"
check "A agreement query side 0"      "  0.000000000    1.000000000          2          2          2" "$tmp/a.out"
check "A baseline recall@1 0.5"       "4      baseline    0.500000   1.000000   1.000000   0.750000" "$tmp/a.out"
check "A vq recall@1 0.0, MRR 1/3"    "4      vq          0.000000   1.000000   1.000000   0.333333" "$tmp/a.out"
check "A both top-1 documents changed" "4                   2         0.500000" "$tmp/a.out"
check "A three items under 0.90"      "          2          2          2" "$tmp/a.out"

# ---- case B: numeric ids, the documented tie-break by value -----------------
cat > "$tmp/b-base.jsonl" <<'EOF'
{"kind":"item","id":"1","ntok":2,"dims":{"4":[1.0000000,0.0000000,0.0000000,0.0000000]}}
{"kind":"item","id":"9","ntok":2,"dims":{"4":[1.0000000,0.0000000,0.0000000,0.0000000]}}
{"kind":"item","id":"10","ntok":2,"dims":{"4":[1.0000000,0.0000000,0.0000000,0.0000000]}}
EOF
cp "$tmp/b-base.jsonl" "$tmp/b-vq.jsonl"
printf '1\t10\t1\n' > "$tmp/b-qrels.tsv"
"$SCORE" "$tmp/b-base.jsonl" "$tmp/b-vq.jsonl" "$tmp/b-qrels.tsv" 4 > "$tmp/b.out"

# 9 and 10 tie exactly with the query, so 9 takes the top spot and the labelled
# document 10 is rank 2: recall@1 0, MRR 0.5. Byte order would give MRR 1.
check "B numeric tie-break: rank 2, not 1" "4      baseline    0.000000   1.000000   1.000000   0.500000" "$tmp/b.out"
check "B no top-1 change between identical legs" "4                   0         0.000000" "$tmp/b.out"

echo
if [ "$fails" -eq 0 ]; then
    echo "selftest: PASSED ($oks cases, both fixtures)"
    exit 0
fi
echo "selftest: FAILED ($fails of $((oks + fails)) cases)" >&2
exit 1
