#!/bin/bash
# eg2_retrieval_score.sh — score the phase-5 measurement: what the VQ codec costs.
#
# Two sides of the comparison arrive as the same jsonl shape the anchor uses
# (tools/eg2_oracle_to_jsonl.sh), one file per weight leg, and the labels are the
# frozen SciFact test split (tools/eg2_retrieval_set.sh). Per MRL width this
# prints:
#
#   the agreement distribution   cosine(baseline item, artifact item) over every
#                                item of the set: mean, min, the 1/5/25/50/75th
#                                percentiles, the mean split by side (query items
#                                average ~26 tokens, document items ~349, so their
#                                pooling lengths differ by an order of magnitude),
#                                and how many items fall below 0.99/0.95/0.90
#   the retrieval table          recall@1/5/10 and MRR@10 for each leg, the delta,
#                                the number of queries whose top-1 document
#                                changed, and the baseline's mean top1-top2 margin
#                                (so the deltas can be read against how close the
#                                ties were)
#
# Nothing here is a threshold: phase 5's gate is that the numbers are committed
# with the hashes they were measured on, not that they pass a bar this script
# invented. What IS asserted are invariants that would make the numbers wrong:
# every vector unit-length, every stored truncated vector equal to the
# renormalized leading part of the same item's widest vector, every item present
# in both legs, and the label set's query and document counts reproduced from the
# score files themselves.
#
# Ranking: cosine (both sides are unit vectors, so the dot product IS the cosine),
# highest first, ties broken by document id ascending so the order is total and
# re-derivable. recall@k counts the relevant documents retrieved in the top k over
# the query's number of relevant documents (SciFact's test split: 277 queries with
# one, 14 with two, 4 with three, 3 with four and 2 with five); MRR@10 is the
# reciprocal rank of the FIRST relevant document, 0 when none is in the top 10.
#
# Usage: tools/eg2_retrieval_score.sh <baseline.jsonl> <artifact.jsonl> <qrels.tsv|- > [width ...]
#
# Pass "-" as the third argument to score agreement only, with no labels: that is
# how the phase-2 sentence set is reported (the codec's cost on the set the anchor
# covers, where retrieval metrics would be meaningless because the set has no
# retrieval labels).
set -euo pipefail
BASE=${1:?usage: eg2_retrieval_score.sh <baseline.jsonl> <artifact.jsonl> <qrels.tsv|- > [width ...]}
ART=${2:?}
QRELS=${3:?}
shift 3
WIDTHS=("$@")
[ ${#WIDTHS[@]} -gt 0 ] || WIDTHS=(768 512 256 128)
LABELLED=1
[ "$QRELS" = "-" ] && LABELLED=0

for f in "$BASE" "$ART"; do
    [ -s "$f" ] || { echo "eg2_retrieval_score.sh: missing $f" >&2; exit 2; }
done
[ "$LABELLED" = 0 ] || [ -s "$QRELS" ] || { echo "eg2_retrieval_score.sh: missing $QRELS" >&2; exit 2; }

tmp=$(mktemp -d "${TMPDIR:-/tmp}/eg2-score.XXXXXX"); trap 'rm -rf "$tmp"' EXIT
flat() { jq -r 'select(.kind=="item") as $it | ($it.dims | to_entries[]) as $d |
        "\($it.id)\t\($d.key)\t\($d.value | join(" "))"' "$1"; }
flat "$BASE" > "$tmp/b.tsv"
flat "$ART" > "$tmp/a.tsv"
if [ "$LABELLED" = 1 ]; then grep -v '^#' "$QRELS" > "$tmp/qrels.tsv"; else : > "$tmp/qrels.tsv"; fi

awk -v bfile="$tmp/b.tsv" -v afile="$tmp/a.tsv" -v qfile="$tmp/qrels.tsv" -v labelled="$LABELLED" \
    -v outdir="$tmp" -v widthlist="${WIDTHS[*]}" '
function die(msg) { printf "eg2_retrieval_score.sh: %s\n", msg > "/dev/stderr"; exit 1 }

# The total order on document ids, which the ranking needs for exact ties: by
# value when both ids are numeric (the datasets are numeric), by byte order when
# they are not, so the order is total either way and a re-run reproduces it.
function idlt(a, b) {
    if (a ~ /^[0-9]+$/ && b ~ /^[0-9]+$/) return (a + 0) < (b + 0)
    return a < b
}

# Load one leg into a flat vector array with integer offsets per (item, width).
# The ranking pool is taken from the ITEM side (every scored item that is not a
# labelled query is a candidate document), never from the labels: a pool that
# contained a distractor the labels do not mention must still be ranked against,
# or the metrics would be flattered by the omission.
function load(file, vec, off, tag,   ln, f, id, w, n, t, i, key, base) {
    while ((getline ln < file) > 0) {
        split(ln, f, "\t")
        id = f[1]; w = f[2] + 0
        if (!(w in want)) continue
        n = split(f[3], t, " ")
        if (n != w) die("item " id " width " w " has " n " values")
        key = id SUBSEP w
        if (key in off) die("item " id " width " w " appears twice in " tag)
        base = nextoff; off[key] = base; nextoff += w
        for (i = 1; i <= w; i++) vec[base + i] = t[i]
        if (tag == "baseline") {
            nkeys++; keys[nkeys] = key; idof[nkeys] = id; wof[nkeys] = w
            if (!(id in seen_id)) { seen_id[id] = 1; n_items++ }
            if (id in qset) {
                if (!(id in seen_q)) { seen_q[id] = 1; n_qitems++ }
            } else {
                if (!(id in seen_d)) { seen_d[id] = 1; n_ditems++ }
                if (!(id in dlist)) { nd++; dlist[id] = nd; did[nd] = id }
            }
        }
    }
    close(file)
}

BEGIN {
    nw = split(widthlist, WW, " ")
    width_max = 0
    for (i = 1; i <= nw; i++) {
        w = WW[i] + 0
        want[w] = 1
        if (w > width_max) width_max = w
    }

    # ---- the labels ---------------------------------------------------------
    while (labelled && (getline ln < qfile) > 0) {
        if (ln == "" || ln ~ /^#/) continue
        split(ln, f, "\t")
        if (f[3] + 0 < 1) continue
        q = f[1]; d = f[2]
        qset[q] = 1
        if (!(d in dset)) { dset[d] = 1; n_dlabel++ }
        if (!((q SUBSEP d) in rel)) { rel[q SUBSEP d] = 1; nrel[q]++; npairs++ }
    }
    close(qfile)
    for (q in qset) nq++
    for (q in qset) { nqorder++; order[nqorder] = q }    # the query list, order-free by construction

    # ---- the two legs -------------------------------------------------------
    load(bfile, vb, ob, "baseline")
    load(afile, va, oa, "artifact")

    # A labelled document that is not among the scored items would be a
    # relationship the ranking can never satisfy; that is an input error, not a
    # low score.
    for (d in dset) {
        if (d in qset) die("id " d " is a labelled query and a labelled document")
        if (!(d in dlist)) die("labelled document " d " is not among the scored items")
    }
    if (labelled && nd != n_ditems) die("the ranking pool is " nd " documents but the document side has " n_ditems " items")

    # ---- invariants over every item and width both legs carry ---------------
    for (k = 1; k <= nkeys; k++) {
        key = keys[k]; w = wof[k]; id = idof[k]
        if (!(key in oa)) { printf "eg2_retrieval_score.sh: item %s width %d missing from the artifact leg\n", id, w > "/dev/stderr"; missing++ }
        obk = ob[key]
        nb = 0
        for (i = 1; i <= w; i++) nb += vb[obk + i] * vb[obk + i]
        dev = sqrt(nb) - 1; if (dev < 0) dev = -dev; if (dev > normdev_b) normdev_b = dev
        oak = oa[key]
        na = 0
        for (i = 1; i <= w; i++) na += va[oak + i] * va[oak + i]
        dev = sqrt(na) - 1; if (dev < 0) dev = -dev; if (dev > normdev_a) normdev_a = dev
        # the MRL invariant: a truncated vector equals the leading dimensions of
        # that same item at the widest width, renormalized after the cut
        if (w < width_max) {
            k768 = id SUBSEP width_max
            if (k768 in ob) {
                s = 0
                for (i = 1; i <= w; i++) s += vb[ob[k768] + i] * vb[ob[k768] + i]
                t = sqrt(s); if (t == 0) t = 1
                for (i = 1; i <= w; i++) {
                    e = vb[obk + i] - vb[ob[k768] + i] / t
                    if (e < 0) e = -e; if (e > truncdev) truncdev = e
                }
            }
        }
    }
    if (missing) die(missing " items missing from the artifact leg")

    # ---- what the codec costs, item by item ---------------------------------
    for (k = 1; k <= nkeys; k++) {
        key = keys[k]; w = wof[k]; id = idof[k]
        obk = ob[key]; oak = oa[key]
        dot = 0
        for (i = 1; i <= w; i++) dot += vb[obk + i] * va[oak + i]
        cnt[w]++; ssum[w] += dot
        if (id in qset) { qn[w]++; qsum[w] += dot } else { dn[w]++; dsum[w] += dot }
        if (cnt[w] == 1 || dot < amin[w]) { amin[w] = dot; amin_id[w] = id }
        if (dot < 0.99) tail99[w]++
        if (dot < 0.95) tail95[w]++
        if (dot < 0.90) tail90[w]++
        printf "%.9f\n", dot > (outdir "/cos-" w ".txt")
    }

    # ---- retrieval ----------------------------------------------------------
    if (labelled) for (wi = 1; wi <= nw; wi++) {
        w = WW[wi] + 0
        nchg = 0; summarg = 0
        r1b = 0; r5b = 0; r10b = 0; mrrb = 0
        r1a = 0; r5a = 0; r10a = 0; mrra = 0
        for (qi = 1; qi <= nqorder; qi++) {
            q = order[qi]
            qkb = q SUBSEP w
            qba = ob[qkb]; qaa = oa[qkb]
            top1b = -2; top2b = -2; argb = ""
            top1a = -2; arga = ""
            for (di = 1; di <= nd; di++) {
                dkb = did[di] SUBSEP w
                dbb = ob[dkb]; daa = oa[dkb]
                sb = 0; sa = 0
                for (i = 1; i <= w; i++) {
                    sb += vb[qba + i] * vb[dbb + i]
                    sa += va[qaa + i] * va[daa + i]
                }
                scoreb[di] = sb; scorea[di] = sa
                if (sb > top1b || (sb == top1b && idlt(did[di], argb))) { top2b = top1b; top1b = sb; argb = did[di] }
                else if (sb > top2b) top2b = sb
                if (sa > top1a || (sa == top1a && idlt(did[di], arga))) { top1a = sa; arga = did[di] }
            }
            summarg += top1b - top2b
            if (argb != arga) nchg++
            hitb = 0; hit5b = 0; hit10b = 0; firstb = 0
            hita = 0; hit5a = 0; hit10a = 0; firsta = 0
            for (j = 1; j <= nd; j++) {
                rd = did[j]
                if (!((q SUBSEP did[j]) in rel)) continue
                rb = 1; ra = 1
                for (di = 1; di <= nd; di++) {
                    if (scoreb[di] > scoreb[j]) rb++
                    else if (scoreb[di] == scoreb[j] && idlt(did[di], rd)) rb++
                    if (scorea[di] > scorea[j]) ra++
                    else if (scorea[di] == scorea[j] && idlt(did[di], rd)) ra++
                }
                if (rb <= 1) hitb++
                if (rb <= 5) hit5b++
                if (rb <= 10) hit10b++
                if (ra <= 1) hita++
                if (ra <= 5) hit5a++
                if (ra <= 10) hit10a++
                if (firstb == 0 || rb < firstb) firstb = rb
                if (firsta == 0 || ra < firsta) firsta = ra
            }
            nr = nrel[q]
            r1b += hitb / nr; r5b += hit5b / nr; r10b += hit10b / nr
            r1a += hita / nr; r5a += hit5a / nr; r10a += hit10a / nr
            mrrb += (firstb >= 1 && firstb <= 10) ? 1 / firstb : 0
            mrra += (firsta >= 1 && firsta <= 10) ? 1 / firsta : 0
        }
        printf "RETR\t%d\tbase\t%.6f\t%.6f\t%.6f\t%.6f\n", w, r1b / nq, r5b / nq, r10b / nq, mrrb / nq
        printf "RETR\t%d\tvq\t%.6f\t%.6f\t%.6f\t%.6f\n", w, r1a / nq, r5a / nq, r10a / nq, mrra / nq
        printf "TOP1CHANGED\t%d\t%d\n", w, nchg
        printf "MARGIN\t%d\t%.6f\n", w, summarg / nq
    }

    for (wi = 1; wi <= nw; wi++) {
        w = WW[wi] + 0
        printf "AGREE\t%d\t%d\t%.9f\t%.9f\t%s\t%.9f\t%.9f\t%d\t%d\t%d\n", w, cnt[w], ssum[w] / cnt[w], \
            amin[w], amin_id[w], (qn[w] ? qsum[w] / qn[w] : 0), (dn[w] ? dsum[w] / dn[w] : 0), tail99[w], tail95[w], tail90[w]
    }
    printf "INVARIANT\titems\t%d\n", n_items
    printf "INVARIANT\titem_width_rows\t%d\n", nkeys
    printf "INVARIANT\tquery_items\t%d\n", n_qitems
    printf "INVARIANT\tdocument_items\t%d\n", n_ditems
    printf "INVARIANT\tqueries_in_labels\t%d\n", nq
    printf "INVARIANT\tdocuments_in_labels\t%d\n", n_dlabel
    printf "INVARIANT\tdocuments_ranked\t%d\n", nd
    if (labelled) printf "INVARIANT\tlabelled_pairs\t%d\n", npairs
    printf "INVARIANT\tunit_norm_max_dev_baseline\t%.3e\n", normdev_b
    printf "INVARIANT\tunit_norm_max_dev_artifact\t%.3e\n", normdev_a
    printf "INVARIANT\ttrunc_consistency_max_dev\t%.3e\n", truncdev
}
' > "$tmp/summary.tsv"

inv() { awk -F'\t' -v k="$1" '$1 == "INVARIANT" && $2 == k { print $3; found = 1 } END { if (!found) print "?" }' "$tmp/summary.tsv"; }
# The per-width cosine lists come out of awk in item order, so they are sorted
# once here: percentiles of an unsorted list are meaningless (and they are not
# monotone, which is how the first version of this file was caught).
for w in "${WIDTHS[@]}"; do sort -g "$tmp/cos-$w.txt" > "$tmp/cos-$w.sorted"; done
pct() { awk -v p="$2" '{ a[NR] = $1 } END {
        if (NR == 0) { print "nan"; exit }
        idx = int(p * NR / 100 + 0.5); if (idx < 1) idx = 1; if (idx > NR) idx = NR
        printf "%.9f", a[idx]
    }' "$1"; }
# An invariant of the percentiles themselves: they rise with the percentile. This
# is what caught the version that read the unsorted list (p25 came out above p50).
for w in "${WIDTHS[@]}"; do
    awk -v w="$w" -v a="$(pct "$tmp/cos-$w.sorted" 1)" \
        -v b="$(pct "$tmp/cos-$w.sorted" 5)" -v c="$(pct "$tmp/cos-$w.sorted" 25)" \
        -v d="$(pct "$tmp/cos-$w.sorted" 50)" -v e="$(pct "$tmp/cos-$w.sorted" 75)" \
        'BEGIN { if (!(a <= b && b <= c && c <= d && d <= e)) { printf "eg2_retrieval_score.sh: percentiles are not monotone at width %s\n", w > "/dev/stderr"; exit 1 } }' \
        || exit 1
done

if [ "$LABELLED" = 1 ]; then
    echo "set: $(inv queries_in_labels) labelled queries, $(inv documents_in_labels) pooled documents, $(inv labelled_pairs) pairs"
    echo "     $(inv items) items per leg ($(inv query_items) query side, $(inv document_items) document side)"
else
    echo "agreement only: no labels given, so no retrieval metric is computed;"
    echo "     $(inv items) items per leg, all on the sentence side"
fi
echo

echo "== the codec's cost: cosine(baseline item, artifact item), every item, per width =="
printf '%-6s %6s %12s %12s %12s %12s %12s %12s %12s %12s\n' \
    width items meancos mincos p1 p5 p25 p50 p75 worst_item
for w in "${WIDTHS[@]}"; do
    row=$(awk -F'\t' -v w="$w" '$1 == "AGREE" && $2 == w { print $3, $4, $5, $6 }' "$tmp/summary.tsv")
    printf '%-6s %6s %12s %12s %12s %12s %12s %12s %12s %12s\n' "$w" \
        "$(echo "$row" | cut -d' ' -f1)" "$(echo "$row" | cut -d' ' -f2)" "$(echo "$row" | cut -d' ' -f3)" \
        "$(pct "$tmp/cos-$w.sorted" 1)" "$(pct "$tmp/cos-$w.sorted" 5)" "$(pct "$tmp/cos-$w.sorted" 25)" \
        "$(pct "$tmp/cos-$w.sorted" 50)" "$(pct "$tmp/cos-$w.sorted" 75)" "$(echo "$row" | cut -d' ' -f4)"
done
echo

if [ "$LABELLED" = 1 ]; then
echo "== the same agreement, split by side, and how deep the tail runs =="
printf '%-6s %14s %14s %10s %10s %10s\n' width 'query side' 'document side' 'n<0.99' 'n<0.95' 'n<0.90'
for w in "${WIDTHS[@]}"; do
    row=$(awk -F'\t' -v w="$w" '$1 == "AGREE" && $2 == w { print $7, $8, $9, $10, $11 }' "$tmp/summary.tsv")
    printf '%-6s %14s %14s %10s %10s %10s\n' "$w" \
        "$(echo "$row" | cut -d' ' -f1)" "$(echo "$row" | cut -d' ' -f2)" "$(echo "$row" | cut -d' ' -f3)" \
        "$(echo "$row" | cut -d' ' -f4)" "$(echo "$row" | cut -d' ' -f5)"
done
echo
else
echo "== how deep the tail runs (all items are sentences here, no side split) =="
printf "%-6s %14s %12s %10s %10s %10s\n" width mincos worst_item "n<0.99" "n<0.95" "n<0.90"
for w in "${WIDTHS[@]}"; do
    row=$(awk -F'\t' -v w="$w" '$1 == "AGREE" && $2 == w { print $5, $6, $9, $10, $11 }' "$tmp/summary.tsv")
    printf '%-6s %14s %12s %10s %10s %10s\n' "$w" \
        "$(echo "$row" | cut -d' ' -f1)" "$(echo "$row" | cut -d' ' -f2)" "$(echo "$row" | cut -d' ' -f3)" \
        "$(echo "$row" | cut -d' ' -f4)" "$(echo "$row" | cut -d' ' -f5)"
done
echo
fi

if [ "$LABELLED" = 1 ]; then
echo "== retrieval over the frozen labelled set (rank by cosine, ties by document id) =="
printf '%-6s %-9s %10s %10s %10s %10s %11s\n' width leg recall@1 recall@5 recall@10 MRR@10 'delta MRR'
for w in "${WIDTHS[@]}"; do
    base=$(awk -F'\t' -v w="$w" '$1 == "RETR" && $2 == w && $3 == "base" { print $4, $5, $6, $7 }' "$tmp/summary.tsv")
    vq=$(awk -F'\t' -v w="$w" '$1 == "RETR" && $2 == w && $3 == "vq" { print $4, $5, $6, $7 }' "$tmp/summary.tsv")
    bm=$(echo "$base" | cut -d' ' -f4); am=$(echo "$vq" | cut -d' ' -f4)
    printf '%-6s %-9s %10s %10s %10s %10s %11s\n' "$w" baseline \
        "$(echo "$base" | cut -d' ' -f1)" "$(echo "$base" | cut -d' ' -f2)" "$(echo "$base" | cut -d' ' -f3)" "$bm" '-'
    printf '%-6s %-9s %10s %10s %10s %10s %11s\n' "$w" vq \
        "$(echo "$vq" | cut -d' ' -f1)" "$(echo "$vq" | cut -d' ' -f2)" "$(echo "$vq" | cut -d' ' -f3)" "$am" \
        "$(awk -v a="$am" -v b="$bm" 'BEGIN { printf "%+.6f", a - b }')"
done
echo

echo "== how close were the calls the codec could have changed =="
printf '%-6s %14s %16s\n' width 'top-1 changed' 'mean top1-top2'
for w in "${WIDTHS[@]}"; do
    printf '%-6s %14s %16s\n' "$w" \
        "$(awk -F'\t' -v w="$w" '$1 == "TOP1CHANGED" && $2 == w { print $3 }' "$tmp/summary.tsv")" \
        "$(awk -F'\t' -v w="$w" '$1 == "MARGIN" && $2 == w { print $3 }' "$tmp/summary.tsv")"
done
echo
fi

echo "== invariants (each of these failing would make the numbers wrong) =="
awk -F'\t' '$1 == "INVARIANT" { printf "%-30s %s\n", $2, $3 }' "$tmp/summary.tsv"
