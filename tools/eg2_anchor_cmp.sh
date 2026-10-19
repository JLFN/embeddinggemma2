#!/bin/bash
# eg2_anchor_cmp.sh — compare two anchor files produced by eg2_anchor.sh.
#
# Prints, per MRL width, the cosine agreement and the element deltas over the
# items both files carry, plus the worst items by cosine and by element delta.
# Used for the phase-2 measurements (the EOS flag, the prompt prefix) and later
# for phase 4/5, where one side is this project's own engine and the other side
# is the frozen anchor.
#
# Usage: tools/eg2_anchor_cmp.sh <a.jsonl> <b.jsonl> [width ...]
set -euo pipefail
A=$1; B=$2; shift 2
WIDTHS=("$@")
[ ${#WIDTHS[@]} -gt 0 ] || WIDTHS=(768 512 256 128)
for f in "$A" "$B"; do [ -f "$f" ] || { echo "eg2_anchor_cmp.sh: no such file: $f" >&2; exit 2; }; done

flat() { jq -r 'select(.kind=="item") as $it | ($it.dims | to_entries[]) as $d |
        "\($it.id)\t\($d.key)\t\($d.value | join(" "))"' "$1"; }

tmp=$(mktemp -d "${TMPDIR:-/tmp}/eg2-cmp.XXXXXX"); trap 'rm -rf "$tmp"' EXIT
flat "$A" > "$tmp/a.tsv"; flat "$B" > "$tmp/b.tsv"
echo "a: $A"
echo "b: $B"
echo
printf '%-6s %6s %12s %12s %12s %12s %12s\n' width items mincos meancos maxdelt meanabsdev worst_item
for w in "${WIDTHS[@]}"; do
    awk -F'\t' -v OFS=' ' -v want="$w" '
    NR == FNR { if ($2 == want) { k[$1] = $3; next } next }
    $2 == want && ($1 in k) {
        n = split(k[$1], x, " "); m = split($3, y, " ")
        if (n != m) { print "shape mismatch on " $1 > "/dev/stderr"; exit 1 }
        dot = 0; na = 0; nb = 0; mx = 0; ad = 0
        for (i = 1; i <= n; i++) {
            dot += x[i] * y[i]; na += x[i] * x[i]; nb += y[i] * y[i]
            e = x[i] - y[i]; if (e < 0) e = -e
            ad += e; if (e > mx) mx = e
        }
        c = dot / (sqrt(na) * sqrt(nb))
        cnt++
        if (cnt == 1 || c < mn) { mn = c; mwid = $1 }
        if (cnt == 1 || mx > gmx) { gmx = mx; gmid = $1 }
        sc += c; md += mx; adt += ad / n
    }
    END {
        if (cnt == 0) { print want, 0, "-", "-", "-", "-", "-"; exit }
        printf "%s %d %.9f %.9f %.3e %.3e %s\n", want, cnt, mn, sc / cnt, gmx, adt / cnt, mwid
    }' "$tmp/a.tsv" "$tmp/b.tsv" | awk '{ printf "%-6s %6s %12.9f %12.9f %12.3e %12.3e %12s\n", $1, $2, $3, $4, $5, $6, $7 }'
done
echo
echo "(mincos and meancos over items present in both; maxdelt is the largest single"
echo " element difference seen, meanabsdev the mean of the per-item max differences;"
echo " worst_item is the item with the lowest cosine at that width)"
