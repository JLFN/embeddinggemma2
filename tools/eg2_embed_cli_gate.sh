#!/bin/bash
# eg2_embed_cli_gate.sh — the --embed verb's end-to-end gate (phase 6).
#
# What it proves, and against what: the CLI takes RAW TEXT, applies the
# checkpoint's prompt prefix, tokenizes it with the family's own encoder and
# runs the tower -- and the vector it prints must equal the vector the parity
# harness produced for the SAME frozen-set item, on the device, by a completely
# separate driver (out/phase7-gpu-raw.txt, itself cross-checked by
# tools/eg2_parity_check.awk).  Text to vector, two independent paths.
#
# The items come from the frozen set: the queries and pool files carry
# (id, prompt_name, text), and out/phase5-ids.tsv carries the same id with the
# token count the harness used, so the gate can insist the CLI's own token count
# matches before it compares vectors.
#
# Usage:
#   tools/eg2_embed_cli_gate.sh <artifact.gguf> [n_items]
#
# Bounds, declared here BEFORE the run (the same ones P3.3 declared for device
# against CPU): min cos >= 0.999999 and max rel_rms <= 1e-03.
set -eu

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ART=${1:?usage: eg2_embed_cli_gate.sh <artifact.gguf> [n_items] [raw.tsv]}
N=${2:-6}
COS_MIN=0.999999
REL_MAX=1e-03

DIS4="$ROOT/ds4-rs/ds4"
RAW=${3:-$ROOT/out/phase7-gpu-raw.txt}
IDS="$ROOT/out/phase5-ids.tsv"
QUERIES="$ROOT/tests/retrieval/scifact-test-queries.tsv"
POOL="$ROOT/tests/retrieval/scifact-test-pool.tsv"
WORK="$ROOT/out/phase6-embed-cli-gate"

for f in "$DIS4" "$RAW" "$IDS" "$QUERIES" "$POOL"; do
    [ -s "$f" ] || { echo "eg2_embed_cli_gate.sh: missing $f" >&2; exit 2; }
done

rm -rf "$WORK"; mkdir -p "$WORK"

# The items to compare: the first N of the raw's own rows, so every one of them
# has a harness vector to compare against.  The queries come first in the raw,
# then the pool documents, so both prompt names are exercised when N is large
# enough -- and both are always tried explicitly below.
awk -F'\t' '$1 ~ /^[0-9]+$/ { print $1 }' "$RAW" | head -n "$((N / 2))" > "$WORK/ids.txt"
awk -F'\t' '$1 ~ /^[0-9]+$/ { print $1 }' "$RAW" | tail -n "$((N / 2))" >> "$WORK/ids.txt"

# item id -> prompt_name, text
awk -F'\t' '!/^#/ && NF >= 3 { print $1 "\t" $2 "\t" $3 }' "$QUERIES" "$POOL" > "$WORK/items.tsv"

: > "$WORK/results.tsv"
n_run=0; n_failed=0
while read -r id; do
    line=$(awk -F'\t' -v id="$id" '$1 == id { print; exit }' "$WORK/items.tsv")
    if [ -z "$line" ]; then
        echo "  item $id: not in the frozen set's two files, skipped"
        continue
    fi
    pname=$(printf '%s' "$line" | cut -f2)
    text=$(printf '%s' "$line" | cut -f3-)
    want_ntok=$(awk -F'\t' -v id="$id" '$1 == id { print $3; exit }' "$IDS")

    printf '%s' "$text" > "$WORK/item-$id.txt"
    if ! "$DIS4" -m "$ART" --embed-file "$WORK/item-$id.txt" --task "$pname" \
            > "$WORK/item-$id.json" 2> "$WORK/item-$id.err"; then
        echo "  item $id ($pname): the CLI refused or failed: $(tail -1 "$WORK/item-$id.err")"
        n_failed=$((n_failed + 1))
        continue
    fi
    ntok=$(awk -F'"' '/"n_tokens"/ { print $3 }' "$WORK/item-$id.json" | tr -dc '0-9')
    sed -n '/"embedding": \[/,/\]/p' "$WORK/item-$id.json" \
        | tr -d '\n' | sed 's/.*\[//; s/\].*//' > "$WORK/item-$id.vec"
    awk -F'\t' -v id="$id" '$1 == id { print $3; exit }' "$RAW" > "$WORK/item-$id.ref"

    awk -v id="$id" -v pname="$pname" -v ntok="$ntok" -v want="$want_ntok" '
        # NR == FNR is the first FILE (the CLI vector, comma separated); the
        # second file is the harness raw vector (space separated).  FNR == 1
        # would be wrong here: it is true for the first record of EACH file,
        # which silently left the raw unsplit and made every item "MALFORMED".
        function abs(x) { return x < 0 ? -x : x }
        NR == FNR { n = split($0, v, ","); next }
        { m = split($0, r, " ") }
        END {
            if (n != 768 || m != 768) {
                printf "%s\t%s\t%s\t%s\tMALFORMED\tMALFORMED\n", id, pname, ntok, want
                exit
            }
            dot = 0; a2 = 0; b2 = 0; d2 = 0
            for (i = 1; i <= 768; i++) {
                a = v[i] + 0.0; b = r[i] + 0.0
                dot += a * b; a2 += a * a; b2 += b * b; d2 += (a - b) * (a - b)
            }
            printf "%s\t%s\t%s\t%s\t%.9f\t%.3e\n", id, pname, ntok, want,
                   dot / (sqrt(a2) * sqrt(b2)), sqrt(d2 / b2)
        }
    ' "$WORK/item-$id.vec" "$WORK/item-$id.ref" >> "$WORK/results.tsv"
    n_run=$((n_run + 1))
done < "$WORK/ids.txt"

echo
echo "== the --embed verb against the device raw =="
printf '%-10s %-20s %6s %6s %12s %10s\n' item task ntok ids_ntok cos rel_rms
awk -F'\t' '{ printf "%-10s %-20s %6s %6s %12s %10s\n", $1, $2, $3, $4, $5, $6 }' "$WORK/results.tsv"
echo
awk -F'\t' -v cmin="$COS_MIN" -v rmax="$REL_MAX" -v failed="$n_failed" -v ran="$n_run" '
    $5 == "MALFORMED" || $6 == "MALFORMED" { bad++; next }
    { n++
      if (n == 1 || $5 + 0.0 < mincos) mincos = $5 + 0.0
      if (n == 1 || $6 + 0.0 > maxrel) maxrel = $6 + 0.0
      if ($3 + 0 != $4 + 0) tok_mismatch++
    }
    END {
        printf "compared          %d items (ran %d, refused/failed %d)\n", n, ran, failed
        printf "token mismatches  %d (against the harness ids file)\n", tok_mismatch + 0
        printf "min cos           %.9f   (declared >= %s)\n", mincos, cmin
        printf "max rel_rms       %.3e   (declared <= %s)\n", maxrel, rmax
        printf "malformed         %d\n", bad + 0
        if (n < 1 || bad > 0 || failed > 0 || tok_mismatch > 0 ||
            mincos + 0.0 < cmin + 0.0 || maxrel + 0.0 > rmax + 0.0) {
            print "EG2VQ_EMBED_CLI_FAIL"
            exit 1
        }
        print "EG2VQ_EMBED_CLI_PASS"
    }
' "$WORK/results.tsv"
