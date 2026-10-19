#!/bin/sh
# eg2_gguf_check.sh — the two-sided check of a GGUF written by eg2_gguf:
#
#   1. the writer's manifest (tools/eg2_gguf.c --manifest) and the ds4-rs reader's
#      own dump (crates/ds4-core/examples/gguf_inventory.rs) must agree on every
#      tensor's name, type, rank, dims, byte count and absolute offset;
#   2. every tensor's bytes must equal the checkpoint's, per the manifest's own
#      st_dtype column:
#        bf16       the matrix is copied verbatim -> cmp -i a:b -n bytes
#        bf16->f32  the 1-D / *_norm.weight tensors the converter writes F32
#                   (conversion/base.py:1244-1246) -> the F32 word's high half
#                   must be the BF16 source element, checked element by element.
#
#   tools/eg2_gguf_check.sh <file.gguf> <manifest.tsv> <hf-dir> <inventory.txt>
#
# The per-tensor verdict goes to stdout; diagnostics go to stderr. Exit non-zero
# on any disagreement.
set -eu
GGUF=$1; MAN=$2; HF=$3; INV=$4
ST="$HF/model.safetensors"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
TAB=$(printf '\t')

# manifest: name type nd dims bytes gguf_abs st_abs st_dtype
awk -F'\t' '!/^#/ { print $1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6"\t"$9"\t"$10 }' "$MAN" | sort > "$TMP/man.tsv"
# reader dump: name type nd dims bytes abs
awk '
  /^T / {
    t = $5; sub(/^type=/, "", t); sub(/\(.*/, "", t);
    print $2"\t"t"\t"substr($3, 6)"\t"substr($4, 6)"\t"substr($7, 7)"\t"substr($9, 5);
  }' "$INV" | sort > "$TMP/inv.tsv"

n_man=$(wc -l < "$TMP/man.tsv")
n_inv=$(wc -l < "$TMP/inv.tsv")
if [ "$n_man" != "$n_inv" ]; then
    echo "the tensor counts differ: writer $n_man, reader $n_inv" >&2
    exit 1
fi

join -t "$TAB" -j 1 "$TMP/man.tsv" "$TMP/inv.tsv" > "$TMP/joined.tsv"

awk -F'\t' '
  {
    bad = 0
    if ($2 != $9)  { printf "type  %s: writer %s, reader %s\n", $1, $2, $9   > "/dev/stderr"; bad = 1 }
    if ($3 != $10) { printf "rank  %s: writer %s, reader %s\n", $1, $3, $10  > "/dev/stderr"; bad = 1 }
    if ($4 != $11) { printf "dims  %s: writer %s, reader %s\n", $1, $4, $11  > "/dev/stderr"; bad = 1 }
    if ($5 != $12) { printf "bytes %s: writer %s, reader %s\n", $1, $5, $12  > "/dev/stderr"; bad = 1 }
    if ($6 != $13) { printf "abs   %s: writer %s, reader %s\n", $1, $6, $13  > "/dev/stderr"; bad = 1 }
    if (bad) { nbad++ }
    else { printf "%s\t%s\t%s\t%s\t%s\n", $1, $6, $7, $5, $8 }
  }
  END { if (nbad) { printf "%d tensors disagree between the writer and the reader\n", nbad > "/dev/stderr"; exit 1 } }
' "$TMP/joined.tsv" > "$TMP/checked.tsv"

copied=0; widened=0; mismatched=0
while IFS="$TAB" read -r name gg st bytes dtype; do
    case "$dtype" in
    bf16)
        if cmp -s -i "$gg:$st" -n "$bytes" "$GGUF" "$ST"; then
            copied=$((copied + 1))
        else
            echo "BYTES DIFFER $name (bf16 copy, gguf $gg, safetensors $st, $bytes bytes)" >&2
            mismatched=$((mismatched + 1))
        fi
        ;;
    bf16-\>f32)
        # the GGUF holds F32; each word's high half (bytes 3 and 4 of the group)
        # must be the BF16 source element (bytes 1 and 2 of its group)
        od -An -tx1 -j "$gg" -N "$bytes" -v "$GGUF" |
            awk '{ for (i = 1; i <= NF; i += 4) printf "%s%s", $(i + 2), $(i + 3) }' > "$TMP/w_gguf"
        src_bytes=$((bytes / 2))
        od -An -tx1 -j "$st" -N "$src_bytes" -v "$ST" |
            awk '{ for (i = 1; i <= NF; i += 2) printf "%s%s", $i, $(i + 1) }' > "$TMP/w_src"
        if cmp -s "$TMP/w_gguf" "$TMP/w_src"; then
            widened=$((widened + 1))
        else
            echo "WIDENING DIFFERS $name (bf16 -> f32, gguf $gg, safetensors $st, $bytes bytes)" >&2
            mismatched=$((mismatched + 1))
        fi
        ;;
    *)
        echo "unknown st_dtype '$dtype' for $name" >&2
        mismatched=$((mismatched + 1))
        ;;
    esac
done < "$TMP/checked.tsv"

echo "checked $((copied + widened)) tensors: writer and ds4-rs reader agree on name/type/rank/dims/bytes/offset;"
echo "  $copied copied byte-for-byte, $widened widened BF16->F32 with every element's high half equal to the source"
[ "$mismatched" = 0 ] || { echo "FAILED: $mismatched tensors differ" >&2; exit 1; }
