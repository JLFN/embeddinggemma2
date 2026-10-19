#!/bin/sh
# eg2_loader_check.sh — gate 3: the `gemma_vq` loader in the ds4 host.
#
#   1. the writer's manifest and the ds4-rs reader's own dump agree on every
#      tensor's name, type, rank, dims, byte count and absolute offset, and the
#      family leg reports its identity, metadata contract, bind set and layout;
#   2. two loads of the same file are byte-identical (the dump, and so every
#      payload check the loader ran);
#   3. each tampered copy is REFUSED, by name, rather than decoded: a wrong
#      payload magic, a wrong `dim`, a wrong `rows`, a zero row gain, a
#      nonfinite codebook word, a nonzero byte in the alignment padding, a
#      truncated file, and a wrong metadata value.
#
#   tools/eg2_loader_check.sh <artifact.gguf> <manifest.tsv>
#
# The reader command is overridable with DS4_INVENTORY so the gate can run
# without a cargo build (it must print the same dump and take the same
# arguments). Exit non-zero on any disagreement.
set -eu
GGUF=$1; MAN=$2
: "${DS4_INVENTORY:=cargo run -q -p ds4-core --release --example gguf_inventory --}"
: "${DS4_ROOT:=ds4-rs}"
TMP=$(mktemp -d "${TMPDIR:-/tmp}/eg2-loader-check.XXXXXX")
trap 'rm -rf "$TMP"' EXIT

dump() {
    # $1 = file, $2 = output; the family leg's verdicts are part of the dump
    ( cd "$DS4_ROOT" && $DS4_INVENTORY "$1" --no-kv --family ) > "$2"
}

# ---- 1. the writer against the reader -------------------------------------
cp "$GGUF" "$TMP/base.gguf"
dump "$TMP/base.gguf" "$TMP/inv.txt"

for line in "IDENTIFY EmbeddingGemma-2 VQ family=16 variant=19" "META ok" \
            "BIND ok slots=1376" "LAYOUT ok specs=1376"; do
    grep -qxF "$line" "$TMP/inv.txt" || {
        echo "the family leg did not report '$line'" >&2
        grep -E '^(IDENTIFY|META|BIND|LAYOUT)' "$TMP/inv.txt" | head -5 >&2
        exit 1
    }
done

awk -F'\t' '!/^#/ { print $1"\t"$2"\t"$3"\t"$4"\t"$5"\t"$6 }' "$MAN" | sort > "$TMP/man.tsv"
awk '
  /^T / {
    t = $5; sub(/^type=/, "", t); sub(/\(.*/, "", t);
    print $2"\t"t"\t"substr($3, 6)"\t"substr($4, 6)"\t"substr($7, 7)"\t"substr($9, 5);
  }' "$TMP/inv.txt" | sort > "$TMP/inv.tsv"

n_man=$(wc -l < "$TMP/man.tsv")
n_inv=$(wc -l < "$TMP/inv.tsv")
[ "$n_man" = "$n_inv" ] || {
    echo "the tensor counts differ: writer $n_man, reader $n_inv" >&2
    exit 1
}
for f in "$TMP/man.tsv" "$TMP/inv.tsv"; do
    dup=$(cut -f 1 "$f" | sort | uniq -d | head -3)
    [ -z "$dup" ] || { echo "duplicate tensor names in $(basename "$f"): $dup" >&2; exit 1; }
done
cmp -s "$TMP/man.tsv" "$TMP/inv.tsv" || {
    echo "the writer's manifest and the reader's dump disagree; first differences:" >&2
    diff "$TMP/man.tsv" "$TMP/inv.tsv" | head -5 >&2
    exit 1
}

# ---- 2. two loads are identical -------------------------------------------
dump "$TMP/base.gguf" "$TMP/inv2.txt"
cmp -s "$TMP/inv.txt" "$TMP/inv2.txt" || {
    echo "two loads of the same file produced different dumps" >&2
    exit 1
}

# ---- 3. a tampered file is refused, not decoded ---------------------------
# Offsets are the manifest's own absolute offsets. The first VQ tensor's payload
# is [u32 magic][u16 dim][u16 nc][u32 rows][u32 cols], then the codebook
# (nc*dim*2 bytes) and the row gains (rows*2 bytes); the contract's section 2
# measured dim 4 and nc 4096, so the header is 16 bytes and the codebook 32768.
first_vq=$(awk -F'\t' '!/^#/ && $2 == 42 { print $6; exit }' "$MAN")
[ -n "$first_vq" ] || { echo "the manifest has no type-42 tensor" >&2; exit 1; }
gain_off=$((first_vq + 16 + 32768))
# The first byte of inter-tensor alignment padding, from the manifest itself.
pad=$(awk -F'\t' '!/^#/ {
    if (n && $6 > prev_end) { print prev_end; exit }
    prev_end = $6 + $5; n = 1
}' "$MAN")
[ -n "$pad" ] || { echo "the manifest has no alignment padding to test" >&2; exit 1; }
# `gemma_vq.block_count` is a 20-byte key: the entry is [u64 len][key][u32 type]
# [value], so the value sits 20 + 4 bytes past the key's first byte.
bc_key=$(grep -abo 'gemma_vq.block_count' "$TMP/base.gguf" | head -1 | cut -d: -f1)
[ -n "$bc_key" ] || { echo "cannot locate gemma_vq.block_count in the file" >&2; exit 1; }
bc_val=$((bc_key + 20 + 4))

refused() {
    # $1 = case name, $2 = token the loader must report, $3 = byte offset,
    # $4 = the printf escape sequence to write there
    case_name=$1; want=$2; off=$3; bytes=$4
    cp --reflink=auto "$TMP/base.gguf" "$TMP/t.gguf" 2>/dev/null ||
        cp "$TMP/base.gguf" "$TMP/t.gguf"
    printf "$bytes" | dd of="$TMP/t.gguf" bs=1 seek="$off" conv=notrunc status=none
    set +e
    dump "$TMP/t.gguf" "$TMP/t.txt" 2> "$TMP/t.err"
    status=$?
    set -e
    [ "$status" != 0 ] || {
        echo "$case_name: the loader accepted a tampered file" >&2
        exit 1
    }
    grep -qF "$want" "$TMP/t.err" || {
        echo "$case_name: refused, but not with '$want'" >&2
        tail -1 "$TMP/t.err" >&2
        exit 1
    }
    printf '  refused %-20s %s\n' "$case_name" "$(tail -1 "$TMP/t.err" | sed 's/^Error: //')"
}

echo "tamper cases (each must be refused by name):"
refused "wrong magic" "vq payload magic" "$first_vq" '\000'
refused "wrong dim" "vq payload byte account" "$((first_vq + 4))" '\010\000'
refused "wrong rows" "vq payload byte account" "$((first_vq + 8))" '\377\017\000\000'
refused "zero row gain" "zero row gain" "$gain_off" '\000\000'
refused "nonfinite codebook" "nonfinite codebook" "$((first_vq + 16))" '\000\174'
refused "nonzero padding" "nonzero padding" "$pad" '\001'
refused "metadata value" "gemma_vq.block_count" "$bc_val" '\031\000\000\000'

# The truncated file is the one case that needs a different tool.
cp --reflink=auto "$TMP/base.gguf" "$TMP/t.gguf" 2>/dev/null || cp "$TMP/base.gguf" "$TMP/t.gguf"
truncate -s "$((first_vq + 4096))" "$TMP/t.gguf"
set +e
dump "$TMP/t.gguf" "$TMP/t.txt" 2> "$TMP/t.err"
status=$?
set -e
[ "$status" != 0 ] || { echo "truncated file: the loader accepted it" >&2; exit 1; }
grep -qF "OutsideFile" "$TMP/t.err" || {
    echo "truncated file: refused, but not with 'OutsideFile'" >&2
    tail -1 "$TMP/t.err" >&2
    exit 1
}
printf '  refused %-20s %s\n' "truncated file" "$(tail -1 "$TMP/t.err" | sed 's/^Error: //')"

# the untouched copy still loads: the harness cannot pass by failing everything
dump "$TMP/base.gguf" "$TMP/final.txt"
cmp -s "$TMP/inv.txt" "$TMP/final.txt" || {
    echo "the untouched copy no longer loads" >&2
    exit 1
}

echo "checked $n_man tensors: the writer's manifest and the ds4-rs reader agree on"
echo "  name/type/rank/dims/bytes/offset; the family leg reports identity, metadata"
echo "  contract, bind set and layout; two loads are identical; and 8 tampered"
echo "  copies were refused by name"
