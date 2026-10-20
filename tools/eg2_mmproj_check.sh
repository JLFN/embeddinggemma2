#!/bin/sh
# eg2_mmproj_check.sh — gate 22: the vision mmproj, checked with THIS PROJECT's
# OWN reader (phase 8, unit 22; evidence docs/eg2vq-vision-oracle-gate22.txt).
#
# Owner directive 2026-10-20: "We will not use llama it must be ported all."  The
# pinned llama.cpp is the AUTHORITY this port reads and cites (rule 0.2) and it
# produced the frozen numbers in tests/vision/reference/ ONCE, but no step of the
# port, no build, no run and no DEFAULT gate executes it.  This gate therefore
# reads the mmproj back with the Rust host's own GGUF reader
# (crates/ds4-core/examples/gguf_inventory).  The one place in this repository
# that still executes the pin is tools/eg2_vision_anchor.sh, opt-in and never
# called from here or from tests/run.sh.
#
#   1. the writer rebuilds from source with the command recorded here;
#   2. the writer is DETERMINISTIC: two writes of the same checkpoint are
#      byte-identical, and the digest equals the recorded one;
#   3. OUR reader reads the file back: 211 tensors whose name, type, rank, dims
#      and byte count equal the writer's manifest, and the 19 clip.* metadata
#      keys equal the recorded list;
#   4. the frozen reference is intact and its two fixtures are DIFFERENT answers
#      (cosine below 0.999 between their committed pooled vectors) — a gate that
#      cannot tell two images apart is not measuring an image path at all;
#   5. the frozen reference's own digests are in SHA256SUMS, checked by
#      tests/run.sh step 1 before this gate runs.
#
# The side of the check that the pin used to carry — "the reference's own loader
# accepts this file" and "the frozen numbers reproduce from a fresh run" — moves
# to unit 23's gate, where the PORT reproduces those same numbers natively.  That
# is the stronger statement of the two: it is the consumer, not a bystander.
#
# Usage: tools/eg2_mmproj_check.sh
#   Run from the repository root.  Env: HF (checkpoint dir), MMPROJ_SHA (the
#   recorded digest), DS4_ROOT / DS4_INVENTORY (the reader command).
set -eu
REPO=$(pwd)
: "${HF:=$REPO/.models/EmbeddingGemma-2-914f7f8}"
: "${YA:=/data/YoungAi}"
: "${MMPROJ_SHA:=9ffb86e3eeb5834095e0bd42b6f5fc0505f02b8e208a6037d01db448e6566838}"
: "${DS4_ROOT:=$REPO/ds4-rs}"
: "${DS4_INVENTORY:=cargo run -q -p ds4-core --release --example gguf_inventory --}"
MMPROJ=$REPO/out/EmbeddingGemma-2-914f7f8-vision-mmproj.gguf
MANIFEST=$REPO/out/mmproj-manifest.tsv
TMP=$(mktemp -d "${TMPDIR:-/tmp}/eg2-mmproj-check.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$REPO/out"

fail() { echo "MMPROJ GATE: FAIL: $1" >&2; exit 1; }

# ---- 1. the writer rebuilds -------------------------------------------------
cc -O3 -std=c99 -Wall -D_POSIX_C_SOURCE=200809L -I"$REPO/tools" -I"$YA/src/common" \
   -o "$REPO/bin/eg2_mmproj" "$REPO/tools/eg2_mmproj.c" -lm || fail "the mmproj writer did not build"

# ---- 2. the writer, twice ---------------------------------------------------
"$REPO/bin/eg2_mmproj" "$HF" "$MMPROJ" --manifest "$MANIFEST" > "$TMP/write1.log" || fail "the first write failed"
"$REPO/bin/eg2_mmproj" "$HF" "$TMP/again.gguf" > "$TMP/write2.log" || fail "the second write failed"
cmp -s "$MMPROJ" "$TMP/again.gguf" || fail "two writes of the same checkpoint differ (the writer is not deterministic)"
GOT=$(sha256sum "$MMPROJ" | cut -d' ' -f1)
[ "$GOT" = "$MMPROJ_SHA" ] || fail "the mmproj digest is $GOT, recorded $MMPROJ_SHA"

# ---- 3. our own reader, and the metadata contract ---------------------------
( cd "$DS4_ROOT" && $DS4_INVENTORY "$MMPROJ" ) > "$TMP/inv.txt" 2>"$TMP/inv.err" \
    || fail "the Rust host's reader refused the mmproj (see $TMP/inv.err)"

grep -qx 'HEADER version=3 n_tensors=211 n_kv=19 alignment=32' "$TMP/inv.txt" \
    || fail "the reader's header line is not the recorded one: $(grep -m1 '^HEADER' "$TMP/inv.txt")"

# the 19 metadata keys, in the writer's emission order: the gemma4v contract
cat > "$TMP/keys.exp" <<'EOF'
general.architecture STRING
general.name STRING
general.description STRING
clip.has_vision_encoder BOOL
clip.has_audio_encoder BOOL
clip.projector_type STRING
clip.vision.embedding_length UINT32
clip.vision.feed_forward_length UINT32
clip.vision.block_count UINT32
clip.vision.attention.head_count UINT32
clip.vision.attention.head_count_kv UINT32
clip.vision.attention.head_dim UINT32
clip.vision.attention.layer_norm_epsilon FLOAT32
clip.vision.patch_size UINT32
clip.vision.projection_dim UINT32
clip.vision.projector.scale_factor UINT32
clip.vision.image_size UINT32
clip.vision.image_mean ARRAY
clip.vision.image_std ARRAY
EOF
sed -n 's/^KV //p' "$TMP/inv.txt" > "$TMP/keys.got"
cmp -s "$TMP/keys.exp" "$TMP/keys.got" || {
    echo "the mmproj's metadata keys are not the recorded contract:" >&2
    diff "$TMP/keys.exp" "$TMP/keys.got" | head -10 >&2
    fail "metadata key disagreement"
}

# the per-tensor readout, reduced to name/type/rank/dims/bytes
awk '$1 == "T" {
        name = $2; nd = 0; dims = ""; type = -1; bytes = 0;
        for (i = 3; i <= NF; i++) {
            if ($i ~ /^ndim=/)       nd = substr($i, 6) + 0;
            else if ($i ~ /^dims=/)  dims = substr($i, 6);
            else if ($i ~ /^type=/)  { t = substr($i, 6); sub(/\(.*/, "", t); type = t + 0 }
            else if ($i ~ /^bytes=/) bytes = substr($i, 7) + 0;
        }
        if (type < 0) { print "unparsed type on " name > "/dev/stderr"; exit 1 }
        n = split(dims, d, ",");
        printf "%s\t%d\t%d\t%s\t%s\t%s\t%s\t%d\n", name, type, nd,
               (nd >= 1 ? d[1] : 1), (nd >= 2 ? d[2] : 1),
               (nd >= 3 ? d[3] : 1), (nd >= 4 ? d[4] : 1), bytes;
    }' "$TMP/inv.txt" | sort > "$TMP/ref.tsv"
N_REF=$(wc -l < "$TMP/ref.tsv")
[ "$N_REF" = "211" ] || fail "the reader read $N_REF tensors, the mmproj has 211"

# the writer's manifest, same columns; dims beyond a tensor's rank read as 1
awk -F'\t' 'NR > 1 {
        d1 = ($3 >= 1 ? $4 : 1); d2 = ($3 >= 2 ? $5 : 1);
        d3 = ($3 >= 3 ? $6 : 1); d4 = ($3 >= 4 ? $7 : 1);
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", $1, $2, $3, d1, d2, d3, d4, $8
    }' "$MANIFEST" | sort > "$TMP/man.tsv"
cmp -s "$TMP/man.tsv" "$TMP/ref.tsv" || {
    echo "the writer's manifest and the Rust reader's own dump disagree:" >&2
    diff "$TMP/man.tsv" "$TMP/ref.tsv" | head -10 >&2
    fail "tensor table disagreement"
}

# ---- 4. the frozen reference is present, and its two answers differ ---------
for f in gradient pattern; do
    for kind in soft text; do
        [ -s "$REPO/tests/vision/reference/eg2-vision-$f.$kind.tsv" ] \
            || fail "the frozen reference tests/vision/reference/eg2-vision-$f.$kind.tsv is missing or empty"
    done
done
COS=$(paste <(awk -F'\t' '$1=="l2"{for(i=2;i<=NF;i++)print $i}' "$REPO/tests/vision/reference/eg2-vision-gradient.text.tsv") \
            <(awk -F'\t' '$1=="l2"{for(i=2;i<=NF;i++)print $i}' "$REPO/tests/vision/reference/eg2-vision-pattern.text.tsv") \
      | awk '{s+=$1*$2; a+=$1*$1; b+=$2*$2} END{printf "%.6f", s/sqrt(a*b)}')
awk -v c="$COS" 'BEGIN{ if (c >= 0.999) exit 1 }' \
    || fail "the two frozen fixtures carry the same pooled vector (cosine $COS): the frozen set cannot tell two images apart"

echo "MMPROJ GATE: our own reader read the mmproj back (211/211 tensors agree with the writer's manifest,"
echo "MMPROJ GATE: 19/19 metadata keys are the recorded gemma4v contract), the writer is deterministic"
echo "MMPROJ GATE: (sha256 $GOT), the frozen reference is intact and its two fixtures differ (cosine $COS)."
echo "MMPROJ GATE: the port reproduces those frozen numbers in gate 23; nothing in this path executes the pin."
echo "EG2VQ_VISION_MMPROJ_PASS"
