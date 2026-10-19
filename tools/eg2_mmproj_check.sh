#!/bin/sh
# eg2_mmproj_check.sh — gate 22: the vision mmproj and the frozen vision oracle
# (phase 8, unit 22; the evidence is docs/eg2vq-vision-oracle-gate22.txt).
#
#   1. both instruments rebuild from source with the commands recorded here;
#   2. the writer is DETERMINISTIC: two writes of the same checkpoint are
#      byte-identical, and the digest equals the recorded one;
#   3. the reference's OWN loader reads the file back — the second side of the
#      check, and the only one that matters, because the file exists to be
#      resolved by it: it must report 211 tensors and every one's name, rank,
#      dims, byte count and type must equal the writer's manifest;
#   4. the frozen reference REPRODUCES: for each fixture the harness's fresh
#      soft-token dump and pooled dump are byte-identical to the committed
#      tests/vision/reference/*;
#   5. the two fixtures are not the same answer (cosine between their pooled
#      vectors below 0.999) — a gate that cannot tell two images apart is not
#      measuring an image path at all.
#
# Usage: tools/eg2_mmproj_check.sh
#   Run from the repository root. Env: HF (checkpoint dir), ANCHOR (the pin's
#   CPU build), MMPROJ_SHA (the recorded digest).
set -eu
REPO=$(pwd)
: "${HF:=$REPO/.models/EmbeddingGemma-2-914f7f8}"
: "${ANCHOR:=/data/llama-anchor-build}"
: "${LLAMA:=/data/llama.cpp}"
: "${YA:=/data/YoungAi}"
: "${MMPROJ_SHA:=9ffb86e3eeb5834095e0bd42b6f5fc0505f02b8e208a6037d01db448e6566838}"
MMPROJ=$REPO/out/EmbeddingGemma-2-914f7f8-vision-mmproj.gguf
MANIFEST=$REPO/out/mmproj-manifest.tsv
BASE=$REPO/out/EmbeddingGemma-2-914f7f8-baseline-bf16.gguf
TMP=$(mktemp -d "${TMPDIR:-/tmp}/eg2-mmproj-check.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$REPO/out"

fail() { echo "MMPROJ GATE: FAIL: $1" >&2; exit 1; }

# ---- 1. rebuild both instruments ------------------------------------------
cc -O3 -std=c99 -Wall -D_POSIX_C_SOURCE=200809L -I"$REPO/tools" -I"$YA/src/common" \
   -o "$REPO/bin/eg2_mmproj" "$REPO/tools/eg2_mmproj.c" -lm || fail "the mmproj writer did not build"
g++ -O2 -std=c++17 -Wno-deprecated-declarations \
    -I"$LLAMA/include" -I"$LLAMA/ggml/include" -I"$LLAMA/tools/mtmd" \
    -o "$REPO/bin/eg2vq_vision_oracle" "$REPO/tests/parity/eg2vq_vision_oracle.cpp" \
    "$ANCHOR/bin/libmtmd.so" "$ANCHOR/bin/libllama.so" "$ANCHOR/bin/libggml.so" \
    "$ANCHOR/bin/libggml-base.so" "$ANCHOR/bin/libggml-cpu.so" -Wl,-rpath,"$ANCHOR/bin" \
    || fail "the vision oracle did not build"

# ---- 2. the writer, twice --------------------------------------------------
"$REPO/bin/eg2_mmproj" "$HF" "$MMPROJ" --manifest "$MANIFEST" > "$TMP/write1.log" || fail "the first write failed"
"$REPO/bin/eg2_mmproj" "$HF" "$TMP/again.gguf" > "$TMP/write2.log" || fail "the second write failed"
cmp -s "$MMPROJ" "$TMP/again.gguf" || fail "two writes of the same checkpoint differ (the writer is not deterministic)"
GOT=$(sha256sum "$MMPROJ" | cut -d' ' -f1)
[ "$GOT" = "$MMPROJ_SHA" ] || fail "the mmproj digest is $GOT, recorded $MMPROJ_SHA"

# ---- 3. the reference's own read of the tensor table ----------------------
"$REPO/bin/eg2vq_vision_oracle" "$MMPROJ" "$BASE" \
    "$REPO/tests/vision/eg2-vision-gradient-512x384.png" "$TMP/gradient" > "$TMP/gradient.log" 2>&1 \
    || fail "the reference refused the mmproj or the run failed (see $TMP/gradient.log)"
grep -q "^\[mmproj\] accepted" "$TMP/gradient.log" || fail "the reference did not report the mmproj as accepted"

# its per-tensor readout, reduced to name/type/rank/dims/bytes
grep '^clip_model_loader: tensor\[' "$TMP/gradient.log" \
  | sed -e 's/.*n_dims = //' -e 's/, name = /\t/' -e 's/, tensor_size=/\t/' \
        -e 's/, offset=[0-9]*//' -e 's/, shape:\[/\t/' -e 's/\], type = /\t/' \
  | awk -F'\t' '{
        nd = $1; name = $2; bytes = $3; dims = $4; type = $5;
        t = (type == "bf16") ? 30 : (type == "f32") ? 0 : (type == "f16") ? 1 : -1;
        if (t < 0) { print "unknown ggml type " type > "/dev/stderr"; exit 1 }
        split(dims, d, ", ");
        printf "%s\t%d\t%d\t%s\t%s\t%s\t%s\t%s\n", name, t, nd, d[1], d[2], d[3], d[4], bytes;
    }' | sort > "$TMP/ref.tsv"
N_REF=$(wc -l < "$TMP/ref.tsv")
[ "$N_REF" = "211" ] || fail "the reference read $N_REF tensors, the mmproj has 211"

# the writer's manifest, same columns. The reference prints ggml's 4-D view, where
# a dim beyond the tensors' rank reads as 1, so the manifest's zero padding is
# normalised to 1 here: the check compares the TENSOR TABLE as a reader sees it.
awk -F'\t' 'NR > 1 {
        d1 = ($3 >= 1 ? $4 : 1); d2 = ($3 >= 2 ? $5 : 1);
        d3 = ($3 >= 3 ? $6 : 1); d4 = ($3 >= 4 ? $7 : 1);
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", $1, $2, $3, d1, d2, d3, d4, $8
    }' "$MANIFEST" | sort > "$TMP/man.tsv"
cmp -s "$TMP/man.tsv" "$TMP/ref.tsv" || {
    echo "the writer's manifest and the reference's own read disagree:" >&2
    diff "$TMP/man.tsv" "$TMP/ref.tsv" | head -10 >&2
    fail "tensor table disagreement"
}

# ---- 4. the frozen reference reproduces -----------------------------------
for f in gradient pattern; do
    "$REPO/bin/eg2vq_vision_oracle" "$MMPROJ" "$BASE" \
        "tests/vision/eg2-vision-$f-512x384.png" "$TMP/$f" > "$TMP/$f.log" 2>&1 \
        || fail "the $f run failed (see the log)"
    # The dumps name the image as it was given on the command line, and the frozen
    # reference was taken with this same relative path from the repository root,
    # so a straight diff IS the check (no normalization to hide a difference).
    cmp -s "$TMP/$f.soft.tsv" "$REPO/tests/vision/reference/eg2-vision-$f.soft.tsv" \
        || fail "the $f soft-token dump differs from the frozen reference"
    cmp -s "$TMP/$f.text.tsv" "$REPO/tests/vision/reference/eg2-vision-$f.text.tsv" \
        || fail "the $f pooled dump differs from the frozen reference"
done

# ---- 5. the two fixtures are different answers ----------------------------
COS=$(paste <(awk -F'\t' '$1=="l2"{for(i=2;i<=NF;i++)print $i}' "$TMP/gradient.text.tsv") \
            <(awk -F'\t' '$1=="l2"{for(i=2;i<=NF;i++)print $i}' "$TMP/pattern.text.tsv") \
      | awk '{s+=$1*$2; a+=$1*$1; b+=$2*$2} END{printf "%.6f", s/sqrt(a*b)}')
awk -v c="$COS" 'BEGIN{ if (c >= 0.999) exit 1 }' \
    || fail "the two fixtures give the same pooled vector (cosine $COS): the image is not reaching the tower"

echo "MMPROJ GATE: the reference accepted the mmproj (211/211 tensors agree with the writer's manifest),"
echo "MMPROJ GATE: the writer is deterministic (sha256 $GOT), the frozen reference reproduces for both"
echo "MMPROJ GATE: fixtures, and the two fixtures differ (cosine $COS)"
echo "EG2VQ_VISION_ORACLE_PASS"
