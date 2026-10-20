#!/bin/sh
# eg2_vision_anchor.sh — the pin's OWN view of the vision path, opt-in.
#
# WHAT THIS IS, AND WHAT IT IS NOT (owner directive 2026-10-20, "We will not use
# llama it must be ported all"):
#
#   This is the ONLY script in this repository that executes the pinned
#   llama.cpp.  It is a REFERENCE-REGENERATION instrument: it rebuilds the anchor
#   harness, feeds it the mmproj our writer produced, and re-derives
#   tests/vision/reference/*.tsv.  It is never called by tests/run.sh, never by
#   tools/eg2_mmproj_check.sh (gate 22), and never by the port's build or run.
#
#   The port does not link, invoke or call any llama code: its own reader reads
#   the mmproj (gate 22) and, from unit 23 on, its own tower reproduces these
#   frozen numbers (gate 23).  The numbers themselves are committed data, frozen
#   ONCE on 2026-10-19, and they are what the port is measured against — the same
#   role the frozen text anchor plays in tests/expected/.  Run this only when the
#   reference has to be re-derived: a new fixture, a changed preprocessing, a new
#   pin revision.
#
#   1. rebuilds the anchor harness against the pin's CPU build;
#   2. the pin's own clip loader must ACCEPT the mmproj, and its per-tensor
#      readout must equal our writer's manifest (211 tensors);
#   3. for each fixture the harness's fresh soft-token and pooled dumps must be
#      byte-identical to the committed tests/vision/reference/*;
#   4. the two fixtures must not be the same answer (cosine below 0.999).
#
# Usage: tools/eg2_vision_anchor.sh
#   Env: HF (checkpoint dir), ANCHOR (the pin's CPU build), LLAMA (the pin),
#        MMPROJ_SHA (the recorded digest).
set -eu
REPO=$(pwd)
: "${HF:=$REPO/.models/EmbeddingGemma-2-914f7f8}"
: "${ANCHOR:=/data/llama-anchor-build}"
: "${LLAMA:=/data/llama.cpp}"
: "${MMPROJ_SHA:=9ffb86e3eeb5834095e0bd42b6f5fc0505f02b8e208a6037d01db448e6566838}"
MMPROJ=$REPO/out/EmbeddingGemma-2-914f7f8-vision-mmproj.gguf
MANIFEST=$REPO/out/mmproj-manifest.tsv
BASE=$REPO/out/EmbeddingGemma-2-914f7f8-baseline-bf16.gguf
TMP=$(mktemp -d "${TMPDIR:-/tmp}/eg2-vision-anchor.XXXXXX")
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$REPO/out"

fail() { echo "VISION ANCHOR: FAIL: $1" >&2; exit 1; }

[ -d "$ANCHOR/bin" ] || fail "the pin's CPU build is not at $ANCHOR (set ANCHOR); this script is opt-in and needs it"
[ -f "$LLAMA/tools/mtmd/mtmd-image.cpp" ] || fail "the pin is not at $LLAMA (set LLAMA)"

# ---- 1. the harness rebuilds ------------------------------------------------
g++ -O2 -std=c++17 -Wno-deprecated-declarations \
    -I"$LLAMA/include" -I"$LLAMA/ggml/include" -I"$LLAMA/tools/mtmd" \
    -o "$REPO/bin/eg2vq_vision_anchor" "$REPO/tests/parity/eg2vq_vision_anchor.cpp" \
    "$ANCHOR/bin/libmtmd.so" "$ANCHOR/bin/libllama.so" "$ANCHOR/bin/libggml.so" \
    "$ANCHOR/bin/libggml-base.so" "$ANCHOR/bin/libggml-cpu.so" -Wl,-rpath,"$ANCHOR/bin" \
    || fail "the anchor harness did not build"

# it runs the mmproj the writer produced: if that file is stale, the numbers
# below are the ones it would still reproduce, so the writer runs first.
"$REPO/bin/eg2_mmproj" "$HF" "$MMPROJ" --manifest "$MANIFEST" > "$TMP/write.log" \
    || fail "the mmproj writer failed"
GOT=$(sha256sum "$MMPROJ" | cut -d' ' -f1)
[ "$GOT" = "$MMPROJ_SHA" ] || fail "the mmproj digest is $GOT, recorded $MMPROJ_SHA"

# ---- 2. the pin's own read of the tensor table ------------------------------
"$REPO/bin/eg2vq_vision_anchor" "$MMPROJ" "$BASE" \
    "$REPO/tests/vision/eg2-vision-gradient-512x384.png" "$TMP/gradient" > "$TMP/gradient.log" 2>&1 \
    || fail "the pin refused the mmproj or the run failed (see $TMP/gradient.log)"
grep -q "^\[mmproj\] accepted" "$TMP/gradient.log" || fail "the pin did not report the mmproj as accepted"

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
[ "$N_REF" = "211" ] || fail "the pin read $N_REF tensors, the mmproj has 211"

# the writer's manifest, same columns (dims beyond rank read as 1)
awk -F'\t' 'NR > 1 {
        d1 = ($3 >= 1 ? $4 : 1); d2 = ($3 >= 2 ? $5 : 1);
        d3 = ($3 >= 3 ? $6 : 1); d4 = ($3 >= 4 ? $7 : 1);
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n", $1, $2, $3, d1, d2, d3, d4, $8
    }' "$MANIFEST" | sort > "$TMP/man.tsv"
cmp -s "$TMP/man.tsv" "$TMP/ref.tsv" || {
    echo "the writer's manifest and the pin's own read disagree:" >&2
    diff "$TMP/man.tsv" "$TMP/ref.tsv" | head -10 >&2
    fail "tensor table disagreement"
}

# ---- 3. the frozen reference reproduces ------------------------------------
for f in gradient pattern; do
    "$REPO/bin/eg2vq_vision_anchor" "$MMPROJ" "$BASE" \
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
# the pin's own per-tensor readout, hparams dump and preprocessing facts, kept
# whole under out/ (gitignored): the anchor is the only thing that can print them
cat "$TMP/gradient.log" "$TMP/pattern.log" > "$REPO/out/eg2vq-vision-anchor.log"

# ---- 4. the two fixtures are different answers -----------------------------
COS=$(paste <(awk -F'\t' '$1=="l2"{for(i=2;i<=NF;i++)print $i}' "$TMP/gradient.text.tsv") \
            <(awk -F'\t' '$1=="l2"{for(i=2;i<=NF;i++)print $i}' "$TMP/pattern.text.tsv") \
      | awk '{s+=$1*$2; a+=$1*$1; b+=$2*$2} END{printf "%.6f", s/sqrt(a*b)}')
awk -v c="$COS" 'BEGIN{ if (c >= 0.999) exit 1 }' \
    || fail "the two fixtures give the same pooled vector (cosine $COS): the image is not reaching the tower"

echo "VISION ANCHOR: the pin accepted the mmproj (211/211 tensors agree with the writer's manifest),"
echo "VISION ANCHOR: the frozen reference reproduces byte-identically for both fixtures, and the two"
echo "VISION ANCHOR: fixtures differ (cosine $COS).  This script is opt-in; nothing else runs the pin."
