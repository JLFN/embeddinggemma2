#!/bin/sh
# eg2_splice_gate25.sh — gate 25: the IMAGE SPLICE end to end (phase 8, unit 25).
# Native end to end: this script builds nothing of the pin and runs nothing of it.
#
# What it checks, in this order:
#
#  1. THE SEQUENCE, against the frozen dumps' own header.  The reference spliced
#     ONE llama_process of 92 positions with the image span at [2, 90) — its own
#     comment says so — and the port must build the same sequence: [BOS]
#     [<|image>] + the image's soft-token positions + [<image|>][EOS], with the
#     four text ids taken from the artifact's own vocabulary (bos 2, boi 255999,
#     eoi 258882, eos 1) and the span's placeholder from the checkpoint's
#     image_token_id (258880).  The oracle builds and prints it.
#
#  2. THE SPLICE ITSELF, both legs, against the frozen pooled rows
#     (tests/vision/reference/eg2-vision-*.text.tsv), with the bound declared and
#     printed BEFORE the run and never widened:
#       cpu  vs the frozen "l2" row: max |difference| <= 0.010, relative RMS <=
#            0.020, cosine >= 0.999
#       cpu  vs the frozen "raw" row: DIRECTION only (it is pre-L2), cosine >= 0.999
#       device vs cpu (the port's own parity): max |difference| <= 0.005,
#            relative RMS <= 0.005, cosine >= 0.9999
#     plus the text-only control (the same ids without the image must NOT give the
#     same vector) and the device entry's four refusals.
#
# Usage: tools/eg2_splice_gate25.sh
#   Env: EG2_SPLICE_ORACLE  reuse a built oracle instead of building one
#        EG2_ARTIFACT       the quantized text artifact
#        MMPROJ             the companion vision mmproj
set -eu

ROOT=$(pwd)
ART=${EG2_ARTIFACT:-$ROOT/out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf}
MMPROJ=${MMPROJ:-$ROOT/out/EmbeddingGemma-2-914f7f8-vision-mmproj.gguf}
BASE=${EG2_BASELINE:-$ROOT/out/EmbeddingGemma-2-914f7f8-baseline-bf16.gguf}
REF=$ROOT/tests/vision/reference
WORK=${TMPDIR:-/tmp}/eg2-splice-gate25.$$
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

fail() { echo "SPLICE GATE 25: FAIL: $1" >&2; exit 1; }

[ -f "$ART" ] || fail "no $ART"
[ -f "$MMPROJ" ] || fail "no $MMPROJ"
[ -f "$BASE" ] || fail "no $BASE (the baseline text model the frozen spliced rows were produced with)"
for f in pattern gradient; do
    [ -f "$ROOT/tests/vision/eg2-vision-$f-512x384.png" ] || fail "the $f fixture is missing"
    [ -f "$REF/eg2-vision-$f.text.tsv" ] || fail "the frozen $f splice dump is missing from $REF"
    [ -f "$REF/eg2-vision-$f.soft.tsv" ] || fail "the frozen $f soft dump is missing from $REF"
done

echo "SPLICE GATE 25: the declared bound (printed before the run):"
echo "SPLICE GATE 25:   THE SPLICE, baseline weights + the reference's own soft rows:"
echo "SPLICE GATE 25:     max |difference| <= 0.005, relative RMS <= 0.010, cosine >= 0.999"
echo "SPLICE GATE 25:     and vs the frozen raw row, direction only: cosine >= 0.999"
echo "SPLICE GATE 25:   device (artifact weights) vs the artifact's own cpu leg:"
echo "SPLICE GATE 25:     max |difference| <= 0.005, relative RMS <= 0.005, cosine >= 0.9999"
echo "SPLICE GATE 25:   the baseline+port-soft and the artifact legs are REPORTED (the vision tower's"
echo "SPLICE GATE 25:   0.82% and the 4-bit tier's own loss), with cosine >= 0.900 as their sanity floor"

ORACLE=${EG2_SPLICE_ORACLE:-$ROOT/ds4-rs/tests/parity/eg2vq_splice_cuda_oracle}
if [ -z "${EG2_SPLICE_ORACLE:-}" ]; then
    echo "SPLICE GATE 25: building the oracle (make tests/parity/eg2vq_splice_cuda_oracle)"
    make -C "$ROOT/ds4-rs" tests/parity/eg2vq_splice_cuda_oracle > "$WORK/build.log" 2>&1 \
        || { tail -25 "$WORK/build.log" >&2; fail "the splice oracle did not build"; }
fi
[ -x "$ORACLE" ] || fail "no oracle at $ORACLE"

"$ORACLE" "$ART" "$MMPROJ" "$BASE" \
    "$ROOT/tests/vision/eg2-vision-pattern-512x384.png" \
    "$ROOT/tests/vision/eg2-vision-gradient-512x384.png" > "$WORK/run.txt" 2> "$WORK/run.err" \
    || { cat "$WORK/run.err" >&2; cat "$WORK/run.txt" >&2; fail "the oracle reported a deviation outside the declared bound"; }
grep -q '^EG2VQ_SPLICE_PASS$' "$WORK/run.txt" || fail "the oracle did not report EG2VQ_SPLICE_PASS"

# The sequence, against the frozen dumps' own header (92 positions, span [2, 90)).
for f in pattern gradient; do
    grep -q "SPLICE ORACLE: eg2-vision-$f: 88 soft tokens, sequence 92 positions, span \[2, 90)" "$WORK/run.txt" \
        || fail "$f: the port's sequence is not the reference's 92 positions with the span at [2, 90)"
done

grep -E '^SPLICE ORACLE: ' "$WORK/run.txt" | sed -e 's/^SPLICE ORACLE: /SPLICE GATE 25: /'

echo "EG2VQ_SPLICE_GATE25_PASS"
