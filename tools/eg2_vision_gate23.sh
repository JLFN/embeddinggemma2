#!/bin/sh
# eg2_vision_gate23.sh — gate 23: the PORT's own vision tower against the frozen
# reference (phase 8, unit 23, row P4.1).  Native end to end: this script builds
# nothing of the pin, links nothing of the pin, and runs nothing of the pin.
#
# What it checks, in this order:
#
#  1. THE PREPROCESSING, BEFORE ANY MATMUL.  From the port's own oracle
#     (ds4-rs/tests/parity/eg2vq_vision_c_oracle.c, built here):
#       a. the geometry — canvas, region, offset, patch grid, patch and
#          soft-token counts — against the reference's OWN printed facts
#          (out/eg2vq-vision-anchor.log: "clip_encode: copying image ... (nx=528,
#          ny=384)" and "image_tokens->nx = 88" when the log is present, otherwise
#          against the values recorded in tests/vision/reference/preproc-facts.txt,
#          which is committed data extracted from that log);
#       b. the PAD_CEIL padding STRUCTURALLY: which canvas columns and rows are
#          entirely the pad value, expected to be exactly offset_x + (canvas -
#          region - offset_x) columns and, here, no rows — a real image cannot
#          fill a whole 384-pixel column with exactly the pad;
#       c. the two position index vectors, recomputed here independently from
#          clip.cpp:5562-5568's rule (pos_x = i % grid_w, pos_y = i / grid_w).
#
#  2. THE TOWER'S SOFT TOKENS for both frozen fixtures, against
#     tests/vision/reference/eg2-vision-*.soft.tsv, measured against a bound
#     DECLARED AND PRINTED BEFORE the run: max |difference|, relative RMS, and
#     the minimum per-row cosine.  All three are printed even when green, and the
#     bound is a constant in this file that is never widened silently.
#
# What the bound discriminates (measured, see
# docs/eg2vq-vision-tower-gate23.txt): every wrong MECHANISM is 5x to 100x
# further away than the port — tanh GELU instead of gelu_quick 7.6% RMS, the
# position slices swapped 47%, the rope axes swapped 103%, kq_scale = 1/sqrt(d)
# 115%, the weightless V RMS over the whole width 58%, gate/up exchanged 135%,
# and bf16-rounding the conv's activation 3.1% RMS / 0.55 max-abs.  The two
# FINE-GRAINED choices the same measurements settled — the bf16 quantization of
# every bf16 matmul's activation, and gelu_quick's own f16 table round trip —
# are 2x away and are evidenced in that document's discrimination table, not by
# this threshold.
#
# Usage: tools/eg2_vision_gate23.sh
#   Env: EG2_VISION_ORACLE  reuse an oracle build instead of compiling one
#        MMPROJ             the mmproj the weights are bound from
set -eu

ROOT=$(pwd)
MMPROJ=${MMPROJ:-$ROOT/out/EmbeddingGemma-2-914f7f8-vision-mmproj.gguf}
REF=$ROOT/tests/vision/reference
LOG=$ROOT/out/eg2vq-vision-anchor.log
FACTS=$REF/preproc-facts.txt
WORK=${TMPDIR:-/tmp}/eg2-vision-gate23.$$
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

# THE DECLARED BOUND.  Printed before anything runs and never widened.
BOUND_MAXABS=0.30
BOUND_RELRMS=0.025
BOUND_MINROWCOST=0.998

fail() { echo "VISION GATE 23: FAIL: $1" >&2; exit 1; }

[ -f "$MMPROJ" ] || fail "no $MMPROJ (build it with tools/eg2_mmproj_check.sh)"
[ -f "$REF/eg2-vision-pattern.soft.tsv" ] || fail "the frozen dumps are missing from $REF"
[ -f "$FACTS" ] || fail "the reference's recorded preprocessing facts are missing ($FACTS)"
for f in pattern gradient; do
    [ -f "$ROOT/tests/vision/eg2-vision-$f-512x384.png" ] || fail "the $f fixture is missing"
done

# ---- 0. the oracle, our own build, at -O3 --------------------------------
ORACLE=${EG2_VISION_ORACLE:-$WORK/eg2vq_vision_c_oracle}
if [ -z "${EG2_VISION_ORACLE:-}" ]; then
    CC=${CC:-cc}
    # The same flag set the Makefile gives the CPU reference path, at -O3
    # because the tower is 240 GFLOP per image.
    FLAGS="-O3 -ffast-math -march=native -Wall -Wextra -std=c99 -D_GNU_SOURCE
           -fno-finite-math-only -DDS4_NO_GPU -ffunction-sections -fdata-sections
           -Wno-unused-function"
    echo "VISION GATE 23: building the oracle: $CC $FLAGS -I. -o <tmp> tests/parity/eg2vq_vision_c_oracle.c"
    ( cd "$ROOT/ds4-rs" && $CC $FLAGS -I. -o "$ORACLE" \
        tests/parity/eg2vq_vision_c_oracle.c -Wl,--gc-sections -lm ) \
        || fail "the vision oracle did not build"
fi

# ---- 1. the preprocessing, before any matmul ------------------------------
echo "VISION GATE 23: the declared bound (printed before the run):"
echo "VISION GATE 23:   max |difference| <= $BOUND_MAXABS"
echo "VISION GATE 23:   relative RMS    <= $BOUND_RELRMS"
echo "VISION GATE 23:   min row cosine  >= $BOUND_MINROWCOST"

for f in pattern gradient; do
    IMG=tests/vision/eg2-vision-$f-512x384.png
    "$ORACLE" prep "$IMG" > "$WORK/$f.prep.txt" 2> "$WORK/$f.prep.err" \
        || fail "the preprocessing refused $f ($(cat "$WORK/$f.prep.err"))"

    # a. the geometry.  The reference for the check is the committed extraction
    # of the pin's own print (tests/vision/reference/preproc-facts.txt); when the
    # anchor's log is still on disk it is cross-checked against it, so the two
    # can never drift apart silently.
    eval "$(sed -n "s/^REFERENCE_$f //p" "$FACTS")"   # CANVAS, TOKENS, SUMSQ
    SRC="the anchor's recorded facts"
    if [ -f "$LOG" ]; then
        L_CANVAS=$(grep -m1 'clip_encode: copying image .* (nx=[0-9]*, ny=[0-9]*)' "$LOG" \
                   | sed -e 's/.*(nx=\([0-9]*\), ny=\([0-9]*\))/\1x\2/')
        L_TOKENS=$(grep -m1 'image_tokens->nx = [0-9]*' "$LOG" | sed -e 's/.*nx = //')
        if [ -n "$L_CANVAS" ] && [ -n "$L_TOKENS" ]; then
            [ "$L_CANVAS" = "$CANVAS" ] && [ "$L_TOKENS" = "$TOKENS" ] \
                || fail "$f: the anchor log and $FACTS disagree ($L_CANVAS/$L_TOKENS vs $CANVAS/$TOKENS)"
            SRC="the anchor's own log, which agrees with the recorded facts"
        fi
    fi
    PREP=$(head -1 "$WORK/$f.prep.txt")
    GOT_CANVAS=$(echo "$PREP" | sed -e 's/^PREP canvas=\([0-9]*x[0-9]*\).*/\1/')
    GOT_SOFT=$(echo "$PREP" | sed -e 's/.* soft=\([0-9]*\)$/\1/')
    GOT_REGION=$(echo "$PREP" | sed -e 's/.*region=\([0-9]*x[0-9]*\).*/\1/')
    GOT_OFFSET=$(echo "$PREP" | sed -e 's/.*offset=\([-0-9]*,[-0-9]*\).*/\1/')
    GOT_GRID=$(echo "$PREP" | sed -e 's/.*grid=\([0-9]*x[0-9]*\).*/\1/')
    GOT_PATCHES=$(echo "$PREP" | sed -e 's/.*patches=\([0-9]*\).*/\1/')
    [ "$GOT_CANVAS" = "$CANVAS" ] \
        || fail "$f: the port's canvas is $GOT_CANVAS, the reference printed $CANVAS"
    [ "$GOT_SOFT" = "$TOKENS" ] \
        || fail "$f: the port produced $GOT_SOFT soft tokens, the reference printed $TOKENS"
    [ $((GOT_PATCHES / 9)) -eq "$GOT_SOFT" ] \
        || fail "$f: $GOT_PATCHES patches do not pool to $GOT_SOFT tokens"
    echo "VISION GATE 23: $f preprocessing: canvas $GOT_CANVAS (reference $SRC: $CANVAS),"
    echo "VISION GATE 23:   region $GOT_REGION at offset $GOT_OFFSET, grid $GOT_GRID, $GOT_PATCHES patches, $GOT_SOFT soft tokens"

    # b. the padding, structurally.  COLS/ROWS are run-length summaries of "is
    # this whole column/row the pad value?"; the expected pattern follows from
    # the geometry alone.
    CW=$(echo "$GOT_CANVAS" | cut -dx -f1); CH=$(echo "$GOT_CANVAS" | cut -dx -f2)
    RW=$(echo "$GOT_REGION" | cut -dx -f1); RH=$(echo "$GOT_REGION" | cut -dx -f2)
    OX=$(echo "$GOT_OFFSET" | cut -d, -f1); OY=$(echo "$GOT_OFFSET" | cut -d, -f2)
    EXP_COLS=$(awk -v o="$OX" -v r="$RW" -v c="$CW" \
        'BEGIN { l = o; m = r; q = c - r - o; s = "";
                 if (l > 0) s = s " 1*" l; if (m > 0) s = s " 0*" m;
                 if (q > 0) s = s " 1*" q; print substr(s, 2) }')
    EXP_ROWS=$(awk -v o="$OY" -v r="$RH" -v c="$CH" \
        'BEGIN { l = o; m = r; q = c - r - o; s = "";
                 if (l > 0) s = s " 1*" l; if (m > 0) s = s " 0*" m;
                 if (q > 0) s = s " 1*" q; print substr(s, 2) }')
    G_COLS=$(grep '^COLS ' "$WORK/$f.prep.txt" | sed -e 's/^COLS //')
    G_ROWS=$(grep '^ROWS ' "$WORK/$f.prep.txt" | sed -e 's/^ROWS //')
    [ "$G_COLS" = "$EXP_COLS" ] \
        || fail "$f: the padded canvas columns are '$G_COLS', geometry says '$EXP_COLS'"
    [ "$G_ROWS" = "$EXP_ROWS" ] \
        || fail "$f: the padded canvas rows are '$G_ROWS', geometry says '$EXP_ROWS'"
    echo "VISION GATE 23:   padding: columns '$G_COLS' ($OX + $((CW - RW - OX)) pad columns), rows '$G_ROWS'"

    # c. the position index vectors, recomputed here from the rule itself.
    awk -v gw="$(echo "$GOT_GRID" | cut -dx -f1)" -v np="$GOT_PATCHES" '
        /^POS_X / { n = split($0, v, " "); for (i = 2; i <= n; i++) {
                        if (v[i] + 0 != (i - 2) % gw) { print "POS_X mismatch at " (i-2) > "/dev/stderr"; exit 1 } } }
        /^POS_Y / { n = split($0, v, " "); for (i = 2; i <= n; i++) {
                        if (v[i] + 0 != int((i - 2) / gw)) { print "POS_Y mismatch at " (i-2) > "/dev/stderr"; exit 1 } } }
    ' "$WORK/$f.prep.txt" || fail "$f: the position vectors do not follow i % grid_w / i / grid_w"
    echo "VISION GATE 23:   positions: $GOT_PATCHES (pos_x, pos_y) pairs follow clip.cpp:5562-5568"
done

# ---- 2. the tower, against the frozen dumps ------------------------------
for f in pattern gradient; do
    IMG=tests/vision/eg2-vision-$f-512x384.png
    eval "$(sed -n "s/^REFERENCE_$f //p" "$FACTS")"   # SUMSQ, the anchor's own digest
    "$ORACLE" soft "$MMPROJ" "$IMG" > "$WORK/$f.soft.txt" 2> "$WORK/$f.soft.err" \
        || fail "the vision forward refused $f ($(cat "$WORK/$f.soft.err"))"
    grep -q "^SOFT canvas=" "$WORK/$f.soft.txt" || fail "$f: the oracle printed no summary line"
    ROWS=$(grep -c '^[0-9]' "$WORK/$f.soft.txt")
    echo "VISION GATE 23: $f: $(cat "$WORK/$f.soft.err")"

    awk -v label="$f" -v bmax="$BOUND_MAXABS" -v brms="$BOUND_RELRMS" \
        -v bcos="$BOUND_MINROWCOST" -F'\t' '
        FNR == NR { if ($1 ~ /^#/ || NF < 2) next; r = $1; for (i = 2; i <= NF; i++) ref[r " " i] = $i; next }
        { if ($1 ~ /^#/ || /^SOFT/ || NF < 2) next
          r = $1
          if (!( (r " 2") in ref)) { print "row " r " is not in the reference" > "/dev/stderr"; bad = 1; next }
          dot = 0; a2 = 0; b2 = 0
          for (i = 2; i <= NF; i++) {
              v = $i; w = ref[r " " i]
              d = v - w; if (d < 0) d = -d
              if (d > mx) mx = d
              ss += d * d; rr += w * w
              dot += v * w; a2 += v * v; b2 += w * w
          }
          c = (a2 > 0 && b2 > 0) ? dot / sqrt(a2 * b2) : 0
          if (c < minc || seen == 0) minc = c
          seen = 1
          nref++
        }
        END {
            rms = sqrt(ss / rr)
            printf "VISION GATE 23: %s: rows %d, maxabs %.8f (bound %.2f), relrms %.6f (bound %.3f), min row cosine %.8f (bound %.3f)\n",
                   label, nref, mx, bmax, rms, brms, minc, bcos
            if (bad || nref != 88) { print "VISION GATE 23: FAIL: " label " compared " nref " of 88 rows" > "/dev/stderr"; exit 1 }
            if (mx > bmax)  { print "VISION GATE 23: FAIL: " label " maxabs " mx " exceeds " bmax > "/dev/stderr"; exit 1 }
            if (rms > brms) { print "VISION GATE 23: FAIL: " label " relrms " rms " exceeds " brms > "/dev/stderr"; exit 1 }
            if (minc < bcos) { print "VISION GATE 23: FAIL: " label " min row cosine " minc " is below " bcos > "/dev/stderr"; exit 1 }
        }' "$REF/eg2-vision-$f.soft.tsv" "$WORK/$f.soft.txt" \
        || fail "$f: the soft tokens are outside the declared bound"

    # The anchor's own sum of squares for this fixture, as a free cross-check of
    # the overall magnitude (informational: it is the reference's printed value).
    REFSUM=$SUMSQ
    GOTSUM=$(sed -n 's/.*sumsq=\([0-9.]*\)$/\1/p' "$WORK/$f.soft.txt")
    if [ -n "$REFSUM" ] && [ -n "$GOTSUM" ]; then
        printf 'VISION GATE 23: %s: sum of squares %.6f, the anchor printed %s\n' "$f" "$GOTSUM" "$REFSUM"
    fi
done

echo "EG2VQ_VISION_GATE23_PASS"
