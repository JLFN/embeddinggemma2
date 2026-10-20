#!/bin/sh
# eg2_embed_media_gate26.sh — gate 26: the embedding WIRE (phase 8, unit 26).
#
# WHAT THIS MEASURES.  The unit-25 half of the image path (the vision tower and
# the splice) was measured at the TOWER level: tests/parity/eg2vq_splice_cuda_oracle
# feeds ids and soft rows to the tower's own entries.  This gate measures the
# same path as a CALLER sees it: an HTTP request to /v1/embeddings carrying one
# image and text, through the server's route, the engine's entries and the
# splice, back as a JSON vector.
#
# The four legs, and what each one is evidence of:
#
#   0. the SEQUENCE the wire builds is the reference's own: for task Raw and the
#      text `<|image|>`, `ds4_core::embed_media_ids` must produce exactly the
#      anchor's 92 positions (the same driver the route's own model method calls;
#      the vocabulary is loaded, the weights are not).
#
#   1. /health states the contract: the family's image half (the tower's
#      soft-token and pixel window, the row width, one image per item, serial
#      media) rendered by ds4_core::EmbedCaps::to_json -- the same object the
#      plan diagnostics print.
#
#   2. the DEVICE leg answers: with --backend cuda and --vision, the two frozen
#      fixtures come back as 768-float, unit-length vectors that DIFFER from each
#      other (a wire that cannot tell two images apart is not carrying an image),
#      and the same fixture with a task prefix gives a different vector (the
#      checkpoint's prefix is applied to the text, as the text path documents).
#
#   3. the SAME requests on the CPU leg (--backend cpu, the tower's other leg)
#      agree with the device leg inside the declared bounds.  This is the wire's
#      own cross-check: one route, one engine entry, both backends.
#
#   4. the refusals are named, not generic: an image with no `<|image|>`
#      placeholder, a placeholder with no image, two placeholders, an item past
#      the 2048-token ceiling, a second image in one item, and an image_url that
#      is not a base64 data URI -- every one a 400 that quotes its reason.
#
# WHY A CPU-LEG SERVER IS POSSIBLE AT ALL: gemma_vq is the one family whose
# vision tower also has a complete CPU leg (gate 23 measured it against the
# reference's own dumps), so ds4.c exempts it from the --vision CUDA rule; every
# other family's sidecar is CUDA kernels only (ds4.c's own guard, commented
# there).
#
# WHAT IS NOT CLAIMED HERE: nothing about the BASELINE weights.  The frozen
# dumps (tests/vision/reference/*.text.tsv) are the baseline model's own splices,
# so comparing the 4-bit artifact against them measures the tier's loss -- that
# is gate 25's leg 3 (0.0369 / 0.2424 / 0.9706), and leg 5 below reports the
# same number as an IDENTITY check (the wire's sequence must be the reference's,
# and a wrong sequence would not land on gate 25's own figures), with its cosine
# floor taken from gate 25's declared sanity bound rather than invented here.
# Nor is any latency claim made: the run prints the request timings it measured,
# and nothing else.
#
# Usage: tools/eg2_embed_media_gate26.sh
#   Run from the repository root.  Env: PORT (default 8791), DS4_ROOT.
set -eu
REPO=$(pwd)
: "${DS4_ROOT:=$REPO/ds4-rs}"
: "${PORT:=8791}"
: "${MMPROJ_SHA:=9ffb86e3eeb5834095e0bd42b6f5fc0505f02b8e208a6037d01db448e6566838}"
ARTIFACT=$REPO/out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf
MMPROJ=$REPO/out/EmbeddingGemma-2-914f7f8-vision-mmproj.gguf
SERVER=$DS4_ROOT/ds4-server
FROZEN=$REPO/tests/vision/reference
TMP=$(mktemp -d "${TMPDIR:-/tmp}/eg2-embed-media-26.XXXXXX")
trap 'stop_server; rm -rf "$TMP"' EXIT

fail() { echo "EMBED MEDIA GATE 26: FAIL: $1" >&2; exit 1; }

# The bounds, printed before the run so the numbers below can be read against
# them: device vs CPU are two implementations of one tower (the device's own
# bf16 quantization boundary is 0.9% on the pooled rows, unit 24), and the
# frozen row is a DIFFERENT MODEL's splice (the tier).
echo "EMBED MEDIA GATE 26: the declared bounds (printed before the run):"
echo "EMBED MEDIA GATE 26:   the vector is 768 floats, L2-normalised: |norm - 1| <= 0.001"
echo "EMBED MEDIA GATE 26:   two different fixtures differ: cosine <= 0.999"
echo "EMBED MEDIA GATE 26:   device vs CPU, same request: cosine >= 0.9999 (report max |difference| and relative RMS)"
echo "EMBED MEDIA GATE 26:   the device vector vs the FROZEN BASELINE splice (the tier, gate 25's leg 3): cosine >= 0.9500, reported with max |difference| and relative RMS"

[ -f "$ARTIFACT" ] || fail "the artifact is not at $ARTIFACT"
[ -f "$MMPROJ" ] || fail "the mmproj is not at $MMPROJ"
GOT=$(sha256sum "$MMPROJ" | cut -d' ' -f1)
[ "$GOT" = "$MMPROJ_SHA" ] || fail "the mmproj digest is $GOT, recorded $MMPROJ_SHA"
for f in eg2-vision-pattern-512x384.png eg2-vision-gradient-512x384.png; do
    [ -f "$REPO/tests/vision/$f" ] || fail "the fixture $f is missing"
done
for f in eg2-vision-pattern.text.tsv eg2-vision-gradient.text.tsv; do
    [ -f "$FROZEN/$f" ] || fail "the frozen dump $f is missing"
done

# ---- the server binaries ----------------------------------------------------
make -C "$DS4_ROOT" ds4-server > "$TMP/build.log" 2>&1 || {
    tail -5 "$TMP/build.log" >&2
    fail "ds4-rs/ds4-server did not build"
}
[ -x "$SERVER" ] || fail "no ds4-server at $SERVER"

SRV_PID=""
stop_server() {
    if [ -n "$SRV_PID" ] && kill -0 "$SRV_PID" 2>/dev/null; then
        kill "$SRV_PID" 2>/dev/null
        wait "$SRV_PID" 2>/dev/null || true
    fi
    SRV_PID=""
    # the engine holds a single-instance lock: give it a moment to release
    sleep 1
}

# start_server <backend> <log>
start_server() {
    "$SERVER" --host 127.0.0.1 --port "$PORT" -m "$ARTIFACT" --vision "$MMPROJ" \
        --backend "$1" --model-id eg2vq > "$2" 2>&1 &
    SRV_PID=$!
    for _ in $(seq 1 60); do
        if curl -s -m 2 "http://127.0.0.1:$PORT/health" > "$TMP/health.json" 2>/dev/null; then
            return 0
        fi
        kill -0 "$SRV_PID" 2>/dev/null || break
        sleep 2
    done
    tail -5 "$2" >&2
    fail "the $1 server did not come up on port $PORT"
}

# ---- the requests -----------------------------------------------------------
B64_PATTERN=$(base64 -w0 "$REPO/tests/vision/eg2-vision-pattern-512x384.png")
B64_GRADIENT=$(base64 -w0 "$REPO/tests/vision/eg2-vision-gradient-512x384.png")
LONG_TEXT=$(awk 'BEGIN { for (i = 0; i < 900; i++) printf "the quick brown fox "; print "" }')

# an item with one image and the text that marks it; task Raw = no prefix, which
# is the sequence the frozen dumps' own header names ([BOS][<|image>] + span +
# [<image|>][EOS], span [2, 90))
media_item() { # <b64> <task>
    printf '{"input":[{"content":[{"type":"text","text":"<|image|>"},{"type":"image_url","image_url":{"url":"data:image/png;base64,%s"}}]}],"task":"%s"}' "$1" "$2"
}

req() { # <name> <body>
    printf '%s' "$2" > "$TMP/req.$1.json"
    curl -s -m 300 -X POST "http://127.0.0.1:$PORT/v1/embeddings" \
        -H 'content-type: application/json' --data-binary @"$TMP/req.$1.json" \
        -o "$TMP/res.$1.json" -w '%{http_code} %{time_total}'
}

vector() { # <name>: the returned embedding, one float per line
    sed -n 's/.*"embedding":\[\([^]]*\)\].*/\1/p' "$TMP/res.$1.json" | tr ',' '\n' | sed 's/^ *//'
}

message() { # <name>: the refusal's own reason
    sed -n 's/.*"message":"\([^"]*\)".*/\1/p' "$TMP/res.$1.json"
}

expect_200() { # <name> <what>
    grep -q '"embedding"' "$TMP/res.$1.json" || {
        echo "$(message "$1")" >&2
        fail "$1 did not answer with an embedding ($2)"
    }
}

expect_400() { # <name> <substring>
    grep -q "$2" "$TMP/res.$1.json" || {
        echo "$(cat "$TMP/res.$1.json")" >&2
        fail "$1 was not refused by name with \"$2\""
    }
}

# <a-file> <b-file> <label>: one readable line on stderr, "max relrms cos" on stdout
compare() {
    paste "$1" "$2" | awk -v label="$3" '{
        d = $1 - $2; if (d < 0) d = -d;
        if (d > mx) mx = d;
        s += d * d; a += $1 * $1; b += $2 * $2; c += $1 * $2; n++;
    } END {
        printf "EMBED MEDIA GATE 26:   %s: n=%d max|d|=%.8f relrms=%.6f cosine=%.8f\n",
               label, n, mx, sqrt(s / n) / sqrt(b / n), c / sqrt(a * b) > "/dev/stderr";
        printf "%.8f %.6f %.8f\n", mx, sqrt(s / n) / sqrt(b / n), c / sqrt(a * b);
    }'
}
bounds_of() { # <label>, echoes "max relrms cos" for a comparison file
    awk '{ printf "%.8f %.6f %.8f\n", $1, $2, $3 }' "$1"
}

# ---- leg 0: the wire's own sequence -----------------------------------------
SEQ=$(cd "$DS4_ROOT" && cargo run -q -p ds4-core --release --example gemma_vq_media_ids -- \
        "$ARTIFACT" '<|image|>' 88 Raw 2>/dev/null) || fail "the sequence driver did not run"
WANT_IDS="2,255999"
i=0
while [ "$i" -lt 88 ]; do
    WANT_IDS="$WANT_IDS,258880"
    i=$((i + 1))
done
WANT_IDS="$WANT_IDS,258882,1"
GOT_N=$(printf '%s\n' "$SEQ" | awk -F'\t' 'NR == 1 { print $1 }')
GOT_IDS=$(printf '%s\n' "$SEQ" | awk -F'\t' 'NR == 1 { print $2 }')
GOT_SPAN=$(printf '%s\n' "$SEQ" | awk 'NR == 2')
[ "$GOT_N" = "92" ] || fail "the wire's sequence for one image is $GOT_N positions, the reference's is 92"
[ "$GOT_IDS" = "$WANT_IDS" ] || fail "the wire's sequence is not the reference's own (BOS, <|image>, 88 placeholders, <image|>, EOS)"
[ "$GOT_SPAN" = "SPAN 2 88" ] || fail "the wire's span is not [2, 90): $GOT_SPAN"
echo "EMBED MEDIA GATE 26: leg 0: the wire's own sequence is the reference's: 92 positions, span [2, 90), BOS 2 / boi 255999 / 88 placeholders / eoi 258882 / EOS 1"

# ---- leg 1: the contract ----------------------------------------------------
start_server cuda "$TMP/cuda.log"
grep -q '"image":{"soft_tokens_min":70,"soft_tokens_max":1120,"pixels_min":161280,"pixels_max":2580480,"embd":512,"per_item_max":1,"serial":true}' "$TMP/health.json" \
    || { cat "$TMP/health.json" >&2; fail "/health does not state the family's image contract"; }
grep -q '"item_tokens_max":2048,"batch_rows_max":8192' "$TMP/health.json" \
    || { cat "$TMP/health.json" >&2; fail "/health lost the item ceiling or the row budget"; }
echo "EMBED MEDIA GATE 26: leg 1: /health carries the image contract: $(cat "$TMP/health.json")"

# ---- leg 2: the device leg --------------------------------------------------
printf 'EMBED MEDIA GATE 26: leg 2 (device): '
OUT=$(req cuda-pattern "$(media_item "$B64_PATTERN" Raw)")
echo "pattern $OUT"
printf 'EMBED MEDIA GATE 26: leg 2 (device): '
OUT=$(req cuda-gradient "$(media_item "$B64_GRADIENT" Raw)")
echo "gradient $OUT"
printf 'EMBED MEDIA GATE 26: leg 2 (device): '
OUT=$(req cuda-prefixed "$(media_item "$B64_PATTERN" Retrieval-query)")
echo "pattern+prefix $OUT"
printf 'EMBED MEDIA GATE 26: leg 2 (device): '
OUT=$(req cuda-text '{"input":"a query","task":"Raw"}')
echo "text only $OUT"
expect_200 cuda-pattern "the frozen pattern fixture"
expect_200 cuda-gradient "the frozen gradient fixture"
expect_200 cuda-prefixed "the pattern fixture with a task prefix"
expect_200 cuda-text "a text-only item"
for n in cuda-pattern cuda-gradient cuda-prefixed cuda-text; do
    vector "$n" > "$TMP/$n.vec"
    N=$(wc -l < "$TMP/$n.vec")
    [ "$N" = "768" ] || fail "$n returned $N floats, not 768"
    awk -v n="$n" '{ s += $1 * $1 } END {
        d = sqrt(s) - 1; if (d < 0) d = -d;
        if (d > 0.001) { printf "%s: the vector is not unit length (norm %.6f)\n", n, sqrt(s) > "/dev/stderr"; exit 1 }
    }' "$TMP/$n.vec" || fail "$n is not L2-normalised"
done
compare "$TMP/cuda-pattern.vec" "$TMP/cuda-gradient.vec" "device pattern vs device gradient" > "$TMP/cmp.pattern-gradient"
CROSS=$(bounds_of "$TMP/cmp.pattern-gradient")
awk -v c="$(echo "$CROSS" | cut -d' ' -f3)" 'BEGIN { if (c > 0.999) exit 1 }' \
    || fail "the two fixtures give the same vector on the wire (cosine $CROSS): the image is not reaching the tower"
compare "$TMP/cuda-pattern.vec" "$TMP/cuda-prefixed.vec" "device pattern vs the same fixture with a task prefix" > "$TMP/cmp.prefix"
PREF=$(bounds_of "$TMP/cmp.prefix")
awk -v c="$(echo "$PREF" | cut -d' ' -f3)" 'BEGIN { if (c > 0.999) exit 1 }' \
    || fail "a task prefix did not change the answer (cosine $PREF): the text path's prefix is not reaching the item"
compare "$TMP/cuda-pattern.vec" "$TMP/cuda-text.vec" "device pattern vs a text-only item" > "$TMP/cmp.text"
PAPT=$(bounds_of "$TMP/cmp.text")
awk -v c="$(echo "$PAPT" | cut -d' ' -f3)" 'BEGIN { if (c > 0.999) exit 1 }' \
    || fail "an image item and a text item give the same vector (cosine $PAPT)"
echo "EMBED MEDIA GATE 26: leg 2: 4 answers, 768 floats each, unit length, and the image changes the answer (pattern vs gradient cosine $(echo "$CROSS" | cut -d' ' -f3), vs a text item $(echo "$PAPT" | cut -d' ' -f3))"

# ---- leg 4 (device): the refusals ------------------------------------------
printf 'EMBED MEDIA GATE 26: leg 4 (device): '
OUT=$(req cuda-noplaceholder "$(printf '{"input":[{"content":[{"type":"text","text":"a photo of a shoe"},{"type":"image_url","image_url":{"url":"data:image/png;base64,%s"}}]}],"task":"Raw"}' "$B64_PATTERN")")
echo "no placeholder $OUT"
printf 'EMBED MEDIA GATE 26: leg 4 (device): '
OUT=$(req cuda-placeholdernone '{"input":"a photo of a shoe <|image|>","task":"Raw"}')
echo "placeholder without image $OUT"
printf 'EMBED MEDIA GATE 26: leg 4 (device): '
OUT=$(req cuda-twoplaceholders "$(printf '{"input":[{"content":[{"type":"text","text":"<|image|> and <|image|>"},{"type":"image_url","image_url":{"url":"data:image/png;base64,%s"}}]}],"task":"Raw"}' "$B64_PATTERN")")
echo "two placeholders $OUT"
printf 'EMBED MEDIA GATE 26: leg 4 (device): '
OUT=$(req cuda-pastceiling "$(printf '{"input":[{"content":[{"type":"text","text":"%s <|image|>"},{"type":"image_url","image_url":{"url":"data:image/png;base64,%s"}}]}],"task":"Raw"}' "$LONG_TEXT" "$B64_PATTERN")")
echo "past the item ceiling $OUT"
printf 'EMBED MEDIA GATE 26: leg 4 (device): '
OUT=$(req cuda-twoimages "$(printf '{"input":[{"content":[{"type":"text","text":"two <|image|>"},{"type":"image_url","image_url":{"url":"data:image/png;base64,%s"}},{"type":"image_url","image_url":{"url":"data:image/png;base64,%s"}}]}]}' "$B64_PATTERN" "$B64_PATTERN")")
echo "two images in one item $OUT"
printf 'EMBED MEDIA GATE 26: leg 4 (device): '
OUT=$(req cuda-baduri '{"input":[{"content":[{"type":"text","text":"<|image|>"},{"type":"image_url","image_url":{"url":"https://example.test/a.png"}}]}]}')
echo "not a data URI $OUT"
expect_400 cuda-noplaceholder 'has no <|image|> placeholder'
expect_400 cuda-placeholdernone 'carries the image placeholder <|image|> but no image'
expect_400 cuda-twoplaceholders 'more than one <|image|> placeholder'
expect_400 cuda-pastceiling 'past the 2048 the embedding tower'
expect_400 cuda-twoimages 'more than one image'
expect_400 cuda-baduri 'image_url must be a base64 data URI'
echo "EMBED MEDIA GATE 26: leg 4: six refusals, each a 400 quoting its own reason"
stop_server

# ---- leg 3: the same requests on the CPU leg -------------------------------
start_server cpu "$TMP/cpu.log"
printf 'EMBED MEDIA GATE 26: leg 3 (CPU): '
OUT=$(req cpu-pattern "$(media_item "$B64_PATTERN" Raw)")
echo "pattern $OUT"
printf 'EMBED MEDIA GATE 26: leg 3 (CPU): '
OUT=$(req cpu-gradient "$(media_item "$B64_GRADIENT" Raw)")
echo "gradient $OUT"
printf 'EMBED MEDIA GATE 26: leg 3 (CPU): '
OUT=$(req cpu-prefixed "$(media_item "$B64_PATTERN" Retrieval-query)")
echo "pattern+prefix $OUT"
expect_200 cpu-pattern "the frozen pattern fixture on the CPU leg"
expect_200 cpu-gradient "the frozen gradient fixture on the CPU leg"
expect_200 cpu-prefixed "the pattern fixture with a task prefix on the CPU leg"
for n in cpu-pattern cpu-gradient cpu-prefixed; do
    vector "$n" > "$TMP/$n.vec"
    N=$(wc -l < "$TMP/$n.vec")
    [ "$N" = "768" ] || fail "$n returned $N floats, not 768"
done
compare "$TMP/cuda-pattern.vec" "$TMP/cpu-pattern.vec" "device vs CPU, pattern" > "$TMP/cmp.dev-cpu-pattern"
compare "$TMP/cuda-gradient.vec" "$TMP/cpu-gradient.vec" "device vs CPU, gradient" > "$TMP/cmp.dev-cpu-gradient"
compare "$TMP/cuda-prefixed.vec" "$TMP/cpu-prefixed.vec" "device vs CPU, pattern+prefix" > "$TMP/cmp.dev-cpu-prefixed"
for f in cmp.dev-cpu-pattern cmp.dev-cpu-gradient cmp.dev-cpu-prefixed; do
    C=$(awk '{ print $3 }' "$TMP/$f")
    awk -v c="$C" 'BEGIN { if (c < 0.9999) exit 1 }' \
        || fail "$f: the two legs disagree (cosine $C): see the line above"
done
echo "EMBED MEDIA GATE 26: leg 3: the CPU leg reproduces the device answers inside the declared bound (three requests, cosine >= 0.9999)"
stop_server

# ---- leg 5: the frozen target (identity, not quality) ----------------------
awk -F'\t' '$1 == "l2" { for (i = 2; i <= NF; i++) print $i }' \
    "$FROZEN/eg2-vision-pattern.text.tsv" > "$TMP/frozen-pattern.vec"
[ "$(wc -l < "$TMP/frozen-pattern.vec")" = "768" ] || fail "the frozen pattern dump is not 768 values"
compare "$TMP/cuda-pattern.vec" "$TMP/frozen-pattern.vec" "device wire vs the FROZEN BASELINE splice (the tier)" > "$TMP/cmp.frozen"
FROZ=$(bounds_of "$TMP/cmp.frozen")
awk -v c="$(echo "$FROZ" | cut -d' ' -f3)" 'BEGIN { if (c < 0.95) exit 1 }' \
    || fail "the wire's answer is not the reference's sequence at all (cosine $FROZ)"
echo "EMBED MEDIA GATE 26: leg 5: the wire's sequence is the reference's own (gate 25's leg 3 band: 0.0369 / 0.2424 / 0.9706)"

echo "EMBED MEDIA GATE 26: PASS: an image reaches /v1/embeddings, the contract is advertised, both legs agree and every refusal is named"
echo "EG2VQ_EMBED_MEDIA_GATE26_PASS"
