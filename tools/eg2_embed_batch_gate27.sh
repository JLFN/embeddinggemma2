#!/bin/sh
# eg2_embed_batch_gate27.sh — gate 27: SEVERAL items in ONE tower call (unit 27,
# phase-doc row P2.3's last slice).
#
# WHAT THIS MEASURES.  Until unit 27 an array request paid one tower pass per
# item: N items, N host round trips, N waves of kernel launches, N readbacks.
# The unit packs the items into one contiguous row block (`ds4_eg2vq_gpu.cuh`'s
# core, `ds4_gpu_eg2vq_embed_batch` / `ds4_eg2vq_ref_embed_batch`) and the route
# hands its text items to that one call.  The claim is narrow and exact: a
# batched item's vector EQUALS the vector the one-item entry returns for it,
# because every row's arithmetic depends only on its own item (its own position
# inside it, its own attention window and its own pool).
#
# The legs, and what each one is evidence of:
#
#   0. the bounds, printed BEFORE the run: the equality floor is BYTE identity
#      (not a tolerance), the row budget's arithmetic, and the row-cost figures
#      the layout charges.  Nothing below is tuned after a measurement.
#
#   1. the DEVICE ORACLE, one call against the per-item calls over a mixed-length
#      set (3 of SciFact's longest pool documents and its 3 shortest queries,
#      ~1414 down to ~14 tokens): every element byte-identical, and what one call
#      costs against the per-item sum.  The same run prints both workspaces: the
#      per-item context at 2048 rows (the ceiling's size class) and the packed
#      3373-row one, which must be the layout's own arithmetic.
#
#   2. the MANY-SMALL-ITEMS case, which is the shape batching exists for: 64
#      short items, one call against 64.  (Leg 1's six items are large enough that
#      the tower is compute-bound and the saving is small; a throughput claim made
#      only from leg 1 would be a claim about the wrong condition.)
#
#   3. the ROW BUDGET's boundary at the device entry: a synthetic call of exactly
#      8192 rows is ACCEPTED (and prints its 912.13 MiB workspace), 8193 rows is
#      REFUSED by name.  And the block's own description must be believed: an
#      empty item, a span table that does not cover the rows, zero items and an
#      item past the 2048-token key plane are all refused with the caller's buffer
#      untouched.
#
#   4. the WIRE, on the device: an array request's embeddings are BYTE-IDENTICAL
#      to the same items sent one per request (the batch is not a different
#      answer), and the advertised row budget holds at the wire: four items of
#      exactly 2048 tokens (8192 rows) are one accepted call, and one item more is
#      a 400 that quotes `embed.batch_rows_max`.
#
#   5. the WIRE, on the CPU leg: the same array-vs-singles comparison through
#      `ds4_eg2vq_ref_embed_batch` (which loops the gated one-item reference, by
#      design: the CPU leg is this port's oracle, not its speed path).
#
#   6. /health still advertises the two numbers the batch is bounded by
#      (`item_tokens_max` 2048, `batch_rows_max` 8192): the contract a caller reads
#      before sending an array.
#
# WHAT IS NOT CLAIMED HERE: no serving latency, no concurrency, no items/s for
# the server -- the two throughput numbers below are wall-clock for the calls
# named, on the set named, in this process.  Nothing about the vision path: one
# device vision workspace per engine is serial by design, so a media item keeps
# its own call, and gate 26 remains its measurement.
#
# Usage: tools/eg2_embed_batch_gate27.sh
#   Run from the repository root.  Env: PORT (default 8793), DS4_ROOT.
set -eu
REPO=$(pwd)
: "${DS4_ROOT:=$REPO/ds4-rs}"
: "${PORT:=8793}"
ARTIFACT=$REPO/out/EmbeddingGemma-2-914f7f8-vq4x4096-q80token-gemma_vq.gguf
CONFIG=$REPO/.models/EmbeddingGemma-2-914f7f8/config_sentence_transformers.json
SERVER=$DS4_ROOT/ds4-server
ORACLE=$DS4_ROOT/tests/parity/eg2vq_embd_gpu_oracle
POOL=$REPO/tests/retrieval/scifact-test-pool.tsv
QUERIES=$REPO/tests/retrieval/scifact-test-queries.tsv
TMP=$(mktemp -d "${TMPDIR:-/tmp}/eg2-embed-batch-27.XXXXXX")

# The server handle, defined before the trap that uses it: a failure anywhere
# below (including in the fixture or build steps) must still not leave a server
# holding the engine's single-instance lock.
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
trap 'stop_server; rm -rf "$TMP"' EXIT

fail() { echo "EMBED BATCH GATE 27: FAIL: $1" >&2; exit 1; }

# The workspace's own cost model, evaluated here from the rows a run reports
# (ds4_eg2vq_gpu.cuh's layout: 116,748 B a row, the budget rounded up in 256-row
# steps, plus the item spans rounded to the layout's 16-byte alignment).  Every
# MiB figure below is checked against this, so a layout change moves both sides
# instead of silently disagreeing with a constant in this script.
expect_mib() { # <rows> [decimal places]
    awk -v r="$1" -v dp="${2:-2}" 'BEGIN {
        want = int((r + 255) / 256) * 256;
        spans = int((int((want + 1) * 4) + 15) / 16) * 16;
        printf "%." dp "f", (want * 116748 + spans) / 1048576
    }'
}
# the rows and the packed workspace of a run's own "call:" line
rows_of() { sed -n 's/^call: *[0-9]* items, \([0-9]*\) rows,.*/\1/p' "$1"; }
packed_mib_of() { sed -n 's/^call: .*one workspace of \([0-9.]*\) MiB.*/\1/p' "$1"; }
# the per-item context's size class and workspace, from the oracle's own summary
peritem_rows_of() { sed -n 's/^device: *max_tokens \([0-9]*\), workspace.*/\1/p' "$1"; }
peritem_mib_of() { sed -n 's/^device: *max_tokens [0-9]*, workspace \([0-9.]*\) MiB.*/\1/p' "$1"; }

# The bounds, printed before the run so the numbers below read against them.
echo "EMBED BATCH GATE 27: the declared bounds (printed before the run):"
echo "EMBED BATCH GATE 27:   a batched item's vector: BYTE-IDENTICAL to its single-item vector"
echo "EMBED BATCH GATE 27:   the row budget: 8192 rows accepted, 8193 refused, one workspace"
echo "EMBED BATCH GATE 27:   the row cost the layout charges: 116,748 B/row + the item spans"
echo "EMBED BATCH GATE 27:     -> 228.03 MiB at 2048 rows, 912.13 MiB at the 8192-row budget (both checked below)"
echo "EMBED BATCH GATE 27:   device vs CPU on the wire's own items: byte-identical (both legs run the same arithmetic)"

[ -f "$ARTIFACT" ] || fail "the artifact is not at $ARTIFACT"
[ -f "$CONFIG" ] || fail "the checkpoint's prompt map is not at $CONFIG"
[ -f "$POOL" ] || fail "the committed pool $POOL is missing"
[ -f "$QUERIES" ] || fail "the committed queries $QUERIES are missing"

# ---- the fixtures -----------------------------------------------------------
# Three of the longest pool documents and the three shortest queries: the lengths
# that make the equality check meaningful (a batch whose items are all one size
# would not catch a key range that leaked across items).  The files are already
# in the tokenizer driver's own items shape (id <TAB> prompt_name <TAB> text).
mix_items() {
    awk -F'\t' '!/^#/ && NF >= 3 { print length($3) "\t" $0 }' "$POOL" |
        sort -rn | head -3 | cut -f2-
    awk -F'\t' '!/^#/ && NF >= 3 { print length($3) "\t" $0 }' "$QUERIES" |
        sort -n | head -3 | cut -f2-
}
mix_items > "$TMP/mix.tsv"
awk -F'\t' '!/^#/ && NF >= 3 { print length($3) "\t" $0 }' "$QUERIES" |
    sort -n | head -64 | cut -f2- > "$TMP/small.tsv"
[ "$(grep -c . "$TMP/mix.tsv")" = 6 ] || fail "the mixed fixture is not 6 items"
[ "$(grep -c . "$TMP/small.tsv")" = 64 ] || fail "the small-items fixture is not 64 items"

# ---- the binaries -----------------------------------------------------------
make -C "$DS4_ROOT" ds4-server tests/parity/eg2vq_embd_gpu_oracle > "$TMP/build.log" 2>&1 || {
    tail -5 "$TMP/build.log" >&2
    fail "ds4-rs did not build"
}
[ -x "$SERVER" ] || fail "no ds4-server at $SERVER"
[ -x "$ORACLE" ] || fail "no device oracle at $ORACLE"

# The ids dumps come from the Rust host's own tokenizer (the same example gate 25
# and phase 4 use), so the gate's fixtures are reproducible from committed data.
tok() { # <items.tsv> <out.ids.tsv>
    ( cd "$DS4_ROOT" && cargo run -q -p ds4-core --release --example gemma_vq_tok -- \
        "$ARTIFACT" "$1" "$CONFIG" ) > "$2" 2> "$TMP/tok.err" || {
        tail -5 "$TMP/tok.err" >&2
        fail "the tokenizer example failed on $1"
    }
    grep -q '^TOKENS ' "$2" || fail "the tokenizer dump has no TOKENS summary"
}
tok "$TMP/mix.tsv" "$TMP/mix.ids.tsv"
tok "$TMP/small.tsv" "$TMP/small.ids.tsv"
MIX_OK=$(grep '^TOKENS ' "$TMP/mix.ids.tsv" | tr -d '\r')
SMALL_OK=$(grep '^TOKENS ' "$TMP/small.ids.tsv" | tr -d '\r')
echo "EMBED BATCH GATE 27: the mixed set:  $MIX_OK"
echo "EMBED BATCH GATE 27: the small set:  $SMALL_OK"

# ---- leg 1: the device oracle, one call against the per-item calls ----------
run_oracle() { # <name> <ids> [extra...]
    name=$1; ids=$2; shift 2
    if "$ORACLE" "$ARTIFACT" "$ids" --no-cpu "$@" > "$TMP/or.$name.log" 2>&1; then
        echo 0
    else
        echo $?
    fi
}
# the lines the legs read back, as evidence
oracle_summary() { # <log>
    sed -n -e '/^call: /p' -e '/^equality: /p' -e '/^per-item/p' -e '/^one call/p' \
        -e '/^ratio: /p' -e '/^accepted: /p' -e '/^refused: /p' "$1" | sed 's/^/  /'
}

MIX_EXIT=$(run_oracle mix "$TMP/mix.ids.tsv" --batch --batch-bad)
[ "$MIX_EXIT" = 0 ] || { tail -12 "$TMP/or.mix.log" >&2; fail "the mixed-set device run failed (exit $MIX_EXIT)"; }
grep -q "equality: every element byte-identical to the per-item run, max per-element delta 0.000e+00" \
    "$TMP/or.mix.log" || { tail -12 "$TMP/or.mix.log" >&2; fail "the batch is not byte-identical to the per-item runs"; }
# the two workspaces the run allocated, against the layout's own cost model
MIX_ROWS=$(rows_of "$TMP/or.mix.log")
MIX_MIB=$(packed_mib_of "$TMP/or.mix.log")
MIX_WANT=$(expect_mib "$MIX_ROWS")
[ -n "$MIX_ROWS" ] && [ -n "$MIX_MIB" ] || fail "the run did not report its packed call"
[ "$MIX_MIB" = "$MIX_WANT" ] || fail "the packed $MIX_ROWS-row workspace is $MIX_MIB MiB; the layout's cost model says $MIX_WANT MiB"
PI_ROWS=$(peritem_rows_of "$TMP/or.mix.log")
PI_MIB=$(peritem_mib_of "$TMP/or.mix.log")
[ "$PI_ROWS" = 2048 ] || fail "the per-item context's size class is $PI_ROWS rows, not the 2048 ceiling"
PI_WANT=$(expect_mib 2048 1)
[ "$PI_MIB" = "$PI_WANT" ] || fail "the per-item context at 2048 rows is $PI_MIB MiB; the cost model says $PI_WANT MiB"
# the refusals: five named cases, each leaving the caller's buffer alone
if [ "$(grep -c "refused, output untouched" "$TMP/or.mix.log")" != 5 ]; then
    grep -E "refused|ACCEPTED" "$TMP/or.mix.log" | sed 's/^/  /' >&2
    fail "the five malformed blocks were not all refused with the buffer untouched"
fi
if grep -q "ACCEPTED (BUG)" "$TMP/or.mix.log"; then
    fail "a malformed span table was accepted"
fi
echo "EMBED BATCH GATE 27: leg 1: one call over the mixed set ($MIX_ROWS rows)"
oracle_summary "$TMP/or.mix.log"
echo "  per-item context: $PI_ROWS rows, $PI_MIB MiB (the cost model's $PI_WANT); packed: $MIX_ROWS rows, $MIX_MIB MiB (the cost model's $MIX_WANT)"
sed -n '/the batch.s refusals/,/^$/p' "$TMP/or.mix.log" | sed 's/^/  /'

# ---- leg 2: 64 small items, the shape batching exists for -------------------
SMALL_EXIT=$(run_oracle small "$TMP/small.ids.tsv" --batch)
[ "$SMALL_EXIT" = 0 ] || { tail -12 "$TMP/or.small.log" >&2; fail "the 64-item device run failed (exit $SMALL_EXIT)"; }
grep -q "equality: every element byte-identical to the per-item run, max per-element delta 0.000e+00" \
    "$TMP/or.small.log" || fail "the 64-item batch is not byte-identical to the per-item calls"
echo "EMBED BATCH GATE 27: leg 2: 64 small items, one call against 64"
oracle_summary "$TMP/or.small.log"

# ---- leg 3: the row budget's boundary --------------------------------------
BOUND_EXIT=$(run_oracle bound8192 "$TMP/mix.ids.tsv" --batch-synth 8192)
[ "$BOUND_EXIT" = 0 ] || { tail -8 "$TMP/or.bound8192.log" >&2; fail "an 8192-row call was refused (exit $BOUND_EXIT)"; }
grep -q "accepted: 8192 rows in one call" "$TMP/or.bound8192.log" || fail "the 8192-row call was not accepted by name"
[ "$(packed_mib_of "$TMP/or.bound8192.log")" = "$(expect_mib 8192)" ] \
    || fail "the 8192-row workspace is not the cost model's $(expect_mib 8192) MiB"
CROSS_EXIT=$(run_oracle bound8193 "$TMP/mix.ids.tsv" --batch-synth 8193)
[ "$CROSS_EXIT" = 9 ] || { tail -8 "$TMP/or.bound8193.log" >&2; fail "an 8193-row call was not refused (exit $CROSS_EXIT, expected 9)"; }
grep -q "^refused: the device batch entry took none of this call" "$TMP/or.bound8193.log" \
    || fail "the 8193-row refusal did not name itself"
echo "EMBED BATCH GATE 27: leg 3: the row budget's boundary"
oracle_summary "$TMP/or.bound8192.log"
oracle_summary "$TMP/or.bound8193.log"

# ---- the server legs --------------------------------------------------------
# start_server <backend>
start_server() {
    "$SERVER" --host 127.0.0.1 --port "$PORT" -m "$ARTIFACT" \
        --backend "$1" --model-id eg2vq > "$TMP/server.$1.log" 2>&1 &
    SRV_PID=$!
    for _ in $(seq 1 60); do
        if curl -s -m 2 "http://127.0.0.1:$PORT/health" > "$TMP/health.$1.json" 2>/dev/null; then
            return 0
        fi
        kill -0 "$SRV_PID" 2>/dev/null || break
        sleep 2
    done
    tail -5 "$TMP/server.$1.log" >&2
    fail "the $1 server did not come up on port $PORT"
}

req() { # <name> <body>
    printf '%s' "$2" > "$TMP/req.$1.json"
    curl -s -m 900 -X POST "http://127.0.0.1:$PORT/v1/embeddings" \
        -H 'content-type: application/json' --data-binary @"$TMP/req.$1.json" \
        -o "$TMP/res.$1.json" -w '%{http_code}'
}

# the k-th embedding array of a response, as one line, for a byte comparison
embedding() { # <name> <k>
    grep -o '"embedding":\[[^]]*\]' "$TMP/res.$1.json" | sed -n "$2p"
}

# the three SHORT items (the queries at the fixture's end): the wire legs are
# about the call's shape, and the long documents' seconds of tower are not needed
# to show it
Q1=$(sed -n '4p' "$TMP/mix.tsv" | cut -f3)
Q2=$(sed -n '5p' "$TMP/mix.tsv" | cut -f3)
Q3=$(sed -n '6p' "$TMP/mix.tsv" | cut -f3)
# The fixture texts are JSON strings: escape the two characters that would break
# one (SciFact abstracts and claims carry no others -- no control characters, and
# the gate's own request is rejected by the server if they did).
esc() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }
J1=$(esc "$Q1"); J2=$(esc "$Q2"); J3=$(esc "$Q3")

array_vs_singles() { # <backend>: the array request must equal the singles, byte for byte
    back=$1
    start_server "$back"
    grep -q '"item_tokens_max":2048,"batch_rows_max":8192' "$TMP/health.$back.json" \
        || fail "/health on the $back leg does not advertise item_tokens_max 2048 / batch_rows_max 8192"
    for n in 1 2 3; do
        eval "body=\$J$n"
        code=$(req "$back.single.$n" "{\"input\":[\"$body\"]}")
        [ "$code" = 200 ] || { cat "$TMP/res.$back.single.$n.json" >&2; fail "the $back single-item request $n answered $code"; }
    done
    code=$(req "$back.array" "{\"input\":[\"$J1\",\"$J2\",\"$J3\"]}")
    [ "$code" = 200 ] || { cat "$TMP/res.$back.array.json" >&2; fail "the $back array request answered $code"; }
    for n in 1 2 3; do
        a=$(embedding "$back.single.$n" 1)
        b=$(embedding "$back.array" "$n")
        [ -n "$a" ] || fail "the $back single request $n returned no embedding"
        [ "$a" = "$b" ] || fail "on the $back leg item $n's array embedding is not the single-item embedding"
    done
    # and the array really carried three items, in order
    [ "$(grep -o '"object":"embedding"' "$TMP/res.$back.array.json" | wc -l)" = 3 ] \
        || fail "the $back array response does not carry three embeddings"
    echo "EMBED BATCH GATE 27: leg $back: three items as one request == three single requests, byte for byte"
    stop_server
}

array_vs_singles cuda
array_vs_singles cpu

# ---- leg 4: the advertised row budget at the wire --------------------------
# An item of exactly 2048 tokens: the family's own ceiling.  The task prefix
# tokenizes to 8 tokens and BOS/EOS to 2, so `k` copies of a one-token unit give
# k + 10 ids (measured: 10 copies -> 20 tokens, 20 -> 30, by the gate's own
# tokenizer above); k = 2038 is therefore exactly 2048.  Four such items are
# exactly the 8192-row budget, and a fifth item crosses it.
UNIT=$(awk 'BEGIN { for (i = 0; i < 2038; i++) printf "a "; print "" }')
FULL="{\"input\":[\"$UNIT\",\"$UNIT\",\"$UNIT\",\"$UNIT\"]}"
OVER="{\"input\":[\"$UNIT\",\"$UNIT\",\"$UNIT\",\"$UNIT\",\"a\"]}"
start_server cuda
code=$(req cuda.budget "$FULL")
[ "$code" = 200 ] || { head -c 400 "$TMP/res.cuda.budget.json" >&2; fail "the exactly-8192-row request answered $code"; }
[ "$(grep -o '"object":"embedding"' "$TMP/res.cuda.budget.json" | wc -l)" = 4 ] \
    || fail "the budget request did not return four embeddings"
code=$(req cuda.over "$OVER")
[ "$code" = 400 ] || { head -c 400 "$TMP/res.cuda.over.json" >&2; fail "the over-budget request answered $code, not 400"; }
grep -q "batch_rows_max" "$TMP/res.cuda.over.json" \
    || fail "the over-budget refusal does not quote embed.batch_rows_max"
echo "EMBED BATCH GATE 27: leg 4: the wire's row budget"
echo "  four items of exactly 2048 tokens (8192 rows): 200, one call, four embeddings"
echo "  the same four plus a fifth item: 400 -- $(cat "$TMP/res.cuda.over.json")"
stop_server

echo "EMBED BATCH GATE 27: PASS: several items reach the tower in one call; a batched item's vector is its"
echo "EMBED BATCH GATE 27:       single-item vector byte for byte on both legs, the row budget binds exactly"
echo "EMBED BATCH GATE 27:       where it is advertised, and every malformed block is refused by name"
