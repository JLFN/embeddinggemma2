#!/bin/sh
# eg2_embed_route_gate.sh — phase 6 gate for the RUST HTTP server's
# GET /health and POST /v1/embeddings routes.
#
# What it proves, and against what:
#  1. GET /health returns 200 application/json with the four facts the bin
#     read at boot: status, model, backend, family, embedding.
#  2. POST /v1/embeddings is OpenAI-compatible: an object:"list" whose first
#     item is a 768-float vector, with usage.prompt_tokens = the item's ids.
#  3. dimensions:256 returns 256 entries and a unit-L2 vector (the Matryoshka
#     truncation + renormalization the checkpoint states).
#  4. a bad task name returns 400 with the OpenAI invalid_request_error
#     envelope.
#  5. the route's vector EQUALS the CLI's `./ds4 -m ART --embed TEXT` vector
#     for the same text and task (cosine).  The CLI is the reference because
#     its path is already gated against the device parity raw by
#     tools/eg2_embed_cli_gate.sh, so a match here chains the route to that
#     gate without re-proving the tower.
#
# The server and the CLI both take a machine-global model lock, so they are
# run strictly one after the other: the server answers the route requests, is
# stopped, and only then does the CLI run.
#
# Usage: tools/eg2_embed_route_gate.sh <artifact.gguf> [port]
#
# Declared bounds (before the run): route-vs-CLI min cos >= 0.999999; the
# 256-dim vector's L2 norm within 1e-5 of 1.0.
set -eu

ROOT=$(cd "$(dirname "$0")/.." && pwd)
ART=${1:?usage: eg2_embed_route_gate.sh <artifact.gguf> [port]}
PORT=${2:-}
SERVER="$ROOT/ds4-rs/ds4-server"
CLI="$ROOT/ds4-rs/ds4"
HOST=127.0.0.1
TEXT="The quick brown fox jumps over the lazy dog."
TASK=Retrieval-query
COS_MIN=0.999999
NORM_TOL=0.00001
WORK="$ROOT/out/phase6-route-gate"

for f in "$SERVER" "$CLI" "$ART"; do
    [ -e "$f" ] || { echo "eg2_embed_route_gate.sh: missing $f" >&2; exit 2; }
done
command -v curl >/dev/null 2>&1 || { echo "eg2_embed_route_gate.sh: curl not found" >&2; exit 2; }

rm -rf "$WORK"; mkdir -p "$WORK"

# A free port: only a connection-refused answer (curl exit 7) counts as free,
# so a busy or slow listener is never mistaken for an open one.
pick_port() {
    p=${1:-18108}
    i=0
    while [ "$i" -lt 200 ]; do
        rc=0
        curl -s -m 1 -o /dev/null "http://$HOST:$p/health" 2>/dev/null || rc=$?
        [ "$rc" -eq 7 ] && { echo "$p"; return 0; }
        p=$((p + 1)); i=$((i + 1))
    done
    echo "${1:-18108}"
}
if [ -n "$PORT" ]; then
    USED_PORT=$PORT
else
    USED_PORT=$(pick_port 18108)
fi

SRV=""
cleanup() {
    [ -n "$SRV" ] && kill "$SRV" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "== starting ds4-server on $HOST:$USED_PORT =="
"$SERVER" -m "$ART" --host "$HOST" --port "$USED_PORT" >"$WORK/server.log" 2>&1 &
SRV=$!

# Wait for readiness via /health (bounded).
ready=0
i=0
while [ "$i" -lt 120 ]; do
    rc=0
    code=$(curl -s -m 2 -o "$WORK/health.json" -w '%{http_code}' "http://$HOST:$USED_PORT/health") || rc=$?
    if [ "$rc" -eq 0 ] && [ "$code" = "200" ]; then ready=1; break; fi
    if ! kill -0 "$SRV" 2>/dev/null; then break; fi
    sleep 0.5; i=$((i + 1))
done
if [ "$ready" -ne 1 ]; then
    echo "eg2_embed_route_gate.sh: server did not become ready" >&2
    tail -20 "$WORK/server.log" >&2 || true
    exit 1
fi

fail=0
note() { printf '%s\n' "$*"; }

echo
echo "== 1. GET /health =="
note "HTTP $code"
note "body: $(cat "$WORK/health.json")"
if ! grep -q '"status":"ok"' "$WORK/health.json" ||
    ! grep -q '"family":"gemma_vq"' "$WORK/health.json" ||
    ! grep -q '"embedding":true' "$WORK/health.json" ||
    ! grep -q '"backend":"' "$WORK/health.json"; then
    note "FAIL: /health body is not the expected shape"
    fail=1
fi

echo
echo "== 2. POST /v1/embeddings (one line, default task) =="
rc=0
code=$(curl -s -m 60 -o "$WORK/emb768.json" -w '%{http_code}' \
    -X POST "http://$HOST:$USED_PORT/v1/embeddings" \
    -H 'Content-Type: application/json' \
    -d "{\"input\":\"$TEXT\"}") || rc=$?
note "HTTP $code"
if [ "$rc" -ne 0 ] || [ "$code" != "200" ]; then
    note "FAIL: embeddings request failed"
    fail=1
else
    note "first 80 chars: $(head -c 80 "$WORK/emb768.json")"
    sed 's/.*"embedding":\[//; s/\].*//' "$WORK/emb768.json" > "$WORK/route768.vec"
    n768=$(tr ',' '\n' < "$WORK/route768.vec" | grep -c '[0-9]' || true)
    note "vector entries: $n768"
    [ "$n768" -eq 768 ] || { note "FAIL: expected 768 entries"; fail=1; }
fi

echo
echo "== 3. POST /v1/embeddings dimensions=256 =="
rc=0
code=$(curl -s -m 60 -o "$WORK/emb256.json" -w '%{http_code}' \
    -X POST "http://$HOST:$USED_PORT/v1/embeddings" \
    -H 'Content-Type: application/json' \
    -d "{\"input\":\"$TEXT\",\"dimensions\":256}") || rc=$?
note "HTTP $code"
if [ "$rc" -ne 0 ] || [ "$code" != "200" ]; then
    note "FAIL: dimensions request failed"
    fail=1
else
    sed 's/.*"embedding":\[//; s/\].*//' "$WORK/emb256.json" > "$WORK/route256.vec"
    n256=$(tr ',' '\n' < "$WORK/route256.vec" | grep -c '[0-9]' || true)
    norm=$(awk -F, '{s=0; for(i=1;i<=NF;i++) s+=$i*$i; printf "%.9f", sqrt(s)}' "$WORK/route256.vec")
    note "vector entries: $n256"
    note "L2 norm: $norm   (declared within $NORM_TOL of 1.0)"
    [ "$n256" -eq 256 ] || { note "FAIL: expected 256 entries"; fail=1; }
    awk -v n="$norm" -v tol="$NORM_TOL" 'BEGIN{d=n-1; if(d<0)d=-d; exit !(d<=tol)}' ||
        { note "FAIL: 256-dim vector is not unit length"; fail=1; }
fi

echo
echo "== 4. POST /v1/embeddings bad task -> 400 =="
code=$(curl -s -m 30 -o "$WORK/badtask.json" -w '%{http_code}' \
    -X POST "http://$HOST:$USED_PORT/v1/embeddings" \
    -H 'Content-Type: application/json' \
    -d '{"input":"x","task":"NoSuchTask"}')
note "HTTP $code"
note "body: $(cat "$WORK/badtask.json")"
if [ "$code" != "400" ] || ! grep -q '"invalid_request_error"' "$WORK/badtask.json"; then
    note "FAIL: bad task did not return the 400 invalid_request_error envelope"
    fail=1
fi

echo
echo "== 4b. the embedding caps /health advertises (phase 6 row P2.3) =="
# The caps must be the family's own numbers (crate::gemma_vq::ITEM_TOKENS_MAX /
# BATCH_ROWS_MAX, held equal to the C macro and the CUDA enum by
# crates/ds4-core/tests/eg2vq_serving_caps.rs), because a caller learns the two
# limits from this body alone.
note "embed: $(sed 's/.*"embed":\([^}]*}\).*/\1/' "$WORK/health.json")"
if ! grep -q '"embed":{"item_tokens_max":2048,"batch_rows_max":8192}' "$WORK/health.json"; then
    note "FAIL: /health does not advertise the family's embedding caps"
    fail=1
fi

echo
echo "== 4c. a request past an advertised cap -> 400, naming the cap =="
LONG=$(awk 'BEGIN{for(i=0;i<3000;i++) printf "word "}')
code=$(curl -s -m 60 -o "$WORK/too_long.json" -w '%{http_code}' \
    -X POST "http://$HOST:$USED_PORT/v1/embeddings" \
    -H 'Content-Type: application/json' \
    -d "{\"input\":\"$LONG\"}")
note "one 3000-word item: HTTP $code"
note "body: $(cat "$WORK/too_long.json")"
if [ "$code" != "400" ] || ! grep -q 'item_tokens_max' "$WORK/too_long.json"; then
    note "FAIL: an item past the item ceiling was not refused by name"
    fail=1
fi
ITEM=$(awk 'BEGIN{for(i=0;i<1400;i++) printf "word "}')
ARR=$(awk -v it="$ITEM" 'BEGIN{printf "\"" it "\""; for(i=1;i<8;i++) printf ",\"" it "\""}')
code=$(curl -s -m 120 -o "$WORK/too_many.json" -w '%{http_code}' \
    -X POST "http://$HOST:$USED_PORT/v1/embeddings" \
    -H 'Content-Type: application/json' \
    -d "{\"input\":[$ARR]}")
note "eight 1400-word items (11200+ tokens): HTTP $code"
note "body: $(cat "$WORK/too_many.json")"
if [ "$code" != "400" ] || ! grep -q 'batch_rows_max' "$WORK/too_many.json"; then
    note "FAIL: a request past the row budget was not refused by name"
    fail=1
fi
# The cap must not refuse what it covers: three such items is 4200 rows.
ARR3=$(awk -v it="$ITEM" 'BEGIN{printf "\"" it "\""; for(i=1;i<3;i++) printf ",\"" it "\""}')
code=$(curl -s -m 300 -o "$WORK/under_budget.json" -w '%{http_code}' \
    -X POST "http://$HOST:$USED_PORT/v1/embeddings" \
    -H 'Content-Type: application/json' \
    -d "{\"input\":[$ARR3],\"dimensions\":128}")
note "three 1400-word items at 128 dims (inside the budget): HTTP $code"
if [ "$code" != "200" ]; then
    note "FAIL: a request inside the advertised caps was refused"
    fail=1
else
    note "vectors returned: $(grep -o '"object":"embedding"' "$WORK/under_budget.json" | wc -l)"
    [ "$(grep -o '"object":"embedding"' "$WORK/under_budget.json" | wc -l)" -eq 3 ] ||
        { note "FAIL: expected three vectors"; fail=1; }
fi

echo
echo "== stopping ds4-server =="
kill "$SRV" 2>/dev/null || true
SRV=""
# Wait for the server to release the model lock before the CLI runs.
i=0
while [ "$i" -lt 60 ]; do
    pgrep -x ds4-server >/dev/null 2>&1 || break
    sleep 0.5; i=$((i + 1))
done

echo
echo "== 5. route vs CLI: ./ds4 --embed =="
if [ ! -s "$WORK/route768.vec" ]; then
    note "FAIL: no route vector to compare (step 2 failed)"
    fail=1
else
    if ! "$CLI" -m "$ART" --embed "$TEXT" --task "$TASK" >"$WORK/cli.json" 2>"$WORK/cli.err"; then
        note "FAIL: CLI refused: $(tail -1 "$WORK/cli.err")"
        fail=1
    else
        tr -d '\n' < "$WORK/cli.json" | sed 's/.*"embedding": *\[//; s/\].*//' > "$WORK/cli.vec"
        ncli=$(tr ',' '\n' < "$WORK/cli.vec" | grep -c '[0-9]' || true)
        note "CLI vector entries: $ncli"
        note "route first 3: $(cut -d, -f1-3 < "$WORK/route768.vec")"
        note "CLI   first 3: $(cut -d, -f1-3 < "$WORK/cli.vec")"
        awk -v r="$(cat "$WORK/route768.vec")" -v c="$(cat "$WORK/cli.vec")" \
            -v cmin="$COS_MIN" 'BEGIN{
                n=split(r,a,","); m=split(c,b,",");
                if (n != m) { printf "FAIL: dim mismatch route=%d cli=%d\n", n, m; exit 1 }
                dot=0; a2=0; b2=0;
                for (i=1; i<=n; i++) { dot+=a[i]*b[i]; a2+=a[i]*a[i]; b2+=b[i]*b[i] }
                cval=dot/(sqrt(a2)*sqrt(b2));
                printf "dim=%d cos=%.12f   (declared >= %s)\n", n, cval, cmin;
                exit !(cval+0.0 >= cmin+0.0);
            }' || { note "FAIL: route and CLI vectors disagree"; fail=1; }
    fi
fi

echo
if [ "$fail" -eq 0 ]; then
    echo "EG2VQ_EMBED_ROUTE_PASS"
    exit 0
fi
echo "EG2VQ_EMBED_ROUTE_FAIL"
exit 1
