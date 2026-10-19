#!/bin/sh
# tests/qa-gate.sh — rule 19 guardrail mount (canonical source:
# ~/.opengrok/templates/qa-gate.sh; copied into this project at unit 20, which is
# what the handoff's rule 19 item recorded as the next code unit's duty).
#
# Enforces global rule 19: before a unit is committed for delivery or pushed, an
# AI QA-tester (deepseek-v4.1-CC-flash, the model this project recorded; never
# hardcoded here) must verify every newly added surface and write a FRESH
# qa-evidence/qa-report.md ending in "verdict: overall PASS" that covers all of
# them. This mount FAILS the local gate (exit 1) until that holds; it SKIPS in
# real CI where the QA subagent cannot run.
#
# WHAT DIFFERS FROM THE CANONICAL TEMPLATE, and why (global rule 0.2: a deviation
# is explicit, measured and recorded, never silent):
#
#  1. QA_BASE defaults to `publish/dev`, not `origin/<branch>`. This repository
#     has no `origin`: its only remote is `publish`
#     (https://github.com/JLFN/embeddinggemma2.git), and dev is pushed there
#     directly under the operator's instruction. `publish/dev` is therefore the
#     last pushed commit, which is exactly what rule 19 item 4 wants the report to
#     be newer than.
#  2. The three surface patterns are overridden for this repository's languages
#     (C, Rust, shell), because the template's own patterns match a TypeScript
#     and Postgres project and would find NOTHING here -- a gate that always
#     prints "no new surfaces" is worse than no gate. The three slots keep the
#     template's meaning:
#       db (surfaces of the runtime)   C engine/device entry points
#       ep (endpoints)                 HTTP routes the wire surface exposes
#       cm (callable surface)          Rust public API added under ds4-rs/crates
#     QA_SURFACE_PATHS lists ds4-rs (both the C tree and the crates) plus this
#     repo's tools/ and tests/, which the template's defaults do not cover.
#
# Env knobs (advisory): QA_BASE, QA_MODEL, QA_SURFACE_PATHS, QA_REPORT,
# QA_DB_RE, QA_EP_RE, QA_CM_RE.
#
# Exit 0 = green (no new surfaces, or report fresh+pass+covering). Exit 1 red.
cd "$(dirname "$0")/.." || exit 1

# Real CI: the QA subagent is a local-session capability, not provisioned there.
if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
  echo "SKIP tests/qa-gate.sh (AI QA-tester is a local-session step, not provisioned in CI)"
  exit 0
fi

BRANCH=$(git symbolic-ref --short HEAD 2>/dev/null || echo "main")
BASE=${QA_BASE:-publish/dev}
MODEL=${QA_MODEL:-deepseek-v4.1-CC-flash}
REPORT=${QA_REPORT:-qa-evidence/qa-report.md}

DB_RE=${QA_DB_RE:-'ds4_(engine|gpu|bridge)_[a-z0-9_]+'}
EP_RE=${QA_EP_RE:-'"/v1/[a-z/]+"'}
CM_RE=${QA_CM_RE:-'pub fn [a-z0-9_]+'}

DEFAULT_PATHS="ds4-rs:db
ds4-rs:ep
ds4-rs/crates:db
ds4-rs/crates:cm
tools:db
tests:db"
SPECS_TEXT=${QA_SURFACE_PATHS:-$DEFAULT_PATHS}

BASE_SHA=$(git rev-parse "${BASE}" 2>/dev/null || echo "")
HEAD_SHA=$(git rev-parse HEAD 2>/dev/null || echo "")
if [ -z "$BASE_SHA" ] || [ "$BASE_SHA" = "$HEAD_SHA" ]; then
  echo "QA GATE: ALL PASS (no unpushed commits yet; nothing new to QA-tester)"
  exit 0
fi

# ---- surface extraction over the paths that were touched -------------------
surfaces=""
while IFS= read -r spec; do
  [ -n "$spec" ] || continue
  path=${spec%%:*}
  type=${spec##*:}
  added=$(git diff "$BASE_SHA..$HEAD_SHA" -- "$path" 2>/dev/null | grep -E '^\+' || true)
  [ -n "$added" ] || continue
  case "$type" in
    db) found=$(printf '%s\n' "$added" | grep -oE "$DB_RE" || true);;
    ep) found=$(printf '%s\n' "$added" | grep -oE "$EP_RE" | tr -d '"' || true);;
    cm) found=$(printf '%s\n' "$added" | grep -oE "$CM_RE" | sed 's/^pub fn //' | sort -u || true);;
    *) found="";;
  esac
  surfaces=$(printf '%s\n%s\n' "$surfaces" "$found")
done <<EOF
$SPECS_TEXT
EOF

# A C entry point and a Rust method can share a name with a test helper; the
# report only has to NAME each surface once, so dedup is by the extracted string.
all_surfaces=$(printf '%s\n' "$surfaces" | sed '/^[[:space:]]*$/d' | sort -u)
if [ -z "$all_surfaces" ]; then
  echo "QA GATE: ALL PASS (no new runtime, route or public-API surfaces relative to ${BASE})"
  exit 0
fi
echo "New surfaces to be QA'd (${MODEL}, base ${BASE}):"
printf '%s\n' "$all_surfaces" | sed 's/^/  /'

failures=0
check() { if eval "$2"; then echo "PASS  $1"; else echo "FAIL  $1"; failures=$((failures+1)); fi; }

check "QA report exists ($(basename "$REPORT"))" "[ -f '$REPORT' ]"
base_sec=$(git log -1 --format=%ct "$BASE" 2>/dev/null || echo 0)
check "QA report is fresh (>= last pushed commit at ${BASE})" \
  "[ -f '$REPORT' ] && [ \"\$(stat -c %Y '$REPORT' 2>/dev/null || echo 0)\" -ge '$base_sec' ]"
# The OPERATIVE verdict is the report's LAST non-empty line, and nothing else --
# the template's two hardened rules, kept verbatim: grepping anywhere let a
# superseded PASS line win over a later FAIL, and taking the last *verdict* line
# let an appended fenced block become the operative one. The contract the reports
# keep is "the file ENDS with the verdict line".
last_verdict=$(grep -v '^[[:space:]]*$' "$REPORT" 2>/dev/null | tail -1 | tr 'A-Z' 'a-z' | tr -s ' ' | sed 's/[[:space:]]*$//')
check "operative verdict is PASS (report's last non-empty line: ${last_verdict:-none})" \
  "[ '$last_verdict' = 'verdict: overall pass' ]"

while IFS= read -r surf; do
  [ -n "$surf" ] || continue
  check "report covers surface: $surf" "grep -qF '$surf' '$REPORT'"
done <<EOF
$all_surfaces
EOF

if [ "$failures" -eq 0 ]; then
  echo "QA GATE: overall PASS (${MODEL} evidence fresh and covering)"
  exit 0
else
  echo "QA GATE: FAIL (${failures} check(s) red). Run the AI QA-tester on ${MODEL},"
  echo "have it verify the surfaces above live, and update $REPORT with"
  echo "'verdict: overall PASS' before committing/pushing."
  exit 1
fi
