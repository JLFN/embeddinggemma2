#!/bin/sh
# tests/qa-gate.sh — rule 19 guardrail mount (canonical source:
# ~/.opengrok/templates/qa-gate.sh; copied into this project at unit 20, which is
# what the handoff's rule 19 item recorded as the next code unit's duty).
#
# Enforces global rule 19: before a unit is committed for delivery or pushed, an
# AI QA-tester (deepseek-v4.1-CC-flash, the model this project recorded; never
# hardcoded in the canonical template) must verify every newly added surface and
# write a FRESH qa-evidence/qa-report.md ending in "verdict: overall PASS" that
# covers all of them. This mount FAILS the local gate (exit 1) until that holds;
# it SKIPS in real CI where the QA subagent cannot run.
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
#     QA_SURFACE_PATHS lists this repo's tools/ and tests/, which the template's
#     defaults do not cover.
#  3. The gate scans TWO repositories. This project ships the outer repo (docs,
#     tools, tests, the frozen set) and ds4-rs, the engine, which the outer
#     .gitignore excludes entirely -- measured: `git diff` in the outer tree
#     reports nothing for a ds4-rs change, which is how the first version of this
#     mount printed "no new surfaces" for a unit whose whole engine change was
#     real. The nested checkout is scanned against its OWN last pushed commit
#     (its publish/dev, 29873445 at the time of writing), and the report must be
#     newer than the newer of the two bases.
#
# Env knobs (advisory): QA_BASE, QA_MODEL, QA_SURFACE_PATHS, QA_REPORT,
# QA_DB_RE, QA_EP_RE, QA_CM_RE, QA_NESTED_REPO, QA_NESTED_BASE,
# QA_NESTED_PATHS.
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
NESTED=${QA_NESTED_REPO:-ds4-rs}
NESTED_BASE=${QA_NESTED_BASE:-publish/dev}
NESTED_PATHS=${QA_NESTED_PATHS:-crates ds4.c cuda}

# The patterns match DEFINITIONS, not mentions: a header name or a call site is
# not a surface a QA-tester can drive. The first version of this mount used a
# loose `ds4_...` pattern and the gate reported file names ("ds4_eg2vq_gpu" from
# "ds4_eg2vq_gpu.cuh"), which is noise the report would then have had to repeat.
DB_RE=${QA_DB_RE:-'^\+[A-Za-z_][A-Za-z0-9_ \*]*ds4_[a-z0-9_]+\($'}
EP_RE=${QA_EP_RE:-'"/v1/[a-z/]+"'}
CM_RE=${QA_CM_RE:-'^\+[[:space:]]*pub fn [a-z0-9_]+'}

DEFAULT_PATHS="tools:db
tests:db"
SPECS_TEXT=${QA_SURFACE_PATHS:-$DEFAULT_PATHS}

BASE_SHA=$(git rev-parse "${BASE}" 2>/dev/null || echo "")
HEAD_SHA=$(git rev-parse HEAD 2>/dev/null || echo "")
NESTED_BASE_SHA=""
NESTED_HEAD_SHA=""
if [ -e "$NESTED/.git" ]; then
  NESTED_BASE_SHA=$(git -C "$NESTED" rev-parse "${NESTED_BASE}" 2>/dev/null || echo "")
  NESTED_HEAD_SHA=$(git -C "$NESTED" rev-parse HEAD 2>/dev/null || echo "")
fi

# Freshness floor: the report must be newer than the newer of the two last
# pushed commits, because a unit can add surfaces to either repository.
base_sec=$(git log -1 --format=%ct "$BASE" 2>/dev/null || echo 0)
if [ -n "$NESTED_BASE_SHA" ]; then
  nsec=$(git -C "$NESTED" log -1 --format=%ct "$NESTED_BASE_SHA" 2>/dev/null || echo 0)
  [ "$nsec" -gt "$base_sec" ] && base_sec=$nsec
fi

# ---- surface extraction ---------------------------------------------------
extract() {  # extract <type> <diff-text>
  case "$1" in
    # The C/Rust runtime surfaces: the definition lines' own names.
    db) printf '%s\n' "$2" | grep -oE "$DB_RE" | sed -E 's/.*(ds4_[a-z0-9_]+)\($/\1/' | sort -u || true;;
    ep) printf '%s\n' "$2" | grep -oE "$EP_RE" | tr -d '"' | sort -u || true;;
    cm) printf '%s\n' "$2" | grep -oE "$CM_RE" | sed 's/^[+[:space:]]*pub fn //' | sort -u || true;;
    *) : ;;
  esac
}

outer_has_work=0
if [ -n "$BASE_SHA" ] && [ "$BASE_SHA" != "$HEAD_SHA" ]; then outer_has_work=1; fi
nested_has_work=0
if [ -n "$NESTED_BASE_SHA" ] && [ "$NESTED_BASE_SHA" != "$NESTED_HEAD_SHA" ]; then nested_has_work=1; fi

if [ "$outer_has_work" -eq 0 ] && [ "$nested_has_work" -eq 0 ]; then
  echo "QA GATE: ALL PASS (no unpushed commits in either repo; nothing new to QA-tester)"
  exit 0
fi

surfaces=""
while IFS= read -r spec; do
  [ -n "$spec" ] || continue
  path=${spec%%:*}
  type=${spec##*:}
  [ "$outer_has_work" -eq 1 ] || continue
  added=$(git diff "$BASE_SHA..$HEAD_SHA" -- "$path" 2>/dev/null | grep -E '^\+' || true)
  [ -n "$added" ] || continue
  surfaces=$(printf '%s\n%s\n' "$surfaces" "$(extract "$type" "$added")")
done <<EOF
$SPECS_TEXT
EOF

if [ "$nested_has_work" -eq 1 ]; then
  nested_added=$(git -C "$NESTED" diff "$NESTED_BASE_SHA..$NESTED_HEAD_SHA" \
      -- $NESTED_PATHS 2>/dev/null | grep -E '^\+' || true)
  if [ -n "$nested_added" ]; then
    for type in db ep cm; do
      surfaces=$(printf '%s\n%s\n' "$surfaces" "$(extract "$type" "$nested_added")")
    done
  fi
fi

all_surfaces=$(printf '%s\n' "$surfaces" | sed '/^[[:space:]]*$/d' | sort -u)
if [ -z "$all_surfaces" ]; then
  echo "QA GATE: ALL PASS (no new runtime, route or public-API surfaces relative to ${BASE} and ${NESTED}/${NESTED_BASE})"
  exit 0
fi
echo "New surfaces to be QA'd (${MODEL}, bases ${BASE} and ${NESTED}/${NESTED_BASE}):"
printf '%s\n' "$all_surfaces" | sed 's/^/  /'

failures=0
check() { if eval "$2"; then echo "PASS  $1"; else echo "FAIL  $1"; failures=$((failures+1)); fi; }

check "QA report exists ($(basename "$REPORT"))" "[ -f '$REPORT' ]"
check "QA report is fresh (>= the newer last pushed commit, epoch ${base_sec})" \
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
