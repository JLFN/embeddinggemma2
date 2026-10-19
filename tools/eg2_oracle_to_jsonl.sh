#!/bin/bash
# eg2_oracle_to_jsonl.sh — turn the C oracle's per-item vector dump into the
# jsonl shape the anchor and the port both use, one row per item:
#
#   {"kind":"item","id":"<id>","ntok":<n>,"dims":{"768":[..],"512":[..],"256":[..],"128":[..]}}
#
# The 768 vector is the engine's own; 512/256/128 are its leading dimensions
# re-normalized after the cut, which is the checkpoint README's MRL rule
# ("Re-normalize after truncating") and the same rule tools/eg2_anchor.sh applies
# to the reference's vectors and tools/eg2_phase4_gate.sh applies to this
# oracle's. The awk body is byte-identical to the to_jsonl() function in
# tools/eg2_phase4_gate.sh (that script keeps its own copy so a gated script is
# never edited retroactively); the phase-5 gate diffs the two bodies and reports
# the result.
#
# Usage: tools/eg2_oracle_to_jsonl.sh <oracle-raw.txt> [id-prefix]
#   id-prefix, when given, is prepended to each item id in the output.
set -euo pipefail
RAW=${1:?usage: eg2_oracle_to_jsonl.sh <oracle-raw.txt> [id-prefix]}
PREFIX=${2:-}

awk -F'\t' -v prefix="$PREFIX" '
function arr(   i, s) { s = "["; for (i = 1; i <= 768; i++) s = s (i > 1 ? "," : "") sprintf("%.7f", v[i]);
                        return s "]" }
function trunc(k,   i, s, n2, t) { n2 = 0; for (i = 1; i <= k; i++) n2 += v[i] * v[i];
                        t = sqrt(n2); if (t == 0) t = 1;
                        s = "["; for (i = 1; i <= k; i++) s = s (i > 1 ? "," : "") sprintf("%.7f", v[i] / t);
                        return s "]" }
{
    n = split($3, v, " ");
    if (n != 768) { print "eg2_oracle_to_jsonl: line " NR " has " n " values" > "/dev/stderr"; exit 1 }
    printf "{\"kind\":\"item\",\"id\":\"%s%s\",\"ntok\":%s,\"dims\":{\"768\":%s,\"512\":%s,\"256\":%s,\"128\":%s}}\n", \
        prefix, $1, $2, arr(), trunc(512), trunc(256), trunc(128)
}' "$RAW"
