# eg2_parity_check.awk — recompute a device-vs-CPU parity result from two raws.
#
# The gemma_vq end-to-end gate (ds4-rs/tests/parity/eg2vq_embd_gpu_oracle) prints
# its own aggregation and can write both legs as raws (--out, --out-cpu).  This
# instrument recomputes the parity numbers from those raws in awk double
# arithmetic, so the gate's summary is not the only reporter of its result.
#
# Usage:
#   awk -F'\t' -f tools/eg2_parity_check.awk <cpu-raw.tsv> <device-raw.tsv>
#
# Both raws are the oracle shape: <item> <TAB> <ntok> <TAB> <768 values>.
# The workload runs the ORACLE's own last argument as the second file, so
# "awk -F'\t' -f this.awk cpu.tsv device.tsv" needs the raws in that order.
#
# The first version of this instrument split fields on whitespace instead of
# tabs, which left it comparing one value per item and still printing a
# plausible-looking line (min cos 1.000000000).  Run it with -F'\t'; the
# malformed counter below is what would have caught that, so it is not optional.

FNR == NR {
    if (NF != 3) { bad++; next }
    n++
    k = split($3, v, " ")
    if (k != 768) { short++; next }
    for (i = 1; i <= 768; i++) { ref[n, i] = v[i] + 0.0 }
    next
}
{
    if (NF != 3) { bad++; next }
    m++
    k = split($3, w, " ")
    if (k != 768) { short++; next }
    dot = 0; na = 0; nb = 0; d2 = 0
    for (i = 1; i <= 768; i++) {
        a = ref[m, i]; b = w[i] + 0.0
        dot += a * b
        na += a * a
        nb += b * b
        d2 += (a - b) * (a - b)
    }
    c = dot / sqrt(na * nb)
    r = sqrt(d2 / na)
    if (m == 1 || c < minc) { minc = c; minc_at = $1 }
    if (m == 1 || r > maxr) { maxr = r; maxr_at = $1 }
}
END {
    printf "items cpu=%d device=%d  min cos %.9f (item %s)  max rel_rms %.3e (item %s)  malformed %d\n",
           n, m, minc, minc_at, maxr, maxr_at, bad + short + 0
}
