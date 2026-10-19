/* eg2_subnorm.c — how many f16 subnormal values the artifact's VQ payloads
 * carry, because this project's two decoders disagree about them.
 *
 * Why it exists: writing the device path (docs/ds4-gemma-port-phases.md row
 * P3.1) surfaced a measured discrepancy in the CPU path's own decoder.  For any
 * subnormal f16 input (0 < |v| < 2^-14 = 6.10352e-05) the tree's vendored
 * `ds4vq_f16` (ds4-rs/ds41_vq_fmt.h:38-43) returns exactly HALF the value the
 * IEEE rule gives, while the artifact writer's own reader `go1b_fp16_to_fp32`
 * (/data/YoungAi/gguf-tools/quantize/onebit_quant.h:141-159) matches the IEEE
 * rule on every pattern tested.  Normals agree exactly in both.  The CPU
 * reference path binds its dense cache through `ds4vq_dequant_f32`
 * (ds4_eg2vq_bind.inc:305-310), so it decodes every subnormal codebook entry at
 * half magnitude; the device kernel decodes the same bytes correctly, so the
 * file, the CPU path and the device path are three different answers.
 *
 * What this tool measures, so the size of that difference is a number rather
 * than a worry: over every type-42 payload in the artifact's manifest, the count
 * of subnormal codebook entries and subnormal row gains, the range of the
 * codebook's magnitudes, and the WEIGHT-level error the vendored rule causes
 * (each codebook entry is shared by rows*cols/dim weights, so a subnormal entry
 * shifts that many weights by half its value — at most 3.05e-05 each).
 *
 * Usage: tools/eg2_subnorm <file> <span: gguf|cont> <manifest.tsv>
 *
 * The payload location comes from the manifest, so the tool reads the same bytes
 * the loader reads; each payload's own header is the authority for
 * dim/nc/rows/cols, exactly as the binders read it (ds4_eg2vq_bind.inc:64-83).
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

#define DQVQ_MAGIC 0x51565144u   /* 'DQVQ' (ds41_vq_fmt.h:32) */

/* The IEEE rule for a half, computed here independently of both converters so
 * the comparison is against arithmetic, not against another implementation. */
static double ieee_f16(uint16_t h) {
    const int s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1F, m = h & 0x3FF;
    if (e == 0) { return (double)s * (double)m * 5.9604644775390625e-08; }   /* m * 2^-24 */
    return (double)s * (double)(1024 + m) / 1024.0 * ldexp(1.0, e - 15);
}

/* The vendored converter, copied so the tool can report the decoded value
 * without linking the engine: ds4-rs/ds41_vq_fmt.h:38-43 verbatim. */
static float vendored_f16(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000u) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF, f;
    if (e == 0) { if (!m) f = s; else { e = 112; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3FF; f = s | (e << 23) | (m << 13); } }
    else if (e == 31) f = s | 0x7F800000u | (m << 13);
    else f = s | ((e + 112) << 23) | (m << 13);
    float out; memcpy(&out, &f, 4); return out;
}

static uint32_t rd_u32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static uint16_t rd_u16(const uint8_t *p) { uint16_t v; memcpy(&v, p, 2); return v; }

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <file> <gguf|cont> <manifest.tsv>\n", argv[0]);
        return 2;
    }
    FILE *f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "%s: cannot open %s\n", argv[0], argv[1]); return 2; }
    FILE *man = fopen(argv[3], "r");
    if (!man) { fprintf(stderr, "%s: cannot open %s\n", argv[0], argv[3]); fclose(f); return 2; }
    const int use_cont = (strcmp(argv[2], "cont") == 0);

    {   /* the two converters, side by side, on the patterns that separate them */
        static const uint16_t hs[] = {0x0001, 0x0200, 0x0346, 0x03FF, 0x0400, 0x0401, 0x3555, 0x8346};
        printf("half bit  IEEE rule          vendored ds4vq_f16   vendored/IEEE\n");
        for (unsigned i = 0; i < sizeof(hs) / sizeof(hs[0]); i++) {
            const double r = ieee_f16(hs[i]), v = (double)vendored_f16(hs[i]);
            printf("  0x%04X  %16.10g  %18.10g   %.4f\n", hs[i], r, v, (v != 0.0) ? r / v : 0.0);
        }
        printf("\n");
    }

    /* manifest columns: 1 name, 2 type, 3 nd, 4 dims, 5 bytes, 6 gguf_abs,
     * 7 role, 8 cont_abs, 9 st_file, 10 st_abs, 11 st_dtype.  The payload spans
     * are the same bytes in either container (X.8 measured the re-wrap
     * byte-preserving), so either column serves. */
    char line[8192];
    uint64_t payloads = 0, cb_entries = 0, cb_sub = 0, cb_zero = 0, gr_entries = 0, gr_sub = 0;
    double cb_sub_max = 0.0, cb_abs_min = 1e30, cb_abs_max = 0.0;
    double sub_mass = 0.0, sub_err_mass = 0.0, all_abs_mass = 0.0;
    double weight_err = 0.0, weight_mass = 0.0;
    char worst_name[256] = {0};
    double worst_err = 0.0;

    while (fgets(line, sizeof(line), man)) {
        if (line[0] == '#' || line[0] == '\n') { continue; }
        char *save = NULL;
        char *name = strtok_r(line, "\t\n", &save);
        if (!name) { continue; }
        char *f5 = NULL, *f8 = NULL, *ftype = NULL, *frole = NULL;
        char *tok = strtok_r(NULL, "\t\n", &save); int idx = 2;
        for (; tok; tok = strtok_r(NULL, "\t\n", &save), idx++) {
            if (idx == 2) { ftype = tok; }
            else if (idx == 5) { f5 = tok; }        /* bytes */
            else if (idx == 6) { f8 = tok; }        /* gguf_abs */
            else if (idx == 7) { frole = tok; }
            else if (idx == 8 && use_cont) { f8 = tok; }   /* cont_abs */
        }
        if (!ftype || !frole || strcmp(ftype, "42") != 0 || strcmp(frole, "vq") != 0) { continue; }
        const uint64_t off = strtoull(f8, NULL, 10), bytes = strtoull(f5, NULL, 10);
        uint8_t *pay = (uint8_t *)malloc(bytes);
        if (!pay) { break; }
        if (fseeko(f, (off_t)off, SEEK_SET) != 0 || fread(pay, 1, bytes, f) != bytes) {
            fprintf(stderr, "%s: cannot read %s payload at %llu\n", argv[0], name, (unsigned long long)off);
            free(pay); fclose(f); fclose(man); return 2;
        }
        if (rd_u32(pay) != DQVQ_MAGIC) {
            fprintf(stderr, "%s: %s does not begin with DQVQ\n", argv[0], name);
            free(pay); continue;
        }
        const uint32_t dim = rd_u16(pay + 4), nc = rd_u16(pay + 6);
        const uint32_t rows = rd_u32(pay + 8), cols = rd_u32(pay + 12);
        const uint8_t *cb = pay + 16, *gr = cb + (size_t)nc * dim * 2;
        const size_t nent = (size_t)nc * dim;
        const double per_entry = (double)rows * (double)cols / (double)dim;   /* weights sharing one codeword */
        payloads++;

        double local_sub_mass = 0.0;
        for (size_t i = 0; i < nent; i++) {
            const uint16_t h = rd_u16(cb + 2 * i);
            const int e = (h >> 10) & 0x1F, m = h & 0x3FF;
            const double a = fabs(ieee_f16(h));
            cb_entries++;
            all_abs_mass += a * per_entry;
            if (a > cb_abs_max) { cb_abs_max = a; }
            if (a > 0.0 && a < cb_abs_min) { cb_abs_min = a; }
            if (e == 0) {
                if (m == 0) { cb_zero++; continue; }
                cb_sub++;
                local_sub_mass += a;
                if (a > cb_sub_max) { cb_sub_max = a; }
            }
        }
        for (uint64_t r = 0; r < rows; r++) {
            const uint16_t h = rd_u16(gr + 2 * r);
            gr_entries++;
            if (((h >> 10) & 0x1F) == 0 && (h & 0x3FF) != 0) { gr_sub++; }
        }
        sub_mass += local_sub_mass;
        sub_err_mass += local_sub_mass * 0.5;
        const double we = local_sub_mass * 0.5 * per_entry;     /* the vendored rule halves each affected entry */
        weight_err += we;
        weight_mass += all_abs_mass - we;                        /* the rest of this payload's mass */
        if (we > worst_err) { worst_err = we; snprintf(worst_name, sizeof(worst_name), "%s", name); }
        free(pay);
    }

    printf("payloads (type 42, role vq)   %llu\n", (unsigned long long)payloads);
    printf("codebook entries              %llu over %llu payloads\n",
           (unsigned long long)cb_entries, (unsigned long long)payloads);
    printf("  zeros                       %llu\n", (unsigned long long)cb_zero);
    printf("  subnormal (e==0, m!=0)      %llu  (%.6f%% of entries)\n",
           (unsigned long long)cb_sub, cb_entries ? 100.0 * (double)cb_sub / (double)cb_entries : 0.0);
    printf("  |value| range               %.6g .. %.6g   (subnormal ceiling 6.10352e-05)\n", cb_abs_min, cb_abs_max);
    printf("  largest subnormal entry     %.6g\n", cb_sub_max);
    printf("  subnormal value mass        %.6g  (sum of |v| over subnormal entries)\n", sub_mass);
    printf("row gains                     %llu entries, %llu subnormal\n",
           (unsigned long long)gr_entries, (unsigned long long)gr_sub);
    printf("\nweight level (each codebook entry is shared by rows*cols/dim weights):\n");
    printf("  decoded weight mass (L1)    %.6g\n", weight_mass);
    printf("  error the vendored rule adds %.6g  (relative %.3g)\n",
           weight_err, weight_mass > 0.0 ? weight_err / weight_mass : 0.0);
    printf("  worst single payload        %s (%.6g)\n", worst_name[0] ? worst_name : "(none)", worst_err);
    fclose(f); fclose(man);
    return cb_sub ? 0 : 0;
}
