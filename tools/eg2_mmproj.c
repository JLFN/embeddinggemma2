/* eg2_mmproj.c — the EmbeddingGemma-2 VISION tower's mmproj writer (phase 8,
 * unit 22, docs/ds4-gemma-port-phases.md rows P4.1/P4.3/P4.4).
 *
 * Why this tool exists at all: the project's method is to gate every ported
 * tower against a frozen reference, and for TEXT that reference was a shipped
 * binary (llama-embedding at the pin). For IMAGES no shipped binary at the pin
 * embeds one — examples/embedding/embedding.cpp has no mtmd support and
 * tools/mtmd/mtmd-cli.cpp has no embedding or pooling mode — so the reference
 * has to be built: this tool writes the mmproj the pinned mtmd loader resolves,
 * and tests/parity/eg2vq_vision_oracle.cpp then runs one frozen image through
 * it and dumps mtmd_get_output_embd. That dump is unit 22's deliverable.
 *
 * Every key, name, dimension and transform below is READ, per project rule 0.2,
 * from these sources — nothing here is invented:
 *
 *   /data/llama.cpp @ 31288c9c (read-only, the form authority)
 *     tools/mtmd/clip-impl.h:31-74      the mmproj KV key spellings (KEY_*)
 *     tools/mtmd/clip-impl.h:112-158    the tensor name templates (TN_*), and
 *     tools/mtmd/clip-impl.h:428-429     v.std_bias / v.std_scale
 *     tools/mtmd/clip.cpp:1201-1288     what the loader reads unconditionally
 *     tools/mtmd/clip.cpp:1342-1352     the required hparams keys
 *     tools/mtmd/clip.cpp:1719-1736     the GEMMA4V hparams block (n_merge 3,
 *                                       bicubic resize, rope theta 100)
 *     tools/mtmd/clip.cpp:2408-2450     the per-layer tensors it reads, and
 *                                       which of them are required (attn_out,
 *                                       ffn_up, ffn_down)
 *     tools/mtmd/clip.cpp:2800-2803     the GEMMA4V-specific ones
 *     tools/mtmd/clip.cpp:376/496/513/527  WHICH norm the graph uses where:
 *                                       ln_1 = pre-attention, attn_post_norm =
 *                                       post-attention, ln_2 = pre-FFN,
 *                                       ff_post_norm = post-FFN
 *     gguf-py/gguf/constants.py:1817-1876   the mmproj tensor names (V_*)
 *     gguf-py/gguf/tensor_mapping.py:1783-2124  the gemma4 HF names for each
 *     conversion/gemma.py:1001-1050     the projector type and the two source
 *                                       transforms (the conv reshape/permute,
 *                                       the position table's index order)
 *   The checkpoint: .models/EmbeddingGemma-2-914f7f8 @ 914f7f89142e (its
 *   config.json's vision_config is the shape authority; the tensor bytes come
 *   out of model.safetensors).
 *
 * DELIBERATE DEVIATIONS from the reference converter, both recorded in
 * docs/eg2vq-vision-oracle-gate22.txt rather than left implicit:
 *
 *   1. Every tensor is written in the checkpoint's OWN dtype (BF16), including
 *      position_embd, which conversion/gemma.py:1015-1018 forces to F32. The
 *      reason: this oracle exists to gate OUR port, and our port reads the
 *      artifact, whose vision tower is the checkpoint's BF16 bytes verbatim
 *      (`class keep` in docs/tensor-inventory.tsv). Writing F32 here would put
 *      a rounding between the reference and the thing being measured.
 *   2. No audio tensors and no clip.has_audio_encoder: the owner asked for
 *      vision, and the audio tower is out of scope (embed_audio is left alone).
 *
 * Usage: ./bin/eg2_mmproj <hfdir> <out.gguf> [--manifest FILE]
 */
#include "ds4_gguf_write.h"
#include <math.h>

static const char *const ARCH = "mmproj";
static const char *const PROJ_TYPE = "gemma4v";     /* clip-impl.h:557 */

static void die(const char *m) { fprintf(stderr, "eg2_mmproj: %s\n", m); exit(1); }
static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) die("out of memory"); return p; }

/* ------------------------------------------------------------------ JSON ---- */
/* The house scanner (eg2_gguf.c:165-345, itself the style of ds4_st41.h:52-90):
 * object keys, strings, arrays, numbers — everything this tool reads. */

static const char *jw(const char *p, const char *e) {
    while (p < e && (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r')) p++;
    return p;
}

static const char *jstr_end(const char *p, const char *e) {
    if (p >= e || *p != '"') die("json: string expected");
    for (p++; p < e; p++) {
        if (*p == '\\') { p++; if (p >= e) break; continue; }
        if (*p == '"') return p + 1;
    }
    die("json: unterminated string");
    return NULL;
}

static const char *jskip(const char *p, const char *e) {
    p = jw(p, e);
    if (p >= e) die("json: value expected");
    if (*p == '"') return jstr_end(p, e);
    if (*p == '{' || *p == '[') {
        int depth = 0;
        for (; p < e; p++) {
            if (*p == '"') { p = jstr_end(p, e) - 1; continue; }
            if (*p == '{' || *p == '[') depth++;
            else if (*p == '}' || *p == ']') { if (--depth == 0) return p + 1; }
        }
        die("json: unterminated container");
    }
    while (p < e && *p != ',' && *p != '}' && *p != ']' &&
           *p != ' ' && *p != '\n' && *p != '\t' && *p != '\r') p++;
    return p;
}

static int jobj_get(const char *obj, const char *e, const char *key,
                    const char **vstart, const char **vend) {
    const char *p = jw(obj, e);
    if (p >= e || *p != '{') die("json: object expected");
    size_t klen = strlen(key);
    for (p = jw(p + 1, e); p < e && *p != '}'; p = jw(p, e)) {
        const char *ks = p, *ke = jstr_end(p, e);
        int hit = ((size_t)(ke - ks) == klen + 2) && !strncmp(ks + 1, key, klen);
        p = jw(ke, e);
        if (p >= e || *p != ':') die("json: ':' expected");
        const char *vs = jw(p + 1, e), *ve = jskip(vs, e);
        if (hit) { *vstart = vs; *vend = ve; return 1; }
        p = jw(ve, e);
        if (p < e && *p == ',') p++;
    }
    return 0;
}

static const char *jstr_dec(const char *p, const char *e, size_t *out_len) {
    static char buf[512];
    if (p >= e || *p != '"') die("json: string expected");
    size_t n = 0;
    for (p++; p < e && *p != '"'; p++) {
        if (*p == '\\') { p++; if (p >= e) break; }
        if (n + 1 < sizeof buf) buf[n++] = *p;
    }
    buf[n] = 0;
    if (out_len) *out_len = n;
    return buf;
}

static uint64_t jnum_u64(const char *p, const char *e) {
    uint64_t v = 0;
    int neg = 0;
    p = jw(p, e);
    if (p < e && *p == '-') { neg = 1; p++; }
    for (; p < e && *p >= '0' && *p <= '9'; p++) v = v * 10 + (uint64_t)(*p - '0');
    return neg ? 0 : v;
}

static double jnum_f64(const char *p, const char *e) {
    char tmp[64];
    size_t n = 0;
    p = jw(p, e);
    while (p < e && n + 1 < sizeof tmp && (*p == '-' || *p == '+' || *p == '.' ||
           (*p >= '0' && *p <= '9') || *p == 'e' || *p == 'E')) tmp[n++] = *p++;
    tmp[n] = 0;
    return strtod(tmp, NULL);
}

static int jis_num(const char *p, const char *e) {
    p = jw(p, e);
    return p < e && (*p == '-' || (*p >= '0' && *p <= '9'));
}

static char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "eg2_mmproj: cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *b = xmalloc((size_t)sz + 1);
    if (fread(b, 1, (size_t)sz, f) != (size_t)sz) die("short read");
    b[sz] = 0;
    fclose(f);
    *len = (size_t)sz;
    return b;
}

static int jget(const char *root, const char *end, const char *path,
                const char **vs, const char **ve) {
    const char *s = root, *e = end;
    const char *p = path;
    for (;;) {
        const char *dot = strchr(p, '.');
        char key[160];
        size_t n = dot ? (size_t)(dot - p) : strlen(p);
        if (n >= sizeof key) die("json: path segment too long");
        memcpy(key, p, n); key[n] = 0;
        if (!jobj_get(s, e, key, vs, ve)) return 0;
        if (!dot) return 1;
        s = *vs; e = *ve; p = dot + 1;
    }
}

static uint64_t cfg_u64(const char *root, const char *end, const char *path) {
    const char *vs, *ve;
    if (!jget(root, end, path, &vs, &ve)) {
        fprintf(stderr, "eg2_mmproj: config.json has no %s\n", path); exit(1);
    }
    if (!jis_num(vs, ve)) {
        fprintf(stderr, "eg2_mmproj: config.json %s is not a number\n", path); exit(1);
    }
    return jnum_u64(vs, ve);
}

static double cfg_f64(const char *root, const char *end, const char *path) {
    const char *vs, *ve;
    if (!jget(root, end, path, &vs, &ve)) {
        fprintf(stderr, "eg2_mmproj: config.json has no %s\n", path); exit(1);
    }
    if (!jis_num(vs, ve)) {
        fprintf(stderr, "eg2_mmproj: config.json %s is not a number\n", path); exit(1);
    }
    return jnum_f64(vs, ve);
}

/* an array of numbers at the dotted path; returns how many were read */
static size_t cfg_arr_f32(const char *root, const char *end, const char *path, float *out, size_t max) {
    const char *vs, *ve;
    if (!jget(root, end, path, &vs, &ve)) {
        fprintf(stderr, "eg2_mmproj: config.json has no %s\n", path); exit(1);
    }
    const char *p = vs;
    while (p < ve && *p != '[') p++;
    if (p < ve) p++;
    size_t n = 0;
    while (p < ve && *p != ']') {
        p = jw(p, ve);
        if (*p == ',') { p++; continue; }
        if (*p == ']') break;
        if (n >= max) {
            fprintf(stderr, "eg2_mmproj: config.json %s has more than %zu elements\n", path, max);
            exit(1);
        }
        out[n++] = (float)jnum_f64(p, ve);
        while (p < ve && *p != ',' && *p != ']') p++;
    }
    return n;
}

/* ------------------------------------------------------- safetensors ---- */
/* The container is trivial: [u64 header length little-endian][JSON header][data
 * blob]. Each header entry is "name": {"dtype": "...", "shape": [...],
 * "data_offsets": [start, end]} with the offsets relative to the data blob.
 * This tool reads only what it writes, and it reads the bytes from the file by
 * seeking, so the 1.4 GB checkpoint is never held in memory. */

typedef struct {
    const char *name;                  /* the checkpoint's own name */
    char dtype[8];
    uint32_t nd;
    uint64_t dim[4];                   /* HF order: outermost first */
    uint64_t off, bytes;               /* within the data blob */
} st_tensor;

static FILE *g_st = NULL;
static uint64_t g_st_data0 = 0;

static void st_open(const char *path, char **hdr, size_t *hdr_len) {
    g_st = fopen(path, "rb");
    if (!g_st) { fprintf(stderr, "eg2_mmproj: cannot open %s\n", path); exit(1); }
    uint8_t len8[8];
    if (fread(len8, 1, 8, g_st) != 8) die("safetensors: short header length");
    uint64_t hl = 0;
    for (int i = 7; i >= 0; i--) hl = (hl << 8) | len8[i];
    if (!hl || hl > (1u << 30)) die("safetensors: implausible header length");
    char *h = xmalloc((size_t)hl + 1);
    if (fread(h, 1, (size_t)hl, g_st) != (size_t)hl) die("safetensors: short header");
    h[hl] = 0;
    *hdr = h;
    *hdr_len = (size_t)hl;
    g_st_data0 = 8 + hl;
}

static int st_find(const char *hdr, size_t hdr_len, const char *name, st_tensor *out) {
    const char *vs, *ve;
    memset(out, 0, sizeof *out);
    if (!jobj_get(hdr, hdr + hdr_len, name, &vs, &ve)) return 0;
    out->name = name;
    const char *ds, *de;
    if (!jobj_get(vs, ve, "dtype", &ds, &de)) die("safetensors: an entry has no dtype");
    {
        size_t dlen = 0;
        const char *dname = jstr_dec(ds, de, &dlen);
        if (dlen >= sizeof out->dtype) die("safetensors: a dtype name is longer than this tool holds");
        snprintf(out->dtype, sizeof out->dtype, "%s", dname);
    }
    const char *ss, *se;
    if (!jobj_get(vs, ve, "shape", &ss, &se)) die("safetensors: an entry has no shape");
    {
        const char *p = ss;
        uint64_t n = 0;
        while (p < se && *p != '[') p++;          /* past the opening bracket */
        if (p < se) p++;
        for (; p < se && *p != ']'; ) {
            p = jw(p, se);
            if (*p == ',') { p++; continue; }
            if (*p == ']') break;
            if (n >= 4) die("safetensors: a tensor with more than 4 dimensions");
            out->dim[n++] = jnum_u64(p, se);
            while (p < se && *p != ',' && *p != ']') p++;
        }
        out->nd = (uint32_t)n;
    }
    const char *os, *oe;
    if (!jobj_get(vs, ve, "data_offsets", &os, &oe)) die("safetensors: an entry has no data_offsets");
    {
        const char *p = os;
        while (p < oe && *p != '[') p++;
        p++;
        out->off = jnum_u64(p, oe);
        while (p < oe && *p != ',') p++;
        p++;
        out->bytes = jnum_u64(p, oe) - out->off;
    }
    return 1;
}

static void st_read(const st_tensor *t, void *dst) {
    if (fseek(g_st, (long)(g_st_data0 + t->off), SEEK_SET) != 0) die("safetensors: seek failed");
    if (fread(dst, 1, (size_t)t->bytes, g_st) != (size_t)t->bytes) die("safetensors: short tensor read");
}

/* ------------------------------------------------------------- the plan ---- */
/* One output tensor: the mmproj name, the checkpoint name it comes from, the
 * GGUF dims (internal dimension first, the GGUF convention), which transform the
 * bytes need, and whether the output type is forced to f32.
 *
 * The f32 force is not cosmetic and not a preference: the reference's own graph
 * refuses a mixed-type elementwise op on the CPU backend — measured this session,
 * "binary_op: unsupported types: dst: f32, src0: f32, src1: bf16" from
 * ggml-cpu/binary-ops.cpp:135, aborting inside ggml_compute_forward_mul. Every
 * tensor the graph multiplies or adds elementwise against an f32 activation must
 * therefore be f32: the four per-layer norms (clip.cpp:376/496/513/527), the two
 * per-head q/k norms, and the position table (whose rows go through ggml_get_rows
 * and are then ADDED to the patch embeddings, gemma4v.cpp:46-47). The reference
 * converter agrees: conversion/gemma.py:1015-1018 forces the position table to
 * F32, and gguf-py writes the 1-D norms as F32. Everything else is a mul_mat
 * operand and keeps the checkpoint's own dtype. */
enum { XF_NONE = 0, XF_CONV_PATCH };

typedef struct {
    const char *gguf;
    char ckpt[200];
    uint64_t ne[4];
    uint32_t nd;
    int xform;
    int f32;
} mm_tensor;

static uint32_t gf_type(const char *dtype) {
    if (!strcmp(dtype, "BF16")) return 30;
    if (!strcmp(dtype, "F16"))  return 1;
    if (!strcmp(dtype, "F32"))  return 0;
    fprintf(stderr, "eg2_mmproj: unsupported source dtype %s\n", dtype);
    exit(1);
}

static uint64_t gf_bytes(uint32_t type, const uint64_t *ne, uint32_t nd) {
    uint64_t n = 1;
    for (uint32_t i = 0; i < nd; i++) n *= ne[i];
    return type == 0 ? n * 4 : n * 2;
}

typedef struct { char name[200]; uint32_t type, nd; uint64_t ne[4], bytes; char src[200]; char xf[24]; } man_row;

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <hfdir> <out.gguf> [--manifest FILE]\n", argv[0]);
        return 2;
    }
    const char *hfdir = argv[1], *outpath = argv[2], *manifest = NULL;
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--manifest") && i + 1 < argc) { manifest = argv[++i]; continue; }
        fprintf(stderr, "eg2_mmproj: unknown argument %s\n", argv[i]);
        return 2;
    }

    char p_config[4300], p_proc[4300], p_st[4300];
    snprintf(p_config, sizeof p_config, "%s/config.json", hfdir);
    snprintf(p_proc, sizeof p_proc, "%s/processor_config.json", hfdir);
    snprintf(p_st, sizeof p_st, "%s/model.safetensors", hfdir);

    size_t cfg_len = 0;
    char *cfg = read_file(p_config, &cfg_len);
    size_t proc_len = 0;
    char *proc = read_file(p_proc, &proc_len);

    /* The preprocessing constants come from the checkpoint's own
     * processor_config.json (image_processor.image_mean / image_std), which is
     * where the reference converter reads them from too
     * (conversion/base.py:2926-2929). The loader REQUIRES both arrays with at
     * least 3 elements (clip.cpp:1446-1452). */
    float img_mean[3] = { 0.0f, 0.0f, 0.0f }, img_std[3] = { 1.0f, 1.0f, 1.0f };
    if (cfg_arr_f32(proc, proc + proc_len, "image_processor.image_mean", img_mean, 3) != 3 ||
        cfg_arr_f32(proc, proc + proc_len, "image_processor.image_std", img_std, 3) != 3) {
        fprintf(stderr, "eg2_mmproj: processor_config.json's image_mean/image_std are not 3 elements each\n");
        return 1;
    }

    /* The shape authority is the checkpoint's own config.json. */
    const uint64_t V_HID   = cfg_u64(cfg, cfg + cfg_len, "vision_config.hidden_size");
    const uint64_t V_LAY   = cfg_u64(cfg, cfg + cfg_len, "vision_config.num_hidden_layers");
    const uint64_t V_HEADS = cfg_u64(cfg, cfg + cfg_len, "vision_config.num_attention_heads");
    const uint64_t V_KV    = cfg_u64(cfg, cfg + cfg_len, "vision_config.num_key_value_heads");
    const uint64_t V_HD    = cfg_u64(cfg, cfg + cfg_len, "vision_config.head_dim");
    const uint64_t V_FF    = cfg_u64(cfg, cfg + cfg_len, "vision_config.intermediate_size");
    const uint64_t V_PAT   = cfg_u64(cfg, cfg + cfg_len, "vision_config.patch_size");
    const uint64_t V_POS   = cfg_u64(cfg, cfg + cfg_len, "vision_config.position_embedding_size");
    const uint64_t V_POOL  = cfg_u64(cfg, cfg + cfg_len, "vision_config.pooling_kernel_size");
    const double   V_EPS   = cfg_f64(cfg, cfg + cfg_len, "vision_config.rms_norm_eps");
    const double   V_ROPE  = cfg_f64(cfg, cfg + cfg_len, "vision_config.rope_parameters.rope_theta");
    const uint64_t T_HID   = cfg_u64(cfg, cfg + cfg_len, "text_config.hidden_size");
    if (cfg_u64(cfg, cfg + cfg_len, "vision_config.num_attention_heads") != V_KV) {
        /* Not a refusal: the reference handles GQA. This build's towers happen
         * to be square, and the mmproj must state what the file holds. */
        fprintf(stderr, "eg2_mmproj: note: vision heads %llu != kv heads %llu\n",
                (unsigned long long)V_HEADS, (unsigned long long)V_KV);
    }

    char *hdr = NULL;
    size_t hdr_len = 0;
    st_open(p_st, &hdr, &hdr_len);

    /* The tensor plan, in the order it is written: the frontend, then the
     * projection, then the blocks. The names are clip-impl.h's TN_ templates
     * with the "v" prefix (clip.cpp:2273), and the source names are the
     * checkpoint's own (tensor_mapping.py:1783-2124 with the "model." that
     * conversion/gemma.py:1057-1060 prepends stripped again). */
    mm_tensor *plan = xmalloc(sizeof(mm_tensor) * (2 + 1 + (size_t)V_LAY * 13));
    int np = 0;

    plan[np++] = (mm_tensor){ "v.patch_embd.weight", "vision_tower.patch_embedder.input_proj.weight",
                              { V_PAT, V_PAT, 3, V_HID }, 4, XF_CONV_PATCH, 0 };
    plan[np++] = (mm_tensor){ "v.position_embd.weight", "vision_tower.patch_embedder.position_embedding_table",
                              { V_HID, V_POS, 2 }, 3, XF_NONE, 1 };
    plan[np++] = (mm_tensor){ "mm.input_projection.weight", "embed_vision.embedding_projection.weight",
                              { V_HID, T_HID }, 2, XF_NONE, 0 };

    for (uint64_t il = 0; il < V_LAY; il++) {
        static const struct { const char *tn, *src; int f32; } L[] = {
            { "ln1",            "input_layernorm.weight",            1 }, /* clip.cpp:376 */
            { "attn_post_norm", "post_attention_layernorm.weight",   1 }, /* clip.cpp:496 */
            { "ln2",            "pre_feedforward_layernorm.weight",  1 }, /* clip.cpp:513 */
            { "ffn_post_norm",  "post_feedforward_layernorm.weight", 1 }, /* clip.cpp:527 */
            { "attn_q_norm",    "self_attn.q_norm.weight",           1 },
            { "attn_k_norm",    "self_attn.k_norm.weight",           1 },
            { "attn_q",         "self_attn.q_proj.linear.weight",    0 },
            { "attn_k",         "self_attn.k_proj.linear.weight",    0 },
            { "attn_v",         "self_attn.v_proj.linear.weight",    0 },
            { "attn_out",       "self_attn.o_proj.linear.weight",    0 },
            { "ffn_gate",       "mlp.gate_proj.linear.weight",       0 },
            { "ffn_up",         "mlp.up_proj.linear.weight",         0 },
            { "ffn_down",       "mlp.down_proj.linear.weight",       0 },
        };
        for (size_t k = 0; k < sizeof L / sizeof L[0]; k++) {
            char ck[200];
            snprintf(ck, sizeof ck, "vision_tower.encoder.layers.%llu.%s",
                     (unsigned long long)il, L[k].src);
            st_tensor st;
            if (!st_find(hdr, hdr_len, ck, &st)) {
                fprintf(stderr, "eg2_mmproj: the checkpoint has no %s\n", ck);
                return 1;
            }
            mm_tensor *t = &plan[np];
            memset(t, 0, sizeof *t);
            snprintf(t->ckpt, sizeof t->ckpt, "%s", ck);
            char g[200];
            snprintf(g, sizeof g, "v.blk.%llu.%s.weight", (unsigned long long)il, L[k].tn);
            t->gguf = strdup(g);
            /* dims: HF order -> GGUF order is a reversal */
            t->nd = st.nd;
            for (uint32_t d = 0; d < st.nd; d++) t->ne[d] = st.dim[st.nd - 1 - d];
            t->xform = XF_NONE;
            t->f32 = L[k].f32;
            np++;
        }
    }

    if (manifest) {
        FILE *mf = fopen(manifest, "wb");
        if (!mf) { fprintf(stderr, "eg2_mmproj: cannot write %s\n", manifest); return 1; }
        fprintf(mf, "name\ttype\tnd\tdim0\tdim1\tdim2\tdim3\tbytes\tsrc\tsrc_dtype\tsrc_bytes\txform\n");
        for (int i = 0; i < np; i++) {
            st_tensor st;
            if (!st_find(hdr, hdr_len, plan[i].ckpt, &st)) die("manifest: a source tensor vanished");
            const uint32_t type = plan[i].f32 ? 0u : gf_type(st.dtype);
            fprintf(mf, "%s\t%u\t%u\t%llu\t%llu\t%llu\t%llu\t%llu\t%s\t%s\t%llu\t%s\n",
                    plan[i].gguf, type, plan[i].nd,
                    (unsigned long long)plan[i].ne[0], (unsigned long long)plan[i].ne[1],
                    (unsigned long long)plan[i].ne[2], (unsigned long long)plan[i].ne[3],
                    (unsigned long long)gf_bytes(type, plan[i].ne, plan[i].nd),
                    plan[i].ckpt, st.dtype, (unsigned long long)st.bytes,
                    plan[i].xform == XF_NONE ? "dims-reversed" : "conv-permuted");
        }
        fclose(mf);
    }

    ds4gw W;
    if (ds4gw_begin(&W, outpath) != 0) return 1;

    /* ---- the keys. Every one of them is a KEY_* the loader resolves; the
     * required ones (no default in the loader) are all here. ---- */
    ds4gw_kv_str(&W, "general.architecture", ARCH);
    ds4gw_kv_str(&W, "general.name", "EmbeddingGemma-2 vision tower (mmproj)");
    ds4gw_kv_str(&W, "general.description", "gemma4v vision tower + the vision tie-in projection, written from the pinned checkpoint (eg2_mmproj)");
    ds4gw_kv_bool(&W, "clip.has_vision_encoder", 1);   /* clip-impl.h:36, read at clip.cpp:1258 */
    ds4gw_kv_bool(&W, "clip.has_audio_encoder", 0);    /* the audio tower is out of scope */
    ds4gw_kv_str(&W, "clip.projector_type", PROJ_TYPE);/* clip-impl.h:34, clip-impl.h:557 */
    ds4gw_kv_u32(&W, "clip.vision.embedding_length", (uint32_t)V_HID);
    ds4gw_kv_u32(&W, "clip.vision.feed_forward_length", (uint32_t)V_FF);
    ds4gw_kv_u32(&W, "clip.vision.block_count", (uint32_t)V_LAY);
    ds4gw_kv_u32(&W, "clip.vision.attention.head_count", (uint32_t)V_HEADS);
    ds4gw_kv_u32(&W, "clip.vision.attention.head_count_kv", (uint32_t)V_KV);
    ds4gw_kv_u32(&W, "clip.vision.attention.head_dim", (uint32_t)V_HD);
    ds4gw_kv_f32(&W, "clip.vision.attention.layer_norm_epsilon", (float)V_EPS);
    ds4gw_kv_u32(&W, "clip.vision.patch_size", (uint32_t)V_PAT);
    ds4gw_kv_u32(&W, "clip.vision.projection_dim", (uint32_t)T_HID);
    ds4gw_kv_u32(&W, "clip.vision.projector.scale_factor", (uint32_t)V_POOL); /* -> hparams.n_merge */
    /* image_size is REQUIRED by the loader (clip.cpp:1356) and unused by this
     * projector; the converter writes 224 with the same note
     * (conversion/gemma.py:984-985). */
    ds4gw_kv_u32(&W, "clip.vision.image_size", 224);
    ds4gw_kv_arr_f32(&W, "clip.vision.image_mean", img_mean, 3);
    ds4gw_kv_arr_f32(&W, "clip.vision.image_std", img_std, 3);

    /* ---- register, then write, in one order ---- */
    for (int i = 0; i < np; i++) {
        st_tensor st;
        if (!st_find(hdr, hdr_len, plan[i].ckpt, &st)) die("a source tensor vanished before registration");
        const uint32_t type = plan[i].f32 ? 0u : gf_type(st.dtype);
        const uint64_t n_bytes = gf_bytes(type, plan[i].ne, plan[i].nd);
        const uint64_t expect = plan[i].f32 ? st.bytes * 2u : st.bytes;
        if (n_bytes != expect) {
            fprintf(stderr, "eg2_mmproj: %s would be %llu bytes, the checkpoint holds %llu\n",
                    plan[i].gguf, (unsigned long long)n_bytes, (unsigned long long)st.bytes);
            return 1;
        }
        if (ds4gw_tensor(&W, plan[i].gguf, type, plan[i].nd, plan[i].ne, n_bytes) < 0) return 1;
    }
    if (ds4gw_header(&W) != 0) return 1;

    for (int i = 0; i < np; i++) {
        st_tensor st;
        if (!st_find(hdr, hdr_len, plan[i].ckpt, &st)) die("a source tensor vanished before writing");
        const uint32_t type = plan[i].f32 ? 0u : gf_type(st.dtype);
        const uint64_t n_bytes = gf_bytes(type, plan[i].ne, plan[i].nd);
        uint8_t *src = xmalloc((size_t)st.bytes);
        st_read(&st, src);
        void *data = src;
        uint8_t *owned1 = NULL, *owned2 = NULL;
        if (plan[i].f32 && type == 0 && !strcmp(st.dtype, "BF16")) {
            /* exact: a bf16 value IS the f32 whose high half it is (the same
             * reading the tree's other readers use, ds4_bf16_to_f32). */
            const uint64_t n = st.bytes / 2u;
            float *w = xmalloc((size_t)n * sizeof(float));
            for (uint64_t k = 0; k < n; k++) {
                uint16_t h;
                memcpy(&h, src + 2u * k, 2);
                const uint32_t u = (uint32_t)h << 16;
                memcpy(&w[k], &u, 4);
            }
            owned1 = (uint8_t *)w;
            data = w;
        }
        if (plan[i].xform == XF_CONV_PATCH) {
            /* conversion/gemma.py:1046-1050: the checkpoint stores the patch
             * embedder as a linear [n_embd][3*patch*patch] whose inner index is
             * (y, x, c); the reference's conv2d wants (c, y, x) within the
             * kernel, i.e. GGUF ne = [patch, patch, 3, n_embd] with
             *   dst[c*P*P + y*P + x] = src[(y*P + x)*3 + c].
             * The element width is the output's (2 bytes here). */
            const uint64_t P = V_PAT, N = V_HID, C = 3;
            const uint64_t w = gf_type(st.dtype) == 0 ? 4 : 2;
            uint8_t *perm = xmalloc((size_t)st.bytes);
            for (uint64_t o = 0; o < N; o++) {
                const uint8_t *srow = (const uint8_t *)data + o * C * P * P * w;
                uint8_t *drow = perm + o * C * P * P * w;
                for (uint64_t c = 0; c < C; c++) {
                    for (uint64_t y = 0; y < P; y++) {
                        for (uint64_t x = 0; x < P; x++) {
                            const uint64_t d = c * P * P + y * P + x;
                            const uint64_t s = (y * P + x) * C + c;
                            memcpy(drow + d * w, srow + s * w, (size_t)w);
                        }
                    }
                }
            }
            owned2 = perm;
            data = perm;
        }
        if (ds4gw_write(&W, i, data, n_bytes) != 0) return 1;
        free(owned2);
        free(owned1);
        free(src);
    }
    if (ds4gw_end(&W) != 0) return 1;

    fprintf(stdout, "[mmproj] %d tensors, vision %llux%llu x%llu, ffn %llu, patch %llu, "
                    "heads %llu/%llu hd %llu, pool %llu, proj -> %llu, eps %.1e, rope %.1f\n",
            np, (unsigned long long)V_HID, (unsigned long long)V_HID, (unsigned long long)V_LAY,
            (unsigned long long)V_FF, (unsigned long long)V_PAT,
            (unsigned long long)V_HEADS, (unsigned long long)V_KV, (unsigned long long)V_HD,
            (unsigned long long)V_POOL, (unsigned long long)T_HID, V_EPS, V_ROPE);
    fprintf(stdout, "[done] %s\n", outpath);
    free(plan); free(hdr); free(cfg);
    if (g_st) fclose(g_st);
    return 0;
}
