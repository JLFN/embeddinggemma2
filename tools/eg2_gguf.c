/* eg2_gguf.c — the baseline EmbeddingGemma-2 GGUF writer (phase 1a, phase-doc
 * rows X.5 / X.5a): the pinned checkpoint re-expressed as a GGUF that the
 * reference loader resolves, so llama.cpp's embedding example (phase 2's anchor)
 * and our own C reference path can read the same file.
 *
 * Every key, type, name and dim below is READ from the file cited next to it,
 * per project rule 0.2; nothing here is invented. The sources:
 *
 *   /data/llama.cpp @ 31288c9c (the form authority, read-only)
 *     src/models/gemma-embedding2.cpp:3-13    the hparams keys the loader resolves
 *     src/models/gemma-embedding2.cpp:34-72   the 413 text-tower tensors and their
 *                                             declared shape, which is what the
 *                                             file's tensor table must match
 *     src/llama-arch.cpp:63-64                the arch string "gemma-embedding2"
 *     src/llama-arch.cpp:202-448              the KV key spellings (LLM_KV_*)
 *     src/llama-model.cpp:1382-1394           context/embedding/out/block_count
 *     src/llama-model.cpp:1462-1471           feed_forward_length, head counts
 *                                             (head_count_kv read as a 24-array)
 *     src/llama-model.cpp:1514-1541           key/value lengths and their SWA pair
 *     src/llama-model.cpp:3902-3912           sliding_window_pattern: array first,
 *                                             scalar fallback
 *     src/llama-hparams.cpp:8-17              set_swa_pattern(6, dense_first=false):
 *                                             is_swa = (il % 6 < 5)
 *     src/llama-hparams.h:524-533             the symmetric mask the pattern feeds
 *     src/llama-vocab.cpp:2236-2276           tokenizer.ggml.model "gemma4" is BPE
 *                                             and REQUIRES tokenizer.ggml.merges
 *     src/llama-vocab.cpp:2595-2637           tokens (required), scores, token_type
 *                                             (optional, length-checked)
 *     conversion/gemma.py:634-641             Gemma4Model.norm_shift = 0.0, so the
 *                                             norm weights are written VERBATIM (the
 *                                             +1.0 shift at :131 is Gemma3's)
 *     conversion/gemma.py:676-733             Gemma4Model.set_gguf_parameters: the
 *                                             exact key set for this arch, emitted
 *                                             as an array sliding_window_pattern,
 *                                             key/value lengths + SWA pair, and the
 *                                             per-layer head_count_kv array
 *     conversion/gemma.py:815-832             EmbeddingGemma2Model.set_gguf_parameters:
 *                                             sliding_window = 2 * sliding_window,
 *                                             embedding_length_out, causal=false,
 *                                             pooling from 1_Pooling/config.json
 *     conversion/gemma.py:641-676             Gemma4Model.set_vocab: model "gemma4",
 *                                             token list + scores + types, merges,
 *                                             add_space_prefix(false), add_bos(true)
 *     gguf-py/gguf/vocab.py:563-684           LlamaHfVocab: tokens are the model.vocab
 *                                             keys by id, score -1000.0, type BYTE for
 *                                             <0xNN>, CONTROL for a special id, else
 *                                             NORMAL; added tokens are those with
 *                                             id >= len(model.vocab)
 *     gguf-py/gguf/vocab.py:191-222           merges taken verbatim when they are a
 *                                             list of plain strings
 *     gguf-py/gguf/constants.py:6274          general.file_type = 32 (MOSTLY_BF16)
 *     gguf-py/gguf/constants.py:6184-6187     PoolingType.MEAN = 1
 *
 *   /data/YoungAi (the writer and payload precedent, revision a4e6d31)
 *     src/common/ds4_gguf_write.h             vendored byte-identical at
 *                                             tools/ds4_gguf_write.h, sha256
 *                                             5bfc5dd4fc8646709d521ad9a212aa0c6cba83cce2f0016baf0ff698364f1fff
 *                                             — the two-phase GGUF v3 writer every
 *                                             ds4 VQ family ships through; used
 *                                             exactly as gguf-tools/quantize/
 *                                             v41_to_gguf.c uses it (head KVs at
 *                                             :357-363, plan/write split at :460-490)
 *     src/common/ds4_st41.h:133-169           v41_st_open / _find / _data, the
 *                                             multi-shard safetensors index
 *
 *   The checkpoint: .models/EmbeddingGemma-2-914f7f8 @ 914f7f89142e, whose
 *   config.json supplies every value (text_config.*) and whose safetensors supply
 *   every tensor byte. Norm weights ship verbatim (Gemma4Model.norm_shift = 0.0).
 *
 * The baseline is the checkpoint FAITHFULLY: all 413 text-tower tensors BF16,
 * including the token table. The artifact (phase 1b) is the one that ships
 * token=q8_0 and the VQ towers, so phase 5's delta against this file is the whole
 * cost of the recipe, and phase 2's anchor is the unquantized original.
 *
 * Usage:
 *   eg2_gguf baseline <hf-dir> <out.gguf> [--limit N] [--only SUBSTR] [--no-tokenizer]
 *   eg2_gguf dump     <out.gguf>          (reads its own header back, as a check)
 *
 * --limit and --only write a SMOKE file: general.name carries the marker, so a
 * partial artifact cannot be mistaken for the baseline.
 *
 * Build (the reference toolchain's include path, as RECIPE.md section 4 does):
 *   cc -O3 -std=c99 -Wall -D_POSIX_C_SOURCE=200809L -Itools \
 *      -I/data/YoungAi/src/common -o bin/eg2_gguf tools/eg2_gguf.c -lm
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <ctype.h>

#include "ds4_gguf_write.h"    /* vendored: the reference's own GGUF v3 writer */
#include "ds4_st41.h"          /* the reference's safetensors index (locate only) */

/* GGUF tensor type numbers (ggml), the same table the engine uses
 * (/data/YoungAi/src/core/core_gguf.c:94-97, our crates/ds4-core/src/tensors.rs:60-92).
 * Only two appear in the baseline: f32 (0) and bf16 (30); q8_0 (8) is the
 * artifact's token table and is not written here. */
#define GGT_F32  0u
#define GGT_Q8_0 8u
#define GGT_BF16 30u

/* gguf-py/gguf/constants.py TokenType (the vocabulary's token_type values) */
#define TKN_NORMAL       1
#define TKN_UNKNOWN      2
#define TKN_CONTROL      3
#define TKN_USER_DEFINED 4
#define TKN_BYTE         6

/* the arch string the loader routes on (src/llama-arch.cpp:64) */
static const char *const ARCH = "gemma-embedding2";

static void die(const char *m) { fprintf(stderr, "eg2_gguf: %s\n", m); exit(1); }
static void *xmalloc(size_t n) { void *p = malloc(n ? n : 1); if (!p) die("out of memory"); return p; }

/* ------------------------------------------------------------------ JSON ---- */
/* A hand-rolled scanner, the house style (cf. ds4_st41.h:52-90). It needs only
 * what this tool reads: object keys, values, arrays of strings, and numbers. */

static const char *jw(const char *p, const char *e) {
    while (p < e && (*p == ' ' || *p == '\n' || *p == '\t' || *p == '\r')) p++;
    return p;
}

/* past a string literal; p at the opening quote */
static const char *jstr_end(const char *p, const char *e) {
    if (p >= e || *p != '"') die("json: string expected");
    for (p++; p < e; p++) {
        if (*p == '\\') { p++; if (p >= e) break; continue; }
        if (*p == '"') return p + 1;
    }
    die("json: unterminated string");
    return NULL;
}

/* past one value */
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

/* value of `key` at the top level of the object starting at obj; NULL if absent.
 * vstart/vend delimit the value. */
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

/* decode a string literal into fresh bytes (escapes and \uXXXX to UTF-8) */
static char *jstr_dec(const char *p, const char *e, size_t *out_len) {
    if (p >= e || *p != '"') die("json: string expected");
    size_t cap = (size_t)(e - p) + 8, n = 0;
    char *o = xmalloc(cap);
    for (p++; p < e; p++) {
        if (*p == '\\') {
            p++;
            if (p >= e) die("json: bad escape");
            char c = *p;
            switch (c) {
                case 'n': o[n++] = '\n'; break;
                case 't': o[n++] = '\t'; break;
                case 'r': o[n++] = '\r'; break;
                case 'b': o[n++] = '\b'; break;
                case 'f': o[n++] = '\f'; break;
                case '/': o[n++] = '/';  break;
                case '"': o[n++] = '"';  break;
                case '\\': o[n++] = '\\'; break;
                case 'u': {
                    if (p + 4 >= e) die("json: short \\u escape");
                    char h[5] = {p[1], p[2], p[3], p[4], 0};
                    unsigned cp = (unsigned)strtoul(h, NULL, 16);
                    p += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF && p + 6 < e && p[1] == '\\' && p[2] == 'u') {
                        char h2[5] = {p[3], p[4], p[5], p[6], 0};
                        unsigned lo = (unsigned)strtoul(h2, NULL, 16);
                        if (lo >= 0xDC00 && lo <= 0xDFFF) {
                            cp = 0x10000u + ((cp - 0xD800u) << 10) + (lo - 0xDC00u);
                            p += 6;
                        }
                    }
                    if (cp < 0x80) o[n++] = (char)cp;
                    else if (cp < 0x800) { o[n++] = (char)(0xC0 | (cp >> 6)); o[n++] = (char)(0x80 | (cp & 0x3F)); }
                    else if (cp < 0x10000) { o[n++] = (char)(0xE0 | (cp >> 12)); o[n++] = (char)(0x80 | ((cp >> 6) & 0x3F)); o[n++] = (char)(0x80 | (cp & 0x3F)); }
                    else { o[n++] = (char)(0xF0 | (cp >> 18)); o[n++] = (char)(0x80 | ((cp >> 12) & 0x3F)); o[n++] = (char)(0x80 | ((cp >> 6) & 0x3F)); o[n++] = (char)(0x80 | (cp & 0x3F)); }
                    break;
                }
                default: die("json: unknown escape");
            }
        } else if (*p == '"') {
            o[n] = 0; *out_len = n; return o;
        } else {
            o[n++] = *p;
        }
        if (n + 8 > cap) { cap *= 2; o = realloc(o, cap); if (!o) die("out of memory"); }
    }
    die("json: unterminated string");
    return NULL;
}

static uint64_t jnum_u64(const char *v, const char *e) {
    char buf[64]; size_t n = (size_t)(e - v); if (n >= sizeof buf) die("json: number too long");
    memcpy(buf, v, n); buf[n] = 0;
    return strtoull(buf, NULL, 10);
}
static double jnum_f64(const char *v, const char *e) {
    char buf[64]; size_t n = (size_t)(e - v); if (n >= sizeof buf) die("json: number too long");
    memcpy(buf, v, n); buf[n] = 0;
    return strtod(buf, NULL);
}
static int jis_num(const char *v, const char *e) { return v < e && (*v == '-' || (*v >= '0' && *v <= '9')); }

/* ------------------------------------------------------------------ model --- */

#define MAX_LAYER 64

typedef struct {
    /* text_config, read from the checkpoint's config.json */
    uint32_t n_layer, n_embd, n_head, n_ff, n_embd_out, n_embd_pl, n_vocab;
    uint32_t head_dim_swa, head_dim_full;
    uint32_t n_head_kv_swa, n_head_kv_full;
    uint32_t n_swa_hf;          /* the checkpoint's sliding_window (HF semantics) */
    uint32_t n_ctx;
    float    rms_eps, rope_base, rope_base_swa;
    uint8_t  is_swa[MAX_LAYER];
} hp_t;

static char *read_file(const char *path, size_t *len) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "eg2_gguf: cannot open %s\n", path); exit(1); }
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

/* the value at the dotted `path` inside `root` (a top-level object) */
static int jget(const char *root, const char *end, const char *path,
                const char **vs, const char **ve) {
    const char *s = root, *e = end;
    const char *p = path;
    for (;;) {
        const char *dot = strchr(p, '.');
        char key[128];
        size_t n = dot ? (size_t)(dot - p) : strlen(p);
        if (n >= sizeof key) die("json: path segment too long");
        memcpy(key, p, n); key[n] = 0;
        if (!jobj_get(s, e, key, vs, ve)) return 0;
        if (!dot) return 1;
        s = *vs; e = *ve; p = dot + 1;
    }
}

static uint64_t get_u64(const char *root, const char *end, const char *path) {
    const char *vs, *ve;
    if (!jget(root, end, path, &vs, &ve)) { fprintf(stderr, "eg2_gguf: config.json has no %s\n", path); exit(1); }
    if (!jis_num(vs, ve)) { fprintf(stderr, "eg2_gguf: config.json %s is not a number\n", path); exit(1); }
    return jnum_u64(vs, ve);
}
static double get_f64(const char *root, const char *end, const char *path) {
    const char *vs, *ve;
    if (!jget(root, end, path, &vs, &ve)) { fprintf(stderr, "eg2_gguf: config.json has no %s\n", path); exit(1); }
    if (!jis_num(vs, ve)) { fprintf(stderr, "eg2_gguf: config.json %s is not a number\n", path); exit(1); }
    return jnum_f64(vs, ve);
}

/* The hparams, each from the checkpoint's own config.json, cross-checked against
 * the arch's SWA rule (llama-hparams.cpp:8-17: is_swa = il % 6 < 5 for
 * sliding_window_pattern 6 with dense_first false — what LLAMA_SWA_TYPE_SYMMETRIC
 * plus load_swa_pattern(ml, 6) at gemma-embedding2.cpp:4-5 produce). */
static void load_hparams(const char *hfdir, hp_t *H) {
    char path[4300];
    snprintf(path, sizeof path, "%s/config.json", hfdir);
    size_t len;
    char *txt = read_file(path, &len);
    const char *end = txt + len;
    const char *ts, *te;
    if (!jget(txt, end, "text_config", &ts, &te)) die("config.json has no text_config");

    memset(H, 0, sizeof *H);
    H->n_layer        = (uint32_t)get_u64(ts, te, "num_hidden_layers");
    H->n_embd         = (uint32_t)get_u64(ts, te, "hidden_size");
    H->n_head         = (uint32_t)get_u64(ts, te, "num_attention_heads");
    H->n_ff           = (uint32_t)get_u64(ts, te, "intermediate_size");
    H->n_embd_out     = (uint32_t)get_u64(ts, te, "embedding_dim");
    H->n_embd_pl      = (uint32_t)get_u64(ts, te, "hidden_size_per_layer_input");
    H->n_vocab        = (uint32_t)get_u64(ts, te, "vocab_size");
    H->head_dim_swa   = (uint32_t)get_u64(ts, te, "head_dim");
    H->n_head_kv_swa  = (uint32_t)get_u64(ts, te, "num_key_value_heads");
    H->n_swa_hf       = (uint32_t)get_u64(ts, te, "sliding_window");
    H->n_ctx          = (uint32_t)get_u64(ts, te, "max_position_embeddings");
    H->rms_eps        = (float)get_f64(ts, te, "rms_norm_eps");
    H->rope_base      = (float)get_f64(ts, te, "rope_parameters.full_attention.rope_theta");
    H->rope_base_swa  = (float)get_f64(ts, te, "rope_parameters.sliding_attention.rope_theta");
    if (H->n_layer > MAX_LAYER) die("config.json: more layers than MAX_LAYER");

    /* layer_types: the SWA pattern, from the checkpoint itself. The converter
     * writes exactly this array (conversion/gemma.py:681-682). */
    const char *as, *ae, *p;
    if (!jget(ts, te, "layer_types", &as, &ae)) die("config.json has no text_config.layer_types");
    if (*as != '[') die("layer_types is not an array");
    p = jw(as + 1, ae);
    int il = 0;
    while (p < ae && *p != ']') {
        size_t sl;
        char *s = jstr_dec(p, ae, &sl);
        if (il >= (int)H->n_layer) die("layer_types longer than the layer count");
        H->is_swa[il] = (strcmp(s, "sliding_attention") == 0);
        if (!H->is_swa[il] && strcmp(s, "full_attention") != 0) die("unknown layer_types entry");
        free(s);
        il++;
        p = jw(jskip(p, ae), ae);
        if (p < ae && *p == ',') p = jw(p + 1, ae);
    }
    if (il != (int)H->n_layer) die("layer_types shorter than the layer count");

    /* per_layer_config: the full-attention layers' head_dim and KV head count
     * (config.json:99-116). Its keys are zero-padded two-digit strings. */
    H->head_dim_full = H->head_dim_swa;
    H->n_head_kv_full = H->n_head_kv_swa;
    const char *cs, *ce;
    if (jget(ts, te, "per_layer_config", &cs, &ce)) {
        char keybuf[16];
        for (int i = 0; i < (int)H->n_layer; i++) {
            if (H->is_swa[i]) continue;
            snprintf(keybuf, sizeof keybuf, "%02d", i);
            const char *ls, *le;
            if (!jobj_get(cs, ce, keybuf, &ls, &le))
                snprintf(keybuf, sizeof keybuf, "%d", i);      /* unpadded fallback */
            if (!jobj_get(cs, ce, keybuf, &ls, &le))
                { fprintf(stderr, "per_layer_config has no %s\n", keybuf); exit(1); }
            H->head_dim_full  = (uint32_t)get_u64(ls, le, "head_dim");
            H->n_head_kv_full = (uint32_t)get_u64(ls, le, "num_key_value_heads");
        }
    }
    free(txt);

    /* cross-check the SWA rule against the arch's own formula */
    for (int i = 0; i < (int)H->n_layer; i++) {
        int want = (i % 6 < 5);
        if ((int)H->is_swa[i] != want) {
            fprintf(stderr, "eg2_gguf: layer_types[%d] contradicts set_swa_pattern(6,false): "
                            "config says swa=%d, llama-hparams.cpp:8-17 says %d\n",
                    i, H->is_swa[i], want);
            exit(1);
        }
    }
    fprintf(stderr, "[hparams] %u layers, embd %u, heads %u (kv %u swa / %u full), ff %u, "
                    "out %u, per-layer %u, vocab %u, head_dim %u swa / %u full, swa window %u (file %u), ctx %u\n",
            H->n_layer, H->n_embd, H->n_head, H->n_head_kv_swa, H->n_head_kv_full, H->n_ff,
            H->n_embd_out, H->n_embd_pl, H->n_vocab, H->head_dim_swa, H->head_dim_full,
            H->n_swa_hf, 2 * H->n_swa_hf, H->n_ctx);
}

/* the per-layer head geometry, hparams' own helpers (llama-hparams.h:463-472) */
static void layer_geom(const hp_t *H, int il, uint32_t *h_k, uint32_t *h_v, uint32_t *kv) {
    *h_k = H->is_swa[il] ? H->head_dim_swa : H->head_dim_full;
    *h_v = *h_k;
    *kv  = H->is_swa[il] ? H->n_head_kv_swa : H->n_head_kv_full;
}

/* the tensor map: checkpoint suffix (after "language_model.") -> GGUF name, and
 * the dims the loader declares for it (gemma-embedding2.cpp:34-72). GGUF order is
 * inner dimension first, so a checkpoint row [rows, cols] becomes ne=[cols, rows]. */
typedef struct { const char *ckpt, *gguf; int per_layer; } tmap_t;

static const tmap_t TMAP[] = {
    { "embed_tokens.weight",                        "token_embd.weight",            0 },
    { "norm.weight",                                "output_norm.weight",           0 },
    { "embedding_projection.weight",                "output.weight",                0 },
    { "ple.per_layer_model_projection.weight",      "per_layer_model_proj.weight",  0 },
    { "ple.per_layer_projection_norm.weight",       "per_layer_proj_norm.weight",   0 },
    { "layers.%d.input_layernorm.weight",           "blk.%d.attn_norm.weight",            1 },
    { "layers.%d.self_attn.q_proj.weight",          "blk.%d.attn_q.weight",               1 },
    { "layers.%d.self_attn.k_proj.weight",          "blk.%d.attn_k.weight",               1 },
    { "layers.%d.self_attn.v_proj.weight",          "blk.%d.attn_v.weight",               1 },
    { "layers.%d.self_attn.o_proj.weight",          "blk.%d.attn_output.weight",          1 },
    { "layers.%d.self_attn.q_norm.weight",          "blk.%d.attn_q_norm.weight",          1 },
    { "layers.%d.self_attn.k_norm.weight",          "blk.%d.attn_k_norm.weight",          1 },
    { "layers.%d.post_attention_layernorm.weight",  "blk.%d.post_attention_norm.weight",  1 },
    { "layers.%d.pre_feedforward_layernorm.weight", "blk.%d.ffn_norm.weight",             1 },
    { "layers.%d.post_feedforward_layernorm.weight","blk.%d.post_ffw_norm.weight",        1 },
    { "layers.%d.mlp.gate_proj.weight",             "blk.%d.ffn_gate.weight",             1 },
    { "layers.%d.mlp.up_proj.weight",               "blk.%d.ffn_up.weight",               1 },
    { "layers.%d.mlp.down_proj.weight",             "blk.%d.ffn_down.weight",             1 },
    { "layers.%d.ple_block.per_layer_input_gate.weight", "blk.%d.inp_gate.weight",        1 },
    { "layers.%d.ple_block.per_layer_projection.weight", "blk.%d.proj.weight",            1 },
    { "layers.%d.ple_block.post_per_layer_input_norm.weight", "blk.%d.post_norm.weight",  1 },
    { "layers.%d.layer_scalar",                     "blk.%d.layer_output_scale.weight",   1 },
};
#define NTMAP ((int)(sizeof TMAP / sizeof TMAP[0]))

/* the shape the loader declares, per gemma-embedding2.cpp:34-72 */
static uint32_t expect_dims(const hp_t *H, int kind, int il, uint64_t ne[4]) {
    uint32_t h_k, h_v, kv;
    layer_geom(H, il, &h_k, &h_v, &kv);
    ne[0] = ne[1] = ne[2] = ne[3] = 0;
    switch (kind) {
        case 0:  ne[0] = H->n_embd; ne[1] = H->n_vocab; return 2;                  /* token_embd */
        case 1:  ne[0] = H->n_embd; return 1;                                       /* output_norm */
        case 2:  ne[0] = H->n_embd; ne[1] = H->n_embd_out; return 2;                /* output */
        case 3:  ne[0] = H->n_embd; ne[1] = H->n_embd_pl * H->n_layer; return 2;    /* per_layer_model_proj */
        case 4:  ne[0] = H->n_embd_pl; return 1;                                    /* per_layer_proj_norm */
        case 5:  ne[0] = H->n_embd; return 1;                                       /* attn_norm */
        case 6:  ne[0] = H->n_embd; ne[1] = h_k * H->n_head; return 2;               /* attn_q */
        case 7:  ne[0] = H->n_embd; ne[1] = h_k * kv; return 2;                      /* attn_k */
        case 8:  ne[0] = H->n_embd; ne[1] = h_v * kv; return 2;                      /* attn_v */
        case 9:  ne[0] = h_k * H->n_head; ne[1] = H->n_embd; return 2;               /* attn_output */
        case 10: ne[0] = h_k; return 1;                                              /* attn_q_norm */
        case 11: ne[0] = h_k; return 1;                                              /* attn_k_norm */
        case 12: ne[0] = H->n_embd; return 1;                                        /* attn_post_norm */
        case 13: ne[0] = H->n_embd; return 1;                                        /* ffn_norm */
        case 14: ne[0] = H->n_embd; return 1;                                        /* ffn_post_norm */
        case 15: ne[0] = H->n_embd; ne[1] = H->n_ff; return 2;                       /* ffn_gate */
        case 16: ne[0] = H->n_embd; ne[1] = H->n_ff; return 2;                       /* ffn_up */
        case 17: ne[0] = H->n_ff; ne[1] = H->n_embd; return 2;                       /* ffn_down */
        case 18: ne[0] = H->n_embd; ne[1] = H->n_embd_pl; return 2;                  /* inp_gate */
        case 19: ne[0] = H->n_embd_pl; ne[1] = H->n_embd; return 2;                  /* proj */
        case 20: ne[0] = H->n_embd; return 1;                                        /* post_norm */
        case 21: ne[0] = 1; return 1;                                                /* layer_output_scale */
        default: die("expect_dims: unknown kind");
    }
    return 0;
}

/* --------------------------------------------------------------- tokenizer -- */
/* LlamaHfVocab (gguf-py/gguf/vocab.py:563-684), reproduced: tokens are the
 * model.vocab keys by id, score -1000.0, token_type BYTE for <0xNN>, CONTROL for
 * a special id, NORMAL otherwise; any added token with id >= len(model.vocab) is
 * appended in id order as USER_DEFINED (CONTROL if it is one of the specials). */
typedef struct {
    uint32_t n;                 /* tokens in the file (base + appended) */
    char  **tok; size_t *toklen;
    float *score; int32_t *type;
    char **merge; uint32_t n_merge;
    uint32_t bos, eos, unk, pad; int have_bos, have_eos, have_unk, have_pad;
    char *chat_template;
} tok_t;

static int is_byte_token(const char *s, size_t n) {
    if (n != 6) return 0;
    if (s[0] != '<' || s[1] != '0' || s[2] != 'x' || s[5] != '>') return 0;
    return isxdigit((unsigned char)s[3]) && isxdigit((unsigned char)s[4]);
}

static void tok_load(const char *hfdir, tok_t *T) {
    char path[4300];
    snprintf(path, sizeof path, "%s/tokenizer.json", hfdir);
    size_t len;
    char *txt = read_file(path, &len);
    const char *end = txt + len, *ms, *me, *vs, *ve, *p;

    if (!jget(txt, end, "model", &ms, &me)) die("tokenizer.json has no model");
    { const char *ts, *te;
      if (!jget(ms, me, "type", &ts, &te)) die("tokenizer.json: model.type missing");
      size_t sl; char *s = jstr_dec(ts, te, &sl); int bpe = !strcmp(s, "BPE"); free(s);
      if (!bpe) die("tokenizer.json: model.type is not BPE (LlamaHfVocab requires it)"); }
    if (!jget(ms, me, "vocab", &vs, &ve)) die("tokenizer.json has no model.vocab");

    /* count the entries */
    uint32_t nv = 0;
    for (p = jw(vs + 1, ve); p < ve && *p != '}'; ) {
        p = jstr_end(p, ve); p = jw(p, ve);
        if (p >= ve || *p != ':') die("vocab: ':' expected");
        p = jw(jskip(p + 1, ve), ve);
        if (p < ve && *p == ',') p = jw(p + 1, ve);
        nv++;
    }
    T->n = nv;
    T->tok = xmalloc(sizeof(char *) * nv); T->toklen = xmalloc(sizeof(size_t) * nv);
    T->score = xmalloc(sizeof(float) * nv); T->type = xmalloc(sizeof(int32_t) * nv);
    uint8_t *seen = xmalloc(nv);

    /* the special ids, so token_type can mark CONTROL (SparkVocab's special_ids) */
    /* first the added-token entries of tokenizer.json flagged special */
    uint8_t *is_special = xmalloc(nv);
    memset(seen, 0, nv); memset(is_special, 0, nv);

    const char *as, *ae;
    if (jget(txt, end, "added_tokens", &as, &ae) && *as == '[') {
        for (p = jw(as + 1, ae); p < ae && *p != ']'; ) {
            const char *os, *oe, *ks, *ke;
            os = p; oe = jskip(p, ae);
            if (jobj_get(os, oe, "id", &ks, &ke)) {
                uint64_t id = jnum_u64(ks, ke);
                const char *ss, *se;
                int sp = jobj_get(os, oe, "special", &ss, &se) && *ss == 't';
                const char *cs, *ce;
                if (sp && id < nv) is_special[id] = 1;
                if (id >= nv) {
                    /* appended token: mirror get_added_vocab() + the >= vocab_size filter */
                    if (!jobj_get(os, oe, "content", &cs, &ce)) die("added_tokens: content missing");
                    size_t sl; char *s = jstr_dec(cs, ce, &sl);
                    T->tok = realloc(T->tok, sizeof(char *) * (T->n + 1));
                    T->toklen = realloc(T->toklen, sizeof(size_t) * (T->n + 1));
                    T->score = realloc(T->score, sizeof(float) * (T->n + 1));
                    T->type = realloc(T->type, sizeof(int32_t) * (T->n + 1));
                    T->tok[T->n] = s; T->toklen[T->n] = sl;
                    T->score[T->n] = -1000.0f;
                    T->type[T->n] = sp ? TKN_CONTROL : TKN_USER_DEFINED;
                    T->n++;
                }
            }
            p = jw(oe, ae); if (p < ae && *p == ',') p = jw(p + 1, ae);
        }
    }

    /* the base vocab by id */
    for (p = jw(vs + 1, ve); p < ve && *p != '}'; ) {
        const char *ke = jstr_end(p, ve);
        size_t klen; char *k = jstr_dec(p, ve, &klen);
        (void)klen;
        p = jw(ke, ve);
        if (p >= ve || *p != ':') die("vocab: ':' expected");
        uint64_t id = jnum_u64(jw(p + 1, ve), jskip(p + 1, ve));
        if (id >= nv) die("vocab: id out of range");
        if (seen[id]) die("vocab: duplicate id");
        seen[id] = 1;
        T->tok[id] = k; T->toklen[id] = klen;
        T->score[id] = -1000.0f;
        T->type[id] = is_byte_token(k, klen) ? TKN_BYTE
                    : (is_special[id] ? TKN_CONTROL : TKN_NORMAL);
        p = jw(jskip(p + 1, ve), ve);
        if (p < ve && *p == ',') p = jw(p + 1, ve);
    }
    for (uint32_t i = 0; i < nv; i++) if (!seen[i]) die("vocab: missing id (not a bijection)");
    free(seen); free(is_special);

    /* merges. Two shapes are possible (vocab.py:197-222): a list of plain strings,
     * and — since transformers 4.45 — a list of [a, b] pairs whose parts may
     * contain spaces, which the reference escapes as U+0101 before joining with a
     * single space. Both are implemented; the pair form is what this checkpoint
     * carries (measured: 514906 pairs, no part contains a space). */
    T->n_merge = 0;
    {
        const char *rs, *re;
        if (jget(ms, me, "merges", &rs, &re) && *rs == '[') {
            uint32_t cap = 1024;
            T->merge = xmalloc(sizeof(char *) * cap);
            int escaped = 0;
            for (p = jw(rs + 1, re); p < re && *p != ']'; ) {
                if (T->n_merge == cap) { cap *= 2; T->merge = realloc(T->merge, sizeof(char *) * cap); }
                if (*p == '"') {                       /* the plain-string form */
                    size_t sl; T->merge[T->n_merge++] = jstr_dec(p, re, &sl);
                } else if (*p == '[') {                /* the pair form */
                    const char *pe = jskip(p, re);
                    const char *q = jw(p + 1, pe);
                    size_t l1, l2;
                    char *a = jstr_dec(q, pe, &l1);
                    q = jw(jskip(q, pe), pe);
                    if (q < pe && *q == ',') q = jw(q + 1, pe);
                    char *b = jstr_dec(q, pe, &l2);
                    /* escape spaces in both parts as U+0101, then join with ' ' */
                    size_t need = (l1 + l2) * 2 + 4, n = 0;
                    char *o = xmalloc(need);
                    for (int side = 0; side < 2; side++) {
                        const char *s = side ? b : a; size_t sl = side ? l2 : l1;
                        if (side) o[n++] = ' ';
                        for (size_t i = 0; i < sl; i++) {
                            if (s[i] == ' ') { o[n++] = (char)0xC4; o[n++] = (char)0x81; escaped++; }
                            else o[n++] = s[i];
                        }
                    }
                    o[n] = 0;
                    free(a); free(b);
                    T->merge[T->n_merge++] = o;
                } else {
                    die("merges: neither a string nor a pair");
                }
                p = jw(jskip(p, re), re);
                if (p < re && *p == ',') p = jw(p + 1, re);
            }
            if (escaped) fprintf(stderr, "[tokenizer] %d merge parts carried a space (escaped as U+0101)\n", escaped);
        }
    }

    /* the specials the loader needs by id, from tokenizer_config.json */
    T->have_bos = T->have_eos = T->have_unk = T->have_pad = 0;
    {
        char tp[4300];
        snprintf(tp, sizeof tp, "%s/tokenizer_config.json", hfdir);
        size_t tl; char *t2 = read_file(tp, &tl);
        const char *tend = t2 + tl;
        static const char *names[4] = { "bos_token", "eos_token", "unk_token", "pad_token" };
        uint32_t *slot[4] = { &T->bos, &T->eos, &T->unk, &T->pad };
        int *have[4] = { &T->have_bos, &T->have_eos, &T->have_unk, &T->have_pad };
        for (int i = 0; i < 4; i++) {
            const char *ss, *se;
            if (!jget(t2, tend, names[i], &ss, &se) || *ss != '"') continue;
            size_t sl; char *s = jstr_dec(ss, se, &sl);
            for (uint32_t id = 0; id < T->n; id++)
                if (T->toklen[id] == sl && !memcmp(T->tok[id], s, sl)) { *slot[i] = id; *have[i] = 1; break; }
            free(s);
        }
        /* chat_template.jinja, as SpecialVocab.add_to_gguf writes it */
        char cp[4300];
        snprintf(cp, sizeof cp, "%s/chat_template.jinja", hfdir);
        FILE *f = fopen(cp, "rb");
        if (f) {
            fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
            T->chat_template = xmalloc((size_t)sz + 1);
            if (fread(T->chat_template, 1, (size_t)sz, f) != (size_t)sz) die("short read of chat_template.jinja");
            T->chat_template[sz] = 0;
            fclose(f);
        }
        free(t2);
    }
    free(txt);
    fprintf(stderr, "[tokenizer] %u tokens (%u merges), bos %d eos %d unk %d pad %d%s\n",
            T->n, T->n_merge, T->have_bos ? (int)T->bos : -1, T->have_eos ? (int)T->eos : -1,
            T->have_unk ? (int)T->unk : -1, T->have_pad ? (int)T->pad : -1,
            T->chat_template ? ", chat template" : "");
}

/* the bool-array key the converter writes (conversion/gemma.py:681-682); the
 * vendored helper has no bool-array writer, and the loader prefers this form
 * over the scalar (llama-model.cpp:3902-3907), so it is written here. */
static void kv_arr_bool(ds4gw *W, const char *k, const uint8_t *v, uint64_t n) {
    ds4gw_put_str(W, k);
    ds4gw_put_u32(W, DS4GW_ARR);
    ds4gw_put_u32(W, DS4GW_BOOL);
    ds4gw_put_u64(W, n);
    ds4gw_put(W, v, n);
    W->nkv++;
}

/* the checkpoint name a scheduled GGUF tensor came from (the inverse of TMAP) */
static int ckpt_name_for(const hp_t *H, const char *gguf, char *out) {
    (void)H;
    if (!strncmp(gguf, "blk.", 4)) {
        int il; char suffix[128];
        if (sscanf(gguf, "blk.%d.%127s", &il, suffix) != 2) return 0;
        for (int kind = 0; kind < NTMAP; kind++) {
            if (!TMAP[kind].per_layer) continue;
            char gn[256];
            snprintf(gn, sizeof gn, TMAP[kind].gguf, il);
            if (!strcmp(gn, gguf)) {
                char cc[200];
                snprintf(cc, sizeof cc, TMAP[kind].ckpt, il);
                snprintf(out, 320, "language_model.%s", cc);
                return 1;
            }
        }
        return 0;
    }
    for (int kind = 0; kind < NTMAP; kind++) {
        if (TMAP[kind].per_layer) continue;
        if (!strcmp(TMAP[kind].gguf, gguf)) { snprintf(out, 320, "language_model.%s", TMAP[kind].ckpt); return 1; }
    }
    return 0;
}

/* -------------------------------------------------------------------- main -- */

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: eg2_gguf baseline <hf-dir> <out.gguf> [--limit N] [--only SUBSTR] [--no-tokenizer]\n");
        return 2;
    }
    const char *mode = argv[1], *hfdir = argv[2], *out = argc > 3 ? argv[3] : NULL;
    if (strcmp(mode, "baseline")) { fprintf(stderr, "eg2_gguf: unknown mode %s\n", mode); return 2; }
    if (!out) { fprintf(stderr, "eg2_gguf: output path required\n"); return 2; }

    long limit = 0; const char *only = NULL, *manifest = NULL; int want_tok = 1;
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--limit") && i + 1 < argc) limit = strtol(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "--only") && i + 1 < argc) only = argv[++i];
        else if (!strcmp(argv[i], "--manifest") && i + 1 < argc) manifest = argv[++i];
        else if (!strcmp(argv[i], "--no-tokenizer")) want_tok = 0;
        else { fprintf(stderr, "eg2_gguf: unknown argument %s\n", argv[i]); return 2; }
    }
    const int smoke = (limit > 0 || only != NULL);

    hp_t H; load_hparams(hfdir, &H);
    tok_t T; memset(&T, 0, sizeof T);
    if (want_tok) tok_load(hfdir, &T);
    else fprintf(stderr, "[tokenizer] skipped on request\n");

    v41_st S;
    if (v41_st_open(&S, hfdir)) return 1;

    ds4gw W;
    if (ds4gw_begin(&W, out)) return 1;

    /* ---- metadata: the key set of conversion/gemma.py:676-733 as the loader
     * reads it (llama-model.cpp:1382-1541). Name and file_type as the converter
     * writes them (base.py; gguf-py/gguf/constants.py:6274). ---- */
    ds4gw_kv_str(&W, "general.architecture", "gemma-embedding2");
    ds4gw_kv_str(&W, "general.name", smoke ? "EmbeddingGemma-2 (SMOKE-TEST-NOT-A-RELEASE)" : "EmbeddingGemma-2");
    ds4gw_kv_u32(&W, "general.file_type", 32);            /* MOSTLY_BF16 */
    ds4gw_kv_u32(&W, "general.alignment", 32);            /* the writer's default */

    char k[128];
#define KVK(name) (snprintf(k, sizeof k, "%s." name, ARCH), k)
    ds4gw_kv_u32(&W, KVK("block_count"), H.n_layer);
    ds4gw_kv_u32(&W, KVK("context_length"), H.n_ctx);
    ds4gw_kv_u32(&W, KVK("embedding_length"), H.n_embd);
    ds4gw_kv_u32(&W, KVK("embedding_length_out"), H.n_embd_out);
    ds4gw_kv_u32(&W, KVK("embedding_length_per_layer_input"), H.n_embd_pl);
    ds4gw_kv_u32(&W, KVK("feed_forward_length"), H.n_ff);
    ds4gw_kv_u32(&W, KVK("attention.head_count"), H.n_head);
    ds4gw_kv_u32(&W, KVK("attention.key_length"), H.head_dim_full);
    ds4gw_kv_u32(&W, KVK("attention.value_length"), H.head_dim_full);
    ds4gw_kv_u32(&W, KVK("attention.key_length_swa"), H.head_dim_swa);
    ds4gw_kv_u32(&W, KVK("attention.value_length_swa"), H.head_dim_swa);
    ds4gw_kv_f32(&W, KVK("attention.layer_norm_rms_epsilon"), H.rms_eps);
    /* HF's sliding_window is one-sided; the converter doubles it so the loader's
     * symmetric mask (llama-hparams.h:524-533, |p1-p0| > n_swa/2) reproduces it
     * (conversion/gemma.py:824-825). */
    ds4gw_kv_u32(&W, KVK("attention.sliding_window"), 2 * H.n_swa_hf);
    kv_arr_bool(&W, KVK("attention.sliding_window_pattern"), H.is_swa, H.n_layer);
    ds4gw_kv_bool(&W, KVK("attention.causal"), 0);
    ds4gw_kv_f32(&W, KVK("rope.freq_base"), H.rope_base);
    ds4gw_kv_f32(&W, KVK("rope.freq_base_swa"), H.rope_base_swa);
    ds4gw_kv_u32(&W, KVK("rope.dimension_count"), H.head_dim_full);
    ds4gw_kv_u32(&W, KVK("rope.dimension_count_swa"), H.head_dim_swa);
    ds4gw_kv_u32(&W, KVK("vocab_size"), H.n_vocab);
    { uint32_t *kvarr = xmalloc(sizeof(uint32_t) * H.n_layer);
      for (uint32_t i = 0; i < H.n_layer; i++) kvarr[i] = H.is_swa[i] ? H.n_head_kv_swa : H.n_head_kv_full;
      ds4gw_kv_arr_u32(&W, KVK("attention.head_count_kv"), kvarr, H.n_layer);
      free(kvarr); }
    ds4gw_kv_u32(&W, KVK("pooling_type"), 1);   /* PoolingType.MEAN */

    if (want_tok) {
        /* The converter's own key set for this arch (conversion/gemma.py:641-676):
         * model "gemma4", the token list, scores and types, then merges and the
         * special ids. It writes no tokenizer.ggml.pre — the loader pins
         * tokenizer_pre = "gemma4" for this model type itself
         * (src/llama-vocab.cpp:2275), so the file does not carry one. */
        ds4gw_kv_str(&W, "tokenizer.ggml.model", "gemma4");
        ds4gw_kv_arr_str(&W, "tokenizer.ggml.tokens", (const char *const *)T.tok, T.n);
        ds4gw_kv_arr_f32(&W, "tokenizer.ggml.scores", T.score, T.n);
        ds4gw_kv_arr_i32(&W, "tokenizer.ggml.token_type", T.type, T.n);
        if (T.n_merge) ds4gw_kv_arr_str(&W, "tokenizer.ggml.merges", (const char *const *)T.merge, T.n_merge);
        if (T.have_bos) ds4gw_kv_u32(&W, "tokenizer.ggml.bos_token_id", T.bos);
        if (T.have_eos) ds4gw_kv_u32(&W, "tokenizer.ggml.eos_token_id", T.eos);
        if (T.have_unk) ds4gw_kv_u32(&W, "tokenizer.ggml.unknown_token_id", T.unk);
        if (T.have_pad) ds4gw_kv_u32(&W, "tokenizer.ggml.padding_token_id", T.pad);
        ds4gw_kv_bool(&W, "tokenizer.ggml.add_bos_token", 1);
        ds4gw_kv_bool(&W, "tokenizer.ggml.add_space_prefix", 0);
        if (T.chat_template) ds4gw_kv_str(&W, "tokenizer.chat_template", T.chat_template);
    }

    /* ---- plan the tensor table ---- */
    uint64_t total_bytes = 0;
    int nt = 0, nt_cap = 0;
    uint8_t *widen_flags = NULL;
    for (int kind = 0; kind < NTMAP; kind++) {
        int reps = TMAP[kind].per_layer ? (int)H.n_layer : 1;
        for (int il = 0; il < reps; il++) {
            char ck[256], gn[256];
            if (TMAP[kind].per_layer) { snprintf(ck, sizeof ck, TMAP[kind].ckpt, il); snprintf(gn, sizeof gn, TMAP[kind].gguf, il); }
            else { snprintf(ck, sizeof ck, "%s", TMAP[kind].ckpt); snprintf(gn, sizeof gn, "%s", TMAP[kind].gguf); }
            char full[320];
            snprintf(full, sizeof full, "language_model.%s", ck);
            if (only && !strstr(full, only)) continue;
            if (limit && nt >= limit) break;

            const v41_st_ent *e = v41_st_find(&S, full);
            if (!e) { fprintf(stderr, "eg2_gguf: checkpoint has no %s\n", full); return 1; }
            if (strcmp(e->dtype, "BF16")) {
                fprintf(stderr, "eg2_gguf: %s is %s, the baseline writes BF16 only\n", full, e->dtype);
                return 1;
            }
            uint64_t ne[4]; uint32_t nd = expect_dims(&H, kind, il, ne);
            /* the checkpoint's own shape must be the declared shape, reversed */
            if ((uint32_t)e->nd != nd) {
                fprintf(stderr, "eg2_gguf: %s rank %d != declared %u\n", full, e->nd, nd);
                return 1;
            }
            for (uint32_t d = 0; d < nd; d++) {
                int64_t want = (int64_t)ne[nd - 1 - d];
                if (e->shape[d] != want) {
                    fprintf(stderr, "eg2_gguf: %s shape[%u] %lld != declared %lld\n",
                            full, d, (long long)e->shape[d], (long long)want);
                    return 1;
                }
            }
            uint64_t elems = 1;
            for (uint32_t d = 0; d < nd; d++) elems *= ne[d];
            uint64_t src_bytes = elems * 2;                       /* the checkpoint is BF16 */
            if (src_bytes != e->nbytes) {
                fprintf(stderr, "eg2_gguf: %s carries %llu B, the declared shape needs %llu\n",
                        full, (unsigned long long)e->nbytes, (unsigned long long)src_bytes);
                return 1;
            }
            /* The converter's own rule (conversion/base.py:1244-1246): 1-D tensors
             * and every *_norm.weight are written F32 even in a BF16 conversion,
             * "most of the codebase that takes in 1D tensors or norms only handles
             * F32". llama.cpp's CPU backend enforces it the hard way — a BF16 1-D
             * operand in a binary op aborts (ggml binary-ops: "unsupported types:
             * dst: f32, src0: f32, src1: bf16", measured this session on the
             * all-BF16 variant, which is why this rule exists here). */
            size_t glen = strlen(gn);
            int widen = (nd <= 1) || (glen > 11 && !strcmp(gn + glen - 11, "_norm.weight"));
            uint64_t bytes = widen ? elems * 4 : src_bytes;
            int ti = ds4gw_tensor(&W, gn, widen ? GGT_F32 : GGT_BF16, nd, ne, bytes);
            if (ti + 1 > nt_cap) { nt_cap = nt_cap ? nt_cap * 2 : 512; widen_flags = realloc(widen_flags, nt_cap); if (!widen_flags) die("out of memory"); }
            widen_flags[ti] = (uint8_t)widen;
            total_bytes += bytes;
            nt++;
        }
        if (limit && nt >= limit) break;
    }
    fprintf(stderr, "[plan] %d tensors, %.3f GB of tensor data\n", nt, (double)total_bytes / 1e9);
    if (ds4gw_header(&W)) return 1;

    /* ---- the manifest: what the writer put where, per tensor, so the reader's
     * dump and the safetensors source can be joined against it field by field.
     * Absolutes are the header's data start plus the registered offset. ---- */
    if (manifest) {
        FILE *mf = fopen(manifest, "w");
        if (!mf) { fprintf(stderr, "eg2_gguf: cannot write %s\n", manifest); return 1; }
        fprintf(mf, "#gguf_name\ttype\tnd\tdims\tbytes\tgguf_abs\tckpt_name\tst_file\tst_abs\tst_dtype\n");
        for (int i = 0; i < nt; i++) {
            const ds4gw_tinfo *t = &W.t[i];
            char ck[256] = {0};
            if (!ckpt_name_for(&H, t->name, ck)) { fprintf(stderr, "eg2_gguf: no checkpoint name for %s\n", t->name); return 1; }
            const v41_st_ent *e = v41_st_find(&S, ck);
            if (!e) { fprintf(stderr, "eg2_gguf: checkpoint has no %s\n", ck); return 1; }
            fprintf(mf, "%s\t%u\t%u\t", t->name, t->type, t->nd);
            for (uint32_t d = 0; d < t->nd; d++) fprintf(mf, "%s%llu", d ? "," : "", (unsigned long long)t->ne[d]);
            fprintf(mf, "\t%llu\t%llu\t%s\t%s\t%llu\t%s\n",
                    (unsigned long long)t->nbytes, (unsigned long long)(W.data0 + t->off), ck,
                    S.sh[e->shard].file, (unsigned long long)(S.sh[e->shard].data0 + e->off),
                    widen_flags[i] ? "bf16->f32" : "bf16");
        }
        fclose(mf);
        fprintf(stderr, "[manifest] %s\n", manifest);
    }

    /* ---- write the bytes, in registration order, straight from the mapping ---- */
    for (int i = 0; i < nt; i++) {
        const ds4gw_tinfo *t = &W.t[i];
        char ck[320];
        if (!ckpt_name_for(&H, t->name, ck)) { fprintf(stderr, "eg2_gguf: no checkpoint name for %s\n", t->name); return 1; }
        const v41_st_ent *e = v41_st_find(&S, ck);
        if (!e) { fprintf(stderr, "eg2_gguf: checkpoint has no %s\n", ck); return 1; }
        const uint8_t *d = v41_st_data(&S, e);
        if (!d) return 1;
        if (widen_flags[i]) {
            /* BF16 -> F32, the widening the reference applies: the BF16 bit
             * pattern becomes the F32 word's high half (ds4_bf16_to_f32,
             * /data/YoungAi/src/common/ds4_float.h:34). */
            uint64_t n = t->nbytes / 4;
            uint8_t *buf = xmalloc(t->nbytes);
            for (uint64_t k = 0; k < n; k++) {
                uint32_t v = ((uint32_t)d[2 * k] | ((uint32_t)d[2 * k + 1] << 8)) << 16;
                memcpy(buf + 4 * k, &v, 4);
            }
            int rc = ds4gw_write(&W, i, buf, t->nbytes);
            free(buf);
            if (rc) return 1;
        } else if (ds4gw_write(&W, i, d, t->nbytes)) {
            return 1;
        }
        if ((i + 1) % 64 == 0) { fprintf(stderr, "[write] %d/%d\n", i + 1, nt); fflush(stderr); }
    }
    if (ds4gw_end(&W)) return 1;
    v41_st_close(&S);
    fprintf(stderr, "[done] %s: %d tensors, %.3f GB\n", out, nt, (double)total_bytes / 1e9);
    free(widen_flags);
    return 0;
}
