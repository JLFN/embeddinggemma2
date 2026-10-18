/* eg2_gpuprobe.cu — decides BY MEASUREMENT how the reference VQ GPU kernels have
 * to be driven on this machine's GPU (RTX 4070 SUPER, sm_89, CUDA 13.3).
 *
 * Why this file exists. The reference kernels in
 * /data/YoungAi @ a4e6d31 gguf-tools/quantize/vq_gpu.cu take host pointers
 * directly and move nothing: vq_gpu.cu:8 states the assumption in its own words,
 * "GB10 统一内存: malloc 指针直传, 无显式搬运" (on the GB10 box the host malloc'd
 * buffers are device-accessible, so no copy and no pinning is needed). Whether
 * that holds on a discrete PCIe card is a property of the machine, not of the
 * code, and it cannot be assumed in either direction. It also fails quietly:
 * vq_gpu.cu:26-28 vqg_last_err() prints at most four sticky errors and returns,
 * so a kernel that cannot reach its buffers does not stop a run.
 *
 * The probe therefore checks every index the kernel returns against a CPU
 * reference computed the way gguf-tools/quantize/vq_qc.h:56-79 vq_assign_q
 * computes it: c2[c] = row energy of the codebook (double sum, cast to float),
 * val = c2[c] - 2*v.c with the dot product accumulated in double and cast to
 * float, and a strict < so the first index wins (that tie rule is what makes the
 * CPU and GPU paths agree, see vq_qc.h:68 and vq_gpu.cu:73).
 *
 * Modes, one per process — a kernel that faults poisons the CUDA context, so a
 * healthy mode must never run after a faulting one:
 *   attr                     device attributes and the shared-memory gate only
 *   shm                      vqg_shm_ok for the tiers this project uses
 *   assign   malloc|pinned|managed    vqg_assign over 200000 vectors (the
 *                                     reference's kmeans subsample cap,
 *                                     vq_qc.h:90)
 *   gptq     malloc|pinned|managed    vqg_gptq_full with has_fb = 0 over a
 *                                     1024x4096 matrix — the no-feedback call
 *                                     vq_qc.h:157 makes on the nocal path
 *                                     (X = NULL), which is the path eg2_pack
 *                                     uses
 *   all      the three allocation modes of assign, in three child processes
 *
 * Build (the reference's own flags for vq_gpu.cu; the Makefile rule at
 * /data/YoungAi/gguf-tools/Makefile:197 uses exactly "nvcc -O3 -arch=native -c"):
 *   nvcc -O3 -arch=native -c /data/YoungAi/gguf-tools/quantize/vq_gpu.cu -o out/vq_gpu.o
 *   nvcc -O3 -arch=native -o bin/eg2_gpuprobe tools/eg2_gpuprobe.cu out/vq_gpu.o -lcublas -lcudart
 */
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>

extern "C" {
int  vqg_ready(void);
int  vqg_shm_ok(int nc, int dim);
void vqg_assign(const float *V, int nv, const float *C, int nc, int dim, int *idx);
void vqg_gptq_full(float *Wk, const float *C, const double *Hi_all_d,
                   int rows, int cols, int grp, int dim, int nc,
                   int has_fb, int *sidx, const float *colw_h);
int  vqg_host_alloc(void **p, size_t bytes);
void vqg_host_free(void *p);
int  vqg_alloc_managed(void **p, size_t bytes);
int  vqg_unpack_dequant(const uint8_t *pay, float *W, int rows, int cols, int dim, int nc, int nbit);
}

static const char *g_mode = "malloc";

static double now_s(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }

/* Deterministic integer LCG, the same generator and constant as the known-answer
 * test in tools/eg2_vq.c, so probe data is reproducible run to run. */
static uint64_t g_seed = 0x123456789abcdef0ULL;
static float rnd(void) {
    g_seed = g_seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return (float)(int32_t)(g_seed >> 33) / 2147483648.0f;
}

static void *host_alloc(size_t bytes) {
    void *p = NULL;
    if (!strcmp(g_mode, "malloc")) {
        p = malloc(bytes);
    } else if (!strcmp(g_mode, "pinned")) {
        if (!vqg_host_alloc(&p, bytes)) { fprintf(stderr, "[probe] cudaMallocHost(%zu) failed\n", bytes); exit(3); }
    } else if (!strcmp(g_mode, "managed")) {
        if (!vqg_alloc_managed(&p, bytes)) { fprintf(stderr, "[probe] cudaMallocManaged(%zu) failed\n", bytes); exit(3); }
    } else {
        fprintf(stderr, "[probe] unknown allocation mode %s\n", g_mode); exit(3);
    }
    if (!p) { fprintf(stderr, "[probe] host_alloc(%zu) returned NULL\n", bytes); exit(3); }
    return p;
}
static void host_free(void *p) {
    if (!strcmp(g_mode, "malloc")) free(p);
    else if (!strcmp(g_mode, "pinned")) vqg_host_free(p);
    else cudaFree(p);
}

/* CPU reference: vq_qc.h:56-79 vq_assign_q semantics, scalar form. */
static void cpu_assign(const float *V, int nv, const float *C, int nc, int dim, int *idx) {
    float *c2 = (float *)malloc((size_t)nc * 4);
    for (int c = 0; c < nc; c++) {
        double s = 0;
        for (int d = 0; d < dim; d++) s += (double)C[c * dim + d] * C[c * dim + d];
        c2[c] = (float)s;
    }
    for (int i = 0; i < nv; i++) {
        const float *v = V + (size_t)i * dim;
        float best = 3.4e38f; int bi = 0;
        for (int c = 0; c < nc; c++) {
            double acc = 0;
            for (int d = 0; d < dim; d++) acc += (double)v[d] * C[c * dim + d];
            const float val = c2[c] - 2.0f * (float)acc;
            if (val < best) { best = val; bi = c; }
        }
        idx[i] = bi;
    }
    free(c2);
}

/* Cosine between a matrix and its codebook reconstruction, where every `dim`
 * consecutive weights take the codebook row their index points at. Row gains are
 * 1.0 on the nocal path (vq_qc.h:236), so no gain factor belongs here. */
static double recon_cos(const float *W, const float *C, const int *idx, size_t n, int dim) {
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < n / dim; i++) {
        const float *q = C + (size_t)idx[i] * dim;
        for (int d = 0; d < dim; d++) {
            const double a = W[i * dim + d], b = q[d];
            dot += a * b; na += a * a; nb += b * b;
        }
    }
    return dot / (sqrt(na) * sqrt(nb) + 1e-30);
}

static void device_report(void) {
    int dev = 0; cudaGetDevice(&dev);
    cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
    struct { const char *n; cudaDeviceAttr a; } attrs[] = {
        { "pageableMemoryAccess",         cudaDevAttrPageableMemoryAccess },
        { "concurrentManagedAccess",      cudaDevAttrConcurrentManagedAccess },
        { "unifiedAddressing",            cudaDevAttrUnifiedAddressing },
        { "canMapHostMemory",             cudaDevAttrCanMapHostMemory },
        { "directManagedMemAccessFromHost", cudaDevAttrDirectManagedMemAccessFromHost },
        { "maxSharedMemoryPerBlock",      cudaDevAttrMaxSharedMemoryPerBlock },
        { "maxSharedMemoryPerBlockOptin", cudaDevAttrMaxSharedMemoryPerBlockOptin },
        { "multiProcessorCount",          cudaDevAttrMultiProcessorCount },
    };
    printf("[attr] %s capability %d.%d  smem/block %zu (optin %zu)  sm %d  cuda %d.%d\n",
           p.name, p.major, p.minor, p.sharedMemPerBlock, p.sharedMemPerBlockOptin,
           p.multiProcessorCount, CUDART_VERSION / 1000, (CUDART_VERSION % 1000) / 10);
    for (size_t i = 0; i < sizeof attrs / sizeof attrs[0]; i++) {
        int v = -1; const cudaError_t e = cudaDeviceGetAttribute(&v, attrs[i].a, dev);
        printf("[attr] %-32s %s%d\n", attrs[i].n, e == cudaSuccess ? "" : "query failed, value ", v);
    }
}

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "attr";
    if (argc > 2) g_mode = argv[2];

    device_report();
    printf("[probe] vqg_ready=%d\n", vqg_ready());

    if (!strcmp(mode, "attr")) return 0;

    if (!strcmp(mode, "shm")) {
        const int tiers[][2] = { {512,4}, {256,8}, {1024,4}, {512,8}, {1024,8}, {4096,8} };
        for (size_t i = 0; i < sizeof tiers / sizeof tiers[0]; i++)
            printf("[shm] nc=%-5d dim=%d  dynamic shared bytes=%6zu  vqg_shm_ok=%d\n",
                   tiers[i][0], tiers[i][1], (size_t)(tiers[i][0] * tiers[i][1] + tiers[i][0]) * 4,
                   vqg_shm_ok(tiers[i][0], tiers[i][1]));
        return 0;
    }

    if (!strcmp(mode, "assign")) {
        const int nc = 512, dim = 4, nv = 200000;   /* vq_qc.h:90 subsample cap */
        float *V = (float *)host_alloc((size_t)nv * dim * 4);
        float *C = (float *)host_alloc((size_t)nc * dim * 4);
        int   *gi = (int *)host_alloc((size_t)nv * sizeof(int));
        int   *ci = (int *)malloc((size_t)nv * sizeof(int));
        for (size_t i = 0; i < (size_t)nv * dim; i++) V[i] = rnd() * 0.05f;
        for (int c = 0; c < nc; c++)                 /* seed the codebook from a strided subsample, vq_qc.h:93 */
            for (int d = 0; d < dim; d++) C[c * dim + d] = V[(size_t)(c * 7) % (size_t)nv * dim + d];
        for (int i = 0; i < nv; i++) gi[i] = -1;

        const double t0 = now_s();
        cpu_assign(V, nv, C, nc, dim, ci);
        const double t_cpu = now_s() - t0;

        const double t1 = now_s();
        vqg_assign(V, nv, C, nc, dim, gi);
        const cudaError_t sync = cudaStreamSynchronize(0);
        cudaError_t last = cudaGetLastError();
        const double t_gpu = now_s() - t1;

        int unwritten = 0, mismatch = 0, tie = 0;
        double max_dval = 0;
        for (int i = 0; i < nv; i++) {
            if (gi[i] < 0 || gi[i] >= nc) { unwritten++; continue; }
            if (gi[i] == ci[i]) continue;
            mismatch++;
            /* how far from a genuine tie: compare the CPU criterion between the
             * two picks. A tiny gap means a first-index tie decided in float. */
            const float *v = V + (size_t)i * dim;
            double ba = 0, bb = 0;
            for (int d = 0; d < dim; d++) { ba += (double)v[d] * C[ci[i] * dim + d]; bb += (double)v[d] * C[gi[i] * dim + d]; }
            double c2a = 0, c2b = 0;
            for (int d = 0; d < dim; d++) { c2a += (double)C[ci[i] * dim + d] * C[ci[i] * dim + d]; c2b += (double)C[gi[i] * dim + d] * C[gi[i] * dim + d]; }
            const double dval = fabs((c2a - 2.0 * ba) - (c2b - 2.0 * bb));
            if (dval > max_dval) max_dval = dval;
            if (dval < 1e-4) tie++;
        }
        printf("[assign:%s] nv=%d nc=%d dim=%d  sync=%s last=%s\n", g_mode, nv, nc, dim,
               cudaGetErrorName(sync), cudaGetErrorName(last));
        printf("[assign:%s] cpu %.3fs  gpu %.3fs  speedup %.1fx\n", g_mode, t_cpu, t_gpu, t_cpu / t_gpu);
        printf("[assign:%s] indices unwritten=%d mismatch=%d (of which within 1e-4 of a tie: %d) max|dval|=%.3g\n",
               g_mode, unwritten, mismatch, tie, max_dval);
        printf("[assign:%s] reconstruction cos: cpu=%.6f gpu=%.6f\n", g_mode,
               recon_cos(V, C, ci, (size_t)nv * dim, dim), recon_cos(V, C, gi, (size_t)nv * dim, dim));
        host_free(V); host_free(C); host_free(gi); free(ci);
        return (unwritten || mismatch) ? 1 : 0;
    }

    if (!strcmp(mode, "gptq")) {
        const int nc = 512, dim = 4, grp = 128, rows = 1024, cols = 4096;
        const size_t N = (size_t)rows * cols;
        float *W  = (float *)host_alloc(N * 4);          /* source, read-only for the kernels */
        float *Wk = (float *)host_alloc(N * 4);          /* the kernel's working copy */
        float *C  = (float *)host_alloc((size_t)nc * dim * 4);
        int   *gi = (int *)host_alloc((N / dim) * sizeof(int));
        int   *ci = (int *)malloc((N / dim) * sizeof(int));
        for (size_t i = 0; i < N; i++) W[i] = rnd() * 0.05f;
        memcpy(Wk, W, N * 4);
        for (int c = 0; c < nc; c++)
            for (int d = 0; d < dim; d++) C[c * dim + d] = W[(size_t)(c * 13) % (size_t)(N / dim) * dim + d];
        for (size_t i = 0; i < N / dim; i++) gi[i] = -1;

        /* CPU reference = the same per-segment nearest neighbour the kernel does
         * with feedback off (vq_gpu.cu:272-282 feedback block is skipped when Hi
         * is NULL), so segment by segment over the whole matrix. */
        const double t0 = now_s();
        for (int r = 0; r < rows; r++)
            cpu_assign(W + (size_t)r * cols, cols / dim, C, nc, dim, ci + (size_t)r * (cols / dim));
        const double t_cpu = now_s() - t0;

        const double t1 = now_s();
        vqg_gptq_full(Wk, C, NULL, rows, cols, grp, dim, nc, 0, gi, NULL);
        const cudaError_t sync = cudaStreamSynchronize(0);
        cudaError_t last = cudaGetLastError();
        const double t_gpu = now_s() - t1;

        int unwritten = 0, mismatch = 0, tie = 0;
        double max_dval = 0;
        for (size_t i = 0; i < N / dim; i++) {
            if (gi[i] < 0 || gi[i] >= nc) { unwritten++; continue; }
            if (gi[i] == ci[i]) continue;
            mismatch++;
            const size_t off = i * dim;
            double ba = 0, bb = 0, c2a = 0, c2b = 0;
            for (int d = 0; d < dim; d++) {
                ba += (double)W[off + d] * C[ci[i] * dim + d]; bb += (double)W[off + d] * C[gi[i] * dim + d];
                c2a += (double)C[ci[i] * dim + d] * C[ci[i] * dim + d]; c2b += (double)C[gi[i] * dim + d] * C[gi[i] * dim + d];
            }
            const double dval = fabs((c2a - 2.0 * ba) - (c2b - 2.0 * bb));
            if (dval > max_dval) max_dval = dval;
            if (dval < 1e-4) tie++;
        }
        printf("[gptq:%s] %dx%d nc=%d dim=%d grp=%d fb=0  sync=%s last=%s\n", g_mode, rows, cols, nc, dim, grp,
               cudaGetErrorName(sync), cudaGetErrorName(last));
        printf("[gptq:%s] cpu %.3fs  gpu %.3fs  speedup %.1fx\n", g_mode, t_cpu, t_gpu, t_cpu / t_gpu);
        printf("[gptq:%s] indices unwritten=%d mismatch=%d (of which within 1e-4 of a tie: %d) max|dval|=%.3g\n",
               g_mode, unwritten, mismatch, tie, max_dval);
        printf("[gptq:%s] reconstruction cos: cpu=%.6f gpu=%.6f\n", g_mode,
               recon_cos(W, C, ci, N, dim), recon_cos(W, C, gi, N, dim));
        host_free(W); host_free(Wk); host_free(C); host_free(gi); free(ci);
        return (unwritten || mismatch) ? 1 : 0;
    }

    if (!strcmp(mode, "budget")) {
        /* Where the GPU build's per-tensor time actually goes. Same geometry as a
         * text/audio 2.25 bpw matrix (1024x4096, dim 4, nc 512, group 128), the
         * stages in the order vq_encode_full and emit_payload run them, so the
         * parts can be summed and compared against a measured per-tensor wall
         * time instead of guessed at. */
        const int nc = 512, dim = 4, grp = 128, rows = 1024, cols = 4096;
        const size_t N = (size_t)rows * cols;
        const int reps = 5;
        double t;
        printf("[budget] geometry %dx%d dim=%d nc=%d (%.1f MB f32), %d reps each\n",
               rows, cols, dim, nc, N * 4 / 1e6, reps);

        t = now_s();
        for (int k = 0; k < reps; k++) { void *p = NULL; vqg_host_alloc(&p, N * 4); vqg_host_free(p); }
        printf("[budget]   cudaMallocHost+Free of %.1f MB:        %8.2f ms each\n", N * 4 / 1e6, 1000 * (now_s() - t) / reps);

        void *pl = malloc(N * 4);
        t = now_s();
        for (int k = 0; k < reps; k++) { void *p = NULL; vqg_host_alloc(&p, N * 4); memcpy(p, pl, N * 4); vqg_host_free(p); }
        printf("[budget]   + memcpy pageable -> pinned:         %8.2f ms each\n", 1000 * (now_s() - t) / reps);

        float *Wk = (float *)host_alloc(N * 4), *C = (float *)host_alloc((size_t)nc * dim * 4);
        int *sidx = (int *)host_alloc((N / dim) * sizeof(int));
        for (size_t i = 0; i < N; i++) Wk[i] = rnd() * 0.05f;
        for (int c = 0; c < nc * dim; c++) C[c] = rnd() * 0.05f;
        t = now_s();
        for (int k = 0; k < reps; k++) vqg_gptq_full(Wk, C, NULL, rows, cols, grp, dim, nc, 0, sidx, NULL);
        printf("[budget]   vqg_gptq_full (fb=0, the nocal path): %8.2f ms each  (cpu reference sidebar)\n", 1000 * (now_s() - t) / reps);

        const int nsub = 200000;
        float *sub = (float *)host_alloc((size_t)nsub * dim * 4);
        int *aidx = (int *)host_alloc((size_t)nsub * sizeof(int));
        for (size_t i = 0; i < (size_t)nsub * dim; i++) sub[i] = rnd() * 0.05f;
        t = now_s();
        for (int k = 0; k < reps; k++) for (int it = 0; it < 8; it++) vqg_assign(sub, nsub, C, nc, dim, aidx);
        printf("[budget]   kmeans assign, 8 rounds x %d vectors:%8.2f ms each pk\n", nsub, 1000 * (now_s() - t) / reps);

        /* the reader path: a payload of this matrix's size, decoded into pinned memory */
        const size_t pay = (size_t)16 + (size_t)nc * dim * 2 + (size_t)rows * 2 + (size_t)N / dim;
        uint8_t *buf = (uint8_t *)host_alloc(pay);
        memset(buf, 0, 16);
        uint32_t mg = 0x51565144u; memcpy(buf, &mg, 4);
        uint16_t d16 = dim, n16 = nc; memcpy(buf + 4, &d16, 2); memcpy(buf + 6, &n16, 2);
        uint32_t r32 = rows, c32 = cols; memcpy(buf + 8, &r32, 4); memcpy(buf + 12, &c32, 4);
        float *Wout = (float *)host_alloc(N * 4);
        t = now_s();
        for (int k = 0; k < reps; k++) vqg_unpack_dequant(buf, Wout, rows, cols, dim, nc, 9);
        printf("[budget]   vqg_unpack_dequant into pinned (%.1f MB out): %6.2f ms each\n", N * 4 / 1e6, 1000 * (now_s() - t) / reps);

        /* CPU-side cost of the same 16 MB in pageable memory, for reference */
        float *Wp = (float *)pl, *Op = (float *)malloc(N * 4);
        t = now_s();
        for (int k = 0; k < reps; k++) memcpy(Op, Wp, N * 4);
        printf("[budget]   memcpy %.1f MB pageable -> pageable:  %8.2f ms (cpu reference)\n", N * 4 / 1e6, 1000 * (now_s() - t) / reps);
        /* CPU-side cost of touching the two kinds of memory, because the encoder
         * does most of its work on the CPU even in the GPU build: the kmeans
         * accumulates over the subsample 8 times and the index-extraction loop
         * reads the kernel's index output element by element. Each measurement
         * ends in a printed checksum, otherwise the compiler deletes the loop (the
         * earlier version of this benchmark did exactly that: it reported 0.00 ms
         * for a 16.8 MB memcpy). */
        float *A = (float *)malloc(N * 4), *Bp = (float *)malloc(N * 4), *P = (float *)host_alloc(N * 4);
        for (size_t i = 0; i < N; i++) A[i] = (float)(i & 1023);
        double chk = 0;
        t = now_s(); for (int k = 0; k < reps; k++) memcpy(Bp, A, N * 4);
        const double ms_pp = 1000 * (now_s() - t) / reps;
        t = now_s(); for (int k = 0; k < reps; k++) memcpy(P, A, N * 4);
        const double ms_to_pin = 1000 * (now_s() - t) / reps;
        t = now_s(); for (int k = 0; k < reps; k++) memcpy(Bp, P, N * 4);
        const double ms_from_pin = 1000 * (now_s() - t) / reps;
        t = now_s(); for (int k = 0; k < reps; k++) { const float *p = A; for (size_t i = 0; i < N; i++) chk += p[i]; }
        const double ms_sum_pageable = 1000 * (now_s() - t) / reps;
        t = now_s(); for (int k = 0; k < reps; k++) { const float *p = P; for (size_t i = 0; i < N; i++) chk += p[i]; }
        const double ms_sum_pinned = 1000 * (now_s() - t) / reps;
        t = now_s(); for (int k = 0; k < reps; k++) for (size_t i = 0; i < N / 4; i++) { const float v = (float)i; memcpy(Bp + 4 * i, &v, 16); }
        const double ms_scat_pageable = 1000 * (now_s() - t) / reps;
        t = now_s(); for (int k = 0; k < reps; k++) for (size_t i = 0; i < N / 4; i++) { const float v = (float)i; memcpy(P + 4 * i, &v, 16); }
        const double ms_scat_pinned = 1000 * (now_s() - t) / reps;
        printf("[budget]   memcpy %.1f MB pageable->pageable:  %8.2f ms\n", N * 4 / 1e6, ms_pp);
        printf("[budget]   memcpy %.1f MB pageable->pinned:    %8.2f ms\n", N * 4 / 1e6, ms_to_pin);
        printf("[budget]   memcpy %.1f MB pinned->pageable:    %8.2f ms\n", N * 4 / 1e6, ms_from_pin);
        printf("[budget]   seq read %.1f MB pageable:          %8.2f ms   (kmeans accumulation pattern)\n", N * 4 / 1e6, ms_sum_pageable);
        printf("[budget]   seq read %.1f MB pinned:            %8.2f ms\n", N * 4 / 1e6, ms_sum_pinned);
        printf("[budget]   16B scattered writes pageable:      %8.2f ms   (index extraction pattern)\n", ms_scat_pageable);
        printf("[budget]   16B scattered writes pinned:        %8.2f ms\n", ms_scat_pinned);
        printf("[budget]   (checksum %.6g, printed so nothing above is optimised away)\n", chk);
        free(A); free(Bp); host_free(P);
        free(Op); free(pl);
        host_free(Wk); host_free(C); host_free(sidx); host_free(sub); host_free(aidx); host_free(buf); host_free(Wout);
        return 0;
    }

    fprintf(stderr, "unknown mode %s\n", mode);
    return 2;
}
