#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <inttypes.h>
#include <assert.h>

/* ========================= USER-PROVIDED FUNCTIONS (不改) ========================= */
float rand49()
{
    static int Num=0;
    double number;
    int i;
    i=rand();
    number=(double)(i)/((unsigned) (RAND_MAX+1));
    Num++;
    if (Num >=RAND_MAX){
        time_t t;
        t=time(NULL);
        Num=0;
        (void)t;
    }
    return (float)number;
}

double Normal()
{
    static int iset = 0;
    static double qset;
    double vx, vy, r, temp;
    if (iset == 0) {
        do {
            vx = 2.0 * rand49() - 1.0;
            vy = 2.0 * rand49() - 1.0;
            r = vx * vx + vy * vy;
        } while (r >= 1.0 || r == 0);
        temp = sqrt(-2.0 * log(r) / r);
        qset = vy * temp;
        iset = 1;
        return (vx * temp);
    } else {
        iset = 0;
        return qset;
    }
}

double phi_func(double a)
{
    double result;
    double absolute;
    double tmp;
    absolute = fabs(a);
    if (absolute < pow(10,-16))
        result = 37.42995;
    else if (absolute > 37.43)
        result = 1.110223e-16;
    else {
        tmp=(exp(a)-1)/(exp(a)+1);
        tmp=fabs(tmp);
        result= -log(tmp);
    }
    return result;
}
/* ========================= END USER-PROVIDED FUNCTIONS ========================= */

/* ========================= Minimal aligned alloc shim (just malloc/free) ========================= */
static void *xaligned_alloc(size_t align, size_t size)
{
    (void)align;
    if (size == 0) size = 1;
    void *p = malloc(size);
    if (!p) { fprintf(stderr, "OOM in xaligned_alloc(%zu)\n", size); exit(1); }
    return p;
}
static void xaligned_free(void *p)
{
    free(p);
}

/* ========================= Tanner graph / messages ========================= */
typedef struct {
    int N, M, E;
    int *deg_v, *deg_c;
    int *vn_ptr, *cn_ptr;
    int *vn_edges, *cn_edges;
    int *edge2v, *edge2c;
} Tanner;

typedef struct {
    float *v2c, *c2v, *mpre, *r;
    int   *eta;           /* per-edge quota η_{c->v} */
    float *Lch, *Ltot;    /* channel LLR, total LLR */
} Msgs;

typedef struct {
    float  *min1, *min2;
    int    *argmin;
    int8_t *sign;
} CNScratch;

/* ========================= Max-heap on residual ========================= */
typedef struct {
    int   E, n;
    int  *idx;
    int  *pos;      /* edge -> heap index */
    float *key;     /* -> M->r */
} Heap;

static inline void heap_swap(Heap *h, int i, int j)
{
    int ei = h->idx[i], ej = h->idx[j];
    h->idx[i] = ej; h->idx[j] = ei;
    h->pos[ej] = i; h->pos[ei] = j;
}
static inline int heap_cmp(Heap *h, int i, int j) { return h->key[h->idx[i]] > h->key[h->idx[j]]; }

static void heap_build(Heap *h, float *key, int E)
{
    h->E = E; h->n = E; h->key = key;
    h->idx = (int *)malloc(sizeof(int) * (size_t)E);
    h->pos = (int *)malloc(sizeof(int) * (size_t)E);
    for (int i = 0; i < E; i++) { h->idx[i] = i; h->pos[i] = i; }
    for (int i = E / 2 - 1; i >= 0; i--) {
        int p = i;
        for (;;) {
            int l = 2*p + 1, r = l + 1, m = p;
            if (l < h->n && heap_cmp(h, l, m)) m = l;
            if (r < h->n && heap_cmp(h, r, m)) m = r;
            if (m == p) break;
            heap_swap(h, p, m); p = m;
        }
    }
}
static inline void heap_sift_up(Heap *h, int i)
{
    while (i > 0) {
        int p = (i - 1) >> 1;
        if (heap_cmp(h, p, i)) break;
        heap_swap(h, p, i); i = p;
    }
}
static inline void heap_sift_down(Heap *h, int i)
{
    for (;;) {
        int l = 2*i + 1, r = l + 1, m = i;
        if (l < h->n && heap_cmp(h, l, m)) m = l;
        if (r < h->n && heap_cmp(h, r, m)) m = r;
        if (m == i) break;
        heap_swap(h, i, m); i = m;
    }
}
static inline void heap_update_key(Heap *h, int e)
{
    int i = h->pos[e];
    heap_sift_up(h, i);
    heap_sift_down(h, i);
}
static inline int heap_pop(Heap *h)
{
    if (h->n == 0) return -1;
    int e = h->idx[0];
    h->idx[0] = h->idx[h->n - 1];
    h->pos[h->idx[0]] = 0;
    h->n--;
    heap_sift_down(h, 0);
    return e;
}
static inline void heap_push(Heap *h, int e)
{
    int i = h->n++;
    h->idx[i] = e;
    h->pos[e] = i;
    heap_sift_up(h, i);
}
static inline void heap_free(Heap *h)
{
    if (!h) return;
    free(h->idx); free(h->pos);
    h->idx = h->pos = NULL; h->key = NULL; h->n = h->E = 0;
}

/* ========================= Min-sum cache (mpre/residual) ========================= */
static inline void cn_refresh_min(const Tanner *T, const Msgs *M, CNScratch *W, int c)
{
    int s = T->cn_ptr[c], e = T->cn_ptr[c + 1];
    float m1 = 1e30f, m2 = 1e30f; int arg = -1; int8_t sgn = 0;
    for (int it = s; it < e; ++it) {
        int ed = T->cn_edges[it];
        float v = fabsf(M->v2c[ed]);
        if (v <= m1) { m2 = m1; m1 = v; arg = T->edge2v[ed]; }
        else if (v < m2) { m2 = v; }
        sgn ^= (M->v2c[ed] < 0.0f);
    }
    W->min1[c] = m1; W->min2[c] = m2; W->argmin[c] = arg; W->sign[c] = sgn;
}

/* 針對「特定邊」(ca -> vb)，排除 v_except = v_max，逐邊更新 mpre 與 residual */
static inline void cn_update_one_mpre_residual_except_v(
    const Tanner *T, Msgs *M, CNScratch *W, int ca, int vb, int v_except)
{
    (void)v_except; /* 依 min-sum 寫法，我們不直接用 v_except 判斷，改用 W 緩存 */
    /* 先刷新 ca 的 min1/min2/sign，確保使用的是最新 v2c 值 */
    cn_refresh_min(T, M, W, ca);

    /* 找到 (ca, vb) 的 edge id */
    int ed = -1;
    for (int jt = T->cn_ptr[ca]; jt < T->cn_ptr[ca + 1]; ++jt) {
        int e = T->cn_edges[jt];
        if (T->edge2v[e] == vb) { ed = e; break; }
    }
    if ( ed < 0) return;

    /* 依式(4) 以 min-sum 近似產生 m_pre(ca->vb) */
    float min_abs = (vb == W->argmin[ca]) ? W->min2[ca] : W->min1[ca];
    int8_t sgn_ex = W->sign[ca] ^ (M->v2c[ed] < 0.0f);
    float mpre = sgn_ex ? -min_abs : min_abs;

    M->mpre[ed] = mpre;
    M->r[ed]    = fabsf(mpre - M->c2v[ed]);
}

/* ========================= Exact selected-edge update: log-SPA ========================= */
static inline float c2v_exact_update(const Tanner *T, const Msgs *M, int c, int v_except)
{
    double sumphi = 0.0; int8_t sgn = 0;
    int s = T->cn_ptr[c], e = T->cn_ptr[c + 1];
    for (int it = s; it < e; ++it) {
        int ed = T->cn_edges[it];
        int v  = T->edge2v[ed];
        if (v == v_except) continue;
        float msg = M->v2c[ed];
        sumphi += phi_func(fabs((double)msg));
        sgn    ^= (msg < 0.0f);
    }
    float out = (float)phi_func(sumphi);
    return sgn ? -out : out;
}

/* ========================= .alist parser & utils ========================= */
static int read_alist(const char *path, Tanner *G)
{
    FILE *f = fopen(path, "r");
    if (!f) return -1;

    int N, M;
    if (fscanf(f, "%d %d", &N, &M) != 2) { fclose(f); return -1; }

    int dvmax, dcmax;
    if (fscanf(f, "%d %d", &dvmax, &dcmax) != 2) { fclose(f); return -1; }

    int *deg_v = (int *)calloc((size_t)N, sizeof(int));
    int *deg_c = (int *)calloc((size_t)M, sizeof(int));
    for (int v = 0; v < N; ++v) if (fscanf(f, "%d", &deg_v[v]) != 1) { fclose(f); return -1; }
    for (int c = 0; c < M; ++c) if (fscanf(f, "%d", &deg_c[c]) != 1) { fclose(f); return -1; }

    int **tmp_vn = (int **)malloc(sizeof(int *) * (size_t)N);
    for (int v = 0; v < N; ++v) {
        tmp_vn[v] = (int *)calloc((size_t)deg_v[v], sizeof(int));
        for (int k = 0; k < dvmax; k++) {
            int x = 0; if (fscanf(f, "%d", &x) != 1) x = 0;
            if (x > 0 && k < deg_v[v]) tmp_vn[v][k] = x - 1;
        }
    }
    for (int c = 0; c < M; ++c) for (int k = 0; k < dcmax; k++) { int d = 0; if (fscanf(f, "%d", &d) != 1) { fclose(f); return -1; } }

    int E = 0; for (int v = 0; v < N; ++v) E += deg_v[v];

    int *vn_ptr   = (int *)malloc(sizeof(int) * (size_t)(N + 1));
    int *vn_edges = (int *)malloc(sizeof(int) * (size_t)E);
    int *cn_ptr   = (int *)malloc(sizeof(int) * (size_t)(M + 1));
    int *cn_edges = (int *)malloc(sizeof(int) * (size_t)E);
    int *edge2v   = (int *)malloc(sizeof(int) * (size_t)E);
    int *edge2c   = (int *)malloc(sizeof(int) * (size_t)E);

    vn_ptr[0] = 0; for (int v = 0; v < N; ++v) vn_ptr[v + 1] = vn_ptr[v] + deg_v[v];
    cn_ptr[0] = 0; for (int c = 0; c < M; ++c) cn_ptr[c + 1] = cn_ptr[c] + deg_c[c];
    int *cn_write = (int *)malloc(sizeof(int) * (size_t)M);
    memcpy(cn_write, cn_ptr, sizeof(int) * (size_t)M);

    for (int v = 0, eidx = 0; v < N; ++v) {
        for (int k = 0; k < deg_v[v]; ++k) {
            int c = tmp_vn[v][k];
            int e = eidx++;
            vn_edges[vn_ptr[v] + k] = e;
            edge2v[e] = v; edge2c[e] = c;
            int p = cn_write[c]++;
            cn_edges[p] = e;
        }
    }

    for (int v = 0; v < N; ++v) free(tmp_vn[v]);
    free(tmp_vn); free(cn_write);

    G->N = N; G->M = M; G->E = E;
    G->deg_v = deg_v; G->deg_c = deg_c;
    G->vn_ptr = vn_ptr; G->vn_edges = vn_edges;
    G->cn_ptr = cn_ptr; G->cn_edges = cn_edges;
    G->edge2v = edge2v; G->edge2c = edge2c;

    fclose(f);
    return 0;
}
static void free_tanner(Tanner *G)
{
    free(G->deg_v); free(G->deg_c);
    free(G->vn_ptr); free(G->vn_edges);
    free(G->cn_ptr); free(G->cn_edges);
    free(G->edge2v); free(G->edge2c);
    memset(G, 0, sizeof(*G));
}

/* ========================= Syndrome & init ========================= */
static int message_is_all_zero(const Tanner *T, const Msgs *M)
{
    for (int v = 0; v < T->N; ++v) {
        if (M->Ltot[v] < 0.0f) return 0;
    }
    return 1;
}

static int syndrome_is_zero_and_all_zero(const Tanner *T, const Msgs *M)
{
    for (int c = 0; c < T->M; ++c) {
        int sum = 0;
        int s = T->cn_ptr[c], e = T->cn_ptr[c + 1];
        for (int it = s; it < e; ++it) {
            int v = T->edge2v[T->cn_edges[it]];
            sum ^= (M->Ltot[v] < 0.0f);
        }
        if (sum) return 0;
    }
    return message_is_all_zero(T, M);
}

static void initialize_msgs(const Tanner *T, Msgs *M, CNScratch *W)
{
    memset(M->c2v, 0, sizeof(float) * (size_t)T->E);
    for (int v = 0; v < T->N; ++v) {
        float L = M->Lch[v];
        M->Ltot[v] = L;
        for (int it = T->vn_ptr[v]; it < T->vn_ptr[v + 1]; ++it) {
            int e = T->vn_edges[it];
            M->v2c[e] = L;
        }
    }
    for (int c = 0; c < T->M; ++c) {
        cn_refresh_min(T, M, W, c);
        /* 初始 mpre/residual 也建好，避免 heap 初值偏差 */
        int s = T->cn_ptr[c], e = T->cn_ptr[c + 1];
        for (int it = s; it < e; ++it) {
            int ed  = T->cn_edges[it];
            int vb  = T->edge2v[ed];
            float min_abs = (vb == W->argmin[c]) ? W->min2[c] : W->min1[c];
            int8_t sgn_ex = W->sign[c] ^ (M->v2c[ed] < 0.0f);
            float mpre = sgn_ex ? -min_abs : min_abs;
            M->mpre[ed] = mpre;
            M->r[ed]    = fabsf(mpre - M->c2v[ed]);
        }
    }
    memset(M->eta, 0, sizeof(int) * (size_t)T->E);
}

/* ========================= Q-RBP (Algorithm-1) ========================= */
static inline int ceil_div_int(long long a, int b) { return (int)((a + b - 1) / b); }

static int qrbp_decode(const Tanner *T, Msgs *M, CNScratch *W,
                       int IM, int QF, int check_period)
{
    Heap H; heap_build(&H, M->r, T->E);
    long long updates_total = 0;
    int I = 0, steps_since_check = 0;
    int *popped = (int *)malloc(sizeof(int) * (size_t)T->E);

    for (;;) {
        int e = -1, pcnt = 0;

        /* Step 7: pick max residual with gating (η < QF + I) */
        for (;;) {
            int top = heap_pop(&H);
            if (top < 0) {
                int newI = ceil_div_int(updates_total, T->E); /* ← ceil */
                I = newI;
                if (I >= IM) { free(popped); heap_free(&H); return syndrome_is_zero_and_all_zero(T, M); }
                for (int i = 0; i < pcnt; ++i) heap_push(&H, popped[i]);
                pcnt = 0;
                continue;
            }
            if (M->eta[top] < QF + I) { e = top; break; }
            popped[pcnt++] = top;
        }
        for (int i = 0; i < pcnt; ++i) heap_push(&H, popped[i]);

        /* Step 8–16（逐邊一致性＋內層呼叫） */
        {
            int c = T->edge2c[e];
            int v = T->edge2v[e];

            /* (8) exact SPA on selected edge */
            float new_c2v = c2v_exact_update(T, M, c, v);
            float old_c2v = M->c2v[e];
            M->c2v[e] = new_c2v;

            /* (9) set r=0 for chosen edge; push back; η++ */
            M->r[e] = 0.0f;
            heap_push(&H, e);
            M->eta[e] += 1;
            updates_total++;

            /* 更新 VN 總 LLR */
            M->Ltot[v] += (new_c2v - old_c2v);

            /* 同步該邊的 v2c[e]，避免 c 的其他邊看到舊值 */
            M->v2c[e] = M->Ltot[v] - M->c2v[e];

            /* 重新計算並刷新同一個 check c 的其它邊 (Step 9) */
            for (int jt = T->cn_ptr[c]; jt < T->cn_ptr[c + 1]; ++jt) {
                int edcn = T->cn_edges[jt];
                if (edcn == e) continue;
                int vb = T->edge2v[edcn];
                cn_update_one_mpre_residual_except_v(T, M, W, c, vb, v);
                heap_update_key(&H, edcn);
            }

            /* (10–16) 對每個 ca ∈ N(v)\{c} */
            for (int it = T->vn_ptr[v]; it < T->vn_ptr[v + 1]; ++it) {
                int e2 = T->vn_edges[it];
                int ca = T->edge2c[e2];
                if (ca == c) continue;

                /* (11) propagate v->ca */
                M->v2c[e2] = M->Ltot[v] - M->c2v[e2];

                /* (12–14) 逐邊：對 N(ca)\{v} 的每個 vb，單獨重算 mpre/residual，並立刻更新 heap key */
                for (int jt = T->cn_ptr[ca]; jt < T->cn_ptr[ca + 1]; ++jt) {
                    int edcn = T->cn_edges[jt];
                    int vb   = T->edge2v[edcn];
                    if (vb == v) continue;  /* 排除回到 v 的那條 */
                    cn_update_one_mpre_residual_except_v(T, M, W, ca, vb, v);
                    heap_update_key(&H, edcn);
                }
            }
        }

        /* (18) I = ceil(sum(eta)/E) */
        I = ceil_div_int(updates_total, T->E);

        if (++steps_since_check >= check_period) {
            steps_since_check = 0;
            if (syndrome_is_zero_and_all_zero(T, M)) { free(popped); heap_free(&H); return 1; }
        }
        if (I >= IM) { free(popped); heap_free(&H); return syndrome_is_zero_and_all_zero(T, M); }
    }
}

/* ========================= AWGN / BPSK ========================= */
static inline float clipf(float x, float lo, float hi)
{
    if (x < lo) return lo;
    if (x > hi) return hi;
    return x;
}

static void transmit_all_zero_and_get_Lch(const Tanner *T, Msgs *M, double EbN0_dB, double rate)
{
    double EbN0 = pow(10.0, EbN0_dB / 10.0);
    double sigma2 = 1.0 / (2.0 * rate * EbN0);
    double sigma  = sqrt(sigma2);
    for (int v = 0; v < T->N; ++v) {
        double n = sigma * Normal();
        double y = 1.0 + n;           /* all-zero -> +1 */
        double L = 2.0 * y / sigma2;  /* exact LLR */
        L = clipf((float)L, -40.0f, 40.0f);  /* clipping ±40 */
        M->Lch[v] = (float)L;
    }
}

/* ========================= Preset SNR & targets ========================= */
static const double snr_db_list[] = {1.00, 1.25, 1.50, 1.75, 2.00};
static const int    targets[]     = { 200,  150,  100,   50,   10};
static const int    S             = (int)(sizeof(snr_db_list) / sizeof(snr_db_list[0]));

static void run_preset_points(const Tanner *G, Msgs *M, CNScratch *W,
                              double R_est, int IM, int QF)
{
    const int CHECK_PERIOD = G->E; /* 與論文一致 */

    for (int s = 2; s < S; ++s) {
        const double snr = snr_db_list[s];
        const unsigned need = (unsigned)targets[s];
        unsigned long long frames = 0ULL, errors = 0ULL;

        while (errors < (unsigned long long)need) {
            transmit_all_zero_and_get_Lch(G, M, snr, R_est);
            initialize_msgs(G, M, W);
            if (!qrbp_decode(G, M, W, IM, QF, CHECK_PERIOD)) ++errors;
            ++frames;

            if ((frames % 10ULL) == 0ULL) {
                const double fer = frames ? (double)errors / (double)frames : 0.0;
                printf("\rSNR %.2f: Frames=%llu  Errors=%llu  FER=%e",
                       snr, frames, errors, fer);
                fflush(stdout);
            }
        }
        const double fer = frames ? (double)errors / (double)frames : 0.0;
        printf("\nSNR %.2f: Frames=%llu  Errors=%llu  FER=%e  (target=%d)\n",
               snr, frames, errors, fer, targets[s]);
        fflush(stdout);
    }
}

/* ========================= Main ========================= */
int main(void)
{
    setvbuf(stdout, NULL, _IOLBF, 0);

    const int   IM   = 50;
    const int   QF   = 30;

    const char *alist_path = "wifi_1944_R12_alist.txt";

    Tanner G = (Tanner){0};
    if (read_alist(alist_path, &G)) {
        fprintf(stderr, "Failed to read alist: %s\n", alist_path);
        return 1;
    }
    fprintf(stderr, "Using alist: %s\n", alist_path);
    fprintf(stderr, "Graph: N=%d M=%d E=%d (avg dv=%.2f, dc=%.2f)\n",
            G.N, G.M, G.E, (double)G.E / (double)G.N, (double)G.E / (double)G.M);

    Msgs M = (Msgs){0};
    M.v2c  = (float *)xaligned_alloc(64, sizeof(float) * (size_t)G.E);
    M.c2v  = (float *)xaligned_alloc(64, sizeof(float) * (size_t)G.E);
    M.mpre = (float *)xaligned_alloc(64, sizeof(float) * (size_t)G.E);
    M.r    = (float *)xaligned_alloc(64, sizeof(float) * (size_t)G.E);
    M.eta  = (int   *)xaligned_alloc(64, sizeof(int)   * (size_t)G.E);
    M.Lch  = (float *)xaligned_alloc(64, sizeof(float) * (size_t)G.N);
    M.Ltot = (float *)xaligned_alloc(64, sizeof(float) * (size_t)G.N);

    CNScratch W = (CNScratch){0};
    W.min1   = (float *)xaligned_alloc(64, sizeof(float) * (size_t)G.M);
    W.min2   = (float *)xaligned_alloc(64, sizeof(float) * (size_t)G.M);
    W.argmin = (int   *)xaligned_alloc(64, sizeof(int)   * (size_t)G.M);
    W.sign   = (int8_t*)xaligned_alloc(64, sizeof(int8_t)* (size_t)G.M);

    /* randomize seed per run for rand49()/Normal() */
    uint32_t seed = (uint32_t)(time(NULL) ^ (uintptr_t)&seed ^ (uint32_t)clock());
    srand(seed);
    fprintf(stderr, "RNG seed=%u\n", seed);

    double R_est = 1.0 - (double)G.M / (double)G.N; /* 0.5 for 1944x972 */

    printf("---------------------------------------------------------------------------------------\n");
    printf("SNR(dB)        FER            Frames        Errors                                     \n");
    printf("---------------------------------------------------------------------------------------\n");

    run_preset_points(&G, &M, &W, R_est, IM, QF);

    xaligned_free(M.v2c); xaligned_free(M.c2v); xaligned_free(M.mpre);
    xaligned_free(M.r);   xaligned_free(M.eta);
    xaligned_free(M.Lch); xaligned_free(M.Ltot);
    xaligned_free(W.min1); xaligned_free(W.min2); xaligned_free(W.argmin); xaligned_free(W.sign);
    free_tanner(&G);
    return 0;
}

