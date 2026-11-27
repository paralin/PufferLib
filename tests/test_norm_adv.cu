// Check train.norm_adv, the constant teacher weight at teacher_steps <= 0,
// selfplay.pfsp_mix league weights and the vec.hist_policy_* value head.
// Build from the repo root (Linux/CUDA):
//   mkdir -p build
//   nvcc -O2 -arch=native -std=c++17 -DPRECISION_FLOAT \
//       -I. -Isrc -Ivendor -Iraylib-5.5_linux_amd64/include \
//       -DPLATFORM_DESKTOP -Xcompiler=-fopenmp -Xcompiler=-Wno-narrowing \
//       --diag-suppress=2361 --diag-suppress=111 --diag-suppress=128 \
//       tests/test_norm_adv.cu raylib-5.5_linux_amd64/lib/libraylib.a \
//       -lcudart -lnccl -lnvidia-ml -lcublas -lcusolver -lcurand \
//       -lGL -lm -lpthread -ldl -lgomp -o build/test_norm_adv
//   ./build/test_norm_adv
// Linux/ROCm, after ./build-hip.sh ENV has translated src/ into build/hip/src:
//   mkdir -p build/hip/tests
//   /opt/rocm/bin/hipify-perl --quiet-warnings tests/test_norm_adv.cu \
//       > build/hip/tests/test_norm_adv.cu
//   /opt/rocm/bin/hipcc --offload-arch=native -std=c++17 -O2 -fopenmp \
//       -Wno-narrowing -Wno-deprecated-declarations -DPRECISION_FLOAT \
//       -Ibuild/hip -Ibuild/hip/src -I. -Isrc -Ivendor \
//       -Iraylib-5.5_linux_amd64/include -I/opt/rocm/include/hipblas \
//       -I/opt/rocm/include/hiprand -I/opt/rocm/include/hipsolver -DPLATFORM_DESKTOP \
//       -x hip build/hip/tests/test_norm_adv.cu -x none \
//       raylib-5.5_linux_amd64/lib/libraylib.a -L/opt/rocm/lib -Wl,-rpath,/opt/rocm/lib \
//       -lhipblas -lhiprand -lrocrand -lhipsolver -lrccl -lGL -lm -lpthread \
//       -o build/hip/test_norm_adv
// Also build without PRECISION_FLOAT to check bf16. Keep assertions enabled.
typedef float obs_t;
#include "pufferenv.h"

#define NUM_ATNS 1
#define TEST_CLASSES 5
#define ACT_SIZES {TEST_CLASSES}
#define OBS_SIZE 4
#define ENV_HEADER "pufferenv.h"
#define PUFFER_ENV_NAME "norm_adv"

struct Log { float perf, n; };
struct Env {
    Log log;
    Agent agents[1];
    int num_agents;
    int tag, boundary_reached;
    unsigned int rng;
};
void puf_init(Env* env, Dict* kwargs) {
    env->num_agents = 1;
}
void puf_reset(Env* env) {}
void puf_step(Env* env) {}
void puf_render(Env* env) {}
void puf_close(Env* env) {}
void puf_log(Log* log, Dict* out) { dict_set(out, "perf", log->perf); }

#include "../src/pufferl.cu"

#ifdef PRECISION_FLOAT
static const float kTolerance = 1e-5f;
#else
static const float kTolerance = 2e-2f;
#endif

// Rows and steps of the test minibatch; NT spans several PPO blocks with a tail.
static const int kRows = 24;
static const int kSteps = 37;
static const int kNT = kRows * kSteps;

// Uploads host floats as a device tensor in the training precision.
static precision_t* upload_prec(const std::vector<float>& host) {
    int n = (int)host.size();
    float* staging = NULL;
    precision_t* out = NULL;
    assert(cudaMalloc((void**)&staging, n * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc((void**)&out, n * sizeof(precision_t)) == cudaSuccess);
    cudaMemcpy(staging, host.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cast<<<grid_size(n), BLOCK_SIZE>>>(out, staging, n);
    assert(cudaDeviceSynchronize() == cudaSuccess);
    cudaFree(staging);
    return out;
}

// Rounds host floats through the training precision, as the kernels read them.
static std::vector<float> round_prec(const std::vector<float>& host) {
    int n = (int)host.size();
    precision_t* dev = upload_prec(host);
    float* wide = NULL;
    assert(cudaMalloc((void**)&wide, n * sizeof(float)) == cudaSuccess);
    cast<<<grid_size(n), BLOCK_SIZE>>>(wide, dev, n);
    std::vector<float> out(n);
    cudaMemcpy(out.data(), wide, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(wide);
    cudaFree(dev);
    return out;
}

static std::vector<float> download(const float* dev, int n) {
    std::vector<float> out(n);
    assert(cudaMemcpy(out.data(), dev, n * sizeof(float), cudaMemcpyDeviceToHost)
        == cudaSuccess);
    return out;
}

// Harness owns one fixed minibatch and reruns the fused PPO loss on it with
// chosen advantages and expert weights.
struct Harness {
    Allocator alloc;
    PPOBufs bufs;
    TrainGraph graph;
    Prec dec;
    Prec logstd;
    int* act_sizes;
    float* losses;
    ValueHead head;
};

static void harness_init(Harness* h) {
    *h = {};
    std::vector<float> logits(kNT * (TEST_CLASSES + 1));
    std::vector<float> actions(kNT);
    std::vector<float> old_logprobs(kNT);
    std::vector<float> values(kNT);
    std::vector<float> returns(kNT);
    std::vector<float> mask(kNT * TEST_CLASSES, 1.0f);
    for (int i = 0; i < kNT; i++) {
        for (int c = 0; c <= TEST_CLASSES; c++) {
            logits[i * (TEST_CLASSES + 1) + c] = 0.8f * sinf(0.37f * i + 1.3f * c);
        }
        actions[i] = (float)(i % TEST_CLASSES);
        old_logprobs[i] = -1.6f + 0.4f * sinf(0.91f * i);
        values[i] = 0.5f * cosf(0.23f * i);
        returns[i] = values[i] + 0.3f * sinf(0.11f * i);
    }

    // The decoder output and rollout views the loss reads.
    h->dec = {.shape = {kRows, kSteps, TEST_CLASSES + 1}};
    h->dec.data = upload_prec(logits);
    h->graph.mb_actions = {.shape = {kRows, kSteps, NUM_ATNS}};
    assert(cudaMalloc((void**)&h->graph.mb_actions.data, kNT * sizeof(float)) == cudaSuccess);
    cudaMemcpy(h->graph.mb_actions.data, actions.data(), kNT * sizeof(float),
        cudaMemcpyHostToDevice);
    h->graph.mb_logprobs.data = upload_prec(old_logprobs);
    h->graph.mb_values.data = upload_prec(values);
    h->graph.mb_action_mask.data = upload_prec(mask);
    h->graph.mb_imp.data = upload_prec(std::vector<float>(kNT, 1.0f));
    h->graph.mb_returns.data = upload_prec(returns);
    h->graph.mb_gae_v.data = upload_prec(std::vector<float>(kNT, 0.0f));

    // Loss scratch, the action layout and loss accumulators.
    register_ppo_buffers(h->bufs, &h->alloc, kRows, kSteps, TEST_CLASSES, 1, false);
    alloc_create(&h->alloc);
    float ent_coef = 0.01f;
    cudaMemcpy(h->bufs.ent_coef, &ent_coef, sizeof(float), cudaMemcpyHostToDevice);
    int classes = TEST_CLASSES;
    assert(cudaMalloc((void**)&h->act_sizes, sizeof(int)) == cudaSuccess);
    cudaMemcpy(h->act_sizes, &classes, sizeof(int), cudaMemcpyHostToDevice);
    assert(cudaMalloc((void**)&h->losses, (LOSS_N + 1) * sizeof(float)) == cudaSuccess);
}

// Result of one loss pass: gradients, the policy loss and the device stats.
struct LossRun {
    std::vector<float> grad_logits;
    std::vector<float> grad_values;
    float pg_loss;
    float stats[2];
};

static LossRun run_loss(Harness* h, const std::vector<float>& advantages,
        const std::vector<float>& expert, bool norm_adv) {
    precision_t* adv = upload_prec(advantages);
    precision_t* weights = upload_prec(expert);
    h->graph.mb_advantages.data = adv;
    h->graph.mb_expert.data = weights;
    cudaMemset(h->losses, 0, (LOSS_N + 1) * sizeof(float));
    float unset[2] = {NAN, NAN};
    cudaMemcpy(h->bufs.adv_stats, unset, sizeof(unset), cudaMemcpyHostToDevice);

    // The forward pass writes log-probs into grad_logits; the loss overwrites them.
    cache_imp_and_v<<<grid_size(kNT), BLOCK_SIZE>>>(h->dec, h->graph.mb_actions.data,
        h->graph.mb_logprobs.data, h->graph.mb_action_mask.data, weights, h->logstd,
        h->act_sizes, h->graph.mb_imp.data, h->graph.mb_gae_v.data,
        h->bufs.grad_logits.data, h->bufs.new_logprobs.data, h->head);
    ppo_loss_fwd_bwd(h->dec, h->logstd, h->graph, h->act_sizes, h->losses,
        0.2f, 0.2f, 0.5f, h->bufs.ent_coef, h->bufs.teacher_coef, h->bufs, h->head,
        false, norm_adv, 0);
    assert(cudaDeviceSynchronize() == cudaSuccess);

    LossRun run;
    run.grad_logits = download(h->bufs.grad_logits.data, kNT * TEST_CLASSES);
    run.grad_values = download(h->bufs.grad_values.data, kNT);
    run.pg_loss = download(h->losses, LOSS_N)[LOSS_PG];
    cudaMemcpy(run.stats, h->bufs.adv_stats, sizeof(run.stats), cudaMemcpyDeviceToHost);
    cudaFree(adv);
    cudaFree(weights);
    return run;
}

// Largest absolute difference, relative to the largest magnitude in want.
static float relative_error(const std::vector<float>& got, const std::vector<float>& want) {
    float scale = 1e-12f;
    float error = 0.0f;
    for (size_t i = 0; i < want.size(); i++) {
        assert(isfinite(got[i]));
        scale = fmaxf(scale, fabsf(want[i]));
        error = fmaxf(error, fabsf(got[i] - want[i]));
    }
    return error / scale;
}

// Reference free-row mean and 1 / (sample std + 1e-8) in double.
static void reference_stats(const std::vector<float>& adv, const std::vector<float>& expert,
        double* mean, double* inv_std) {
    double count = 0;
    double sum = 0;
    for (int i = 0; i < kNT; i++) {
        if (isnan(expert[i])) {
            count += 1;
            sum += adv[i];
        }
    }
    *mean = sum / count;
    double sq = 0;
    for (int i = 0; i < kNT; i++) {
        if (isnan(expert[i])) {
            sq += (adv[i] - *mean) * (adv[i] - *mean);
        }
    }
    *inv_std = 1.0 / (sqrt(sq / (count - 1)) + 1e-8);
}

// Normalized advantages match the reference, and training on them matches
// training on advantages normalized before the loss. Expert rows are excluded.
static void test_norm_adv(Harness* h) {
    std::vector<float> raw(kNT);
    std::vector<float> expert(kNT);
    for (int i = 0; i < kNT; i++) {
        raw[i] = 3.0f + 5.0f * sinf(0.731f * i);
        expert[i] = i % 7 == 0 ? 0.5f : NAN;
    }
    // Expert rows carry outlying advantages that must not move the statistics.
    for (int i = 0; i < kNT; i += 7) {
        raw[i] = 400.0f;
    }
    std::vector<float> adv = round_prec(raw);
    double mean, inv_std;
    reference_stats(adv, expert, &mean, &inv_std);

    LossRun normed = run_loss(h, adv, expert, true);
    printf("norm_adv stats: mean %.6f (want %.6f), inv_std %.6f (want %.6f)\n",
        normed.stats[0], mean, normed.stats[1], inv_std);
    assert(fabs(normed.stats[0] - mean) < 1e-4 * fmax(1.0, fabs(mean)));
    assert(fabs(normed.stats[1] - inv_std) < 1e-4 * inv_std);

    std::vector<float> prenormed(kNT);
    for (int i = 0; i < kNT; i++) {
        prenormed[i] = (float)((adv[i] - mean) * inv_std);
    }
    LossRun reference = run_loss(h, prenormed, expert, false);
    float logit_error = relative_error(normed.grad_logits, reference.grad_logits);
    float value_error = relative_error(normed.grad_values, reference.grad_values);
    float pg_error = fabsf(normed.pg_loss - reference.pg_loss)
        / fmaxf(1.0f, fabsf(reference.pg_loss));
    printf("norm_adv vs prenormalized: logits %.3g, values %.3g, pg loss %.3g\n",
        logit_error, value_error, pg_error);
    assert(logit_error < kTolerance && value_error < kTolerance && pg_error < kTolerance);

    // Normalization only rescales the policy gradient, not the value gradient.
    LossRun raw_run = run_loss(h, adv, expert, false);
    assert(relative_error(raw_run.grad_values, normed.grad_values) == 0.0f);
    assert(relative_error(raw_run.grad_logits, normed.grad_logits) > kTolerance);
}

// Minibatches with fewer than two free rows keep unit scale and stay finite.
static void test_norm_adv_degenerate(Harness* h) {
    std::vector<float> adv(kNT);
    for (int i = 0; i < kNT; i++) {
        adv[i] = 2.0f + sinf(0.5f * i);
    }
    adv = round_prec(adv);

    // An all-expert minibatch trains only imitation, so normalization is moot.
    std::vector<float> all_expert(kNT, 1.0f);
    LossRun none = run_loss(h, adv, all_expert, true);
    LossRun none_raw = run_loss(h, adv, all_expert, false);
    assert(none.stats[0] == 0.0f && none.stats[1] == 1.0f);
    assert(relative_error(none.grad_logits, none_raw.grad_logits) == 0.0f);

    // A lone free row has zero advantage after centering.
    std::vector<float> one_free(kNT, 1.0f);
    one_free[5] = NAN;
    LossRun one = run_loss(h, adv, one_free, true);
    assert(one.stats[0] == adv[5] && one.stats[1] == 1.0f);
    std::vector<float> centered = adv;
    centered[5] = 0.0f;
    LossRun one_ref = run_loss(h, centered, one_free, false);
    assert(relative_error(one.grad_logits, one_ref.grad_logits) == 0.0f);
    printf("norm_adv degenerate minibatches: ok\n");
}

// teacher_steps > 0 decays the weight linearly; teacher_steps <= 0 holds it.
static void test_teacher_coef() {
    PuffeRL* p = (PuffeRL*)calloc(1, sizeof(PuffeRL));
    p->hypers.teacher_weight = 0.7f;
    p->hypers.teacher_steps = 100;
    p->global_step = 50;
    assert(puf_teacher_coef(p) == 0.0f);

    p->teaching = true;
    assert(fabsf(puf_teacher_coef(p) - 0.35f) < 1e-6f);
    p->global_step = 200;
    assert(puf_teacher_coef(p) == 0.0f);

    p->hypers.teacher_steps = 0;
    p->global_step = 0;
    assert(puf_teacher_coef(p) == 0.7f);
    p->global_step = 1000000000000L;
    assert(puf_teacher_coef(p) == 0.7f);
    free(p);
    printf("teacher coef: ok\n");
}

// pfsp_mix blends (1 - p)^2 and p(1 - p); 0 keeps the legacy weights.
static void test_pfsp_mix() {
    // Win rates 0.05, 0.5 and 0.95 after the one-half prior.
    LeagueMember members[3] = {};
    members[0].games = 9;
    members[0].wins = 0;
    members[1].games = 9;
    members[1].wins = 4.5;
    members[2].games = 9;
    members[2].wins = 9;
    League league = {};
    league.members = members;
    league.size = 3;

    for (int i = 0; i < 3; i++) {
        double p = league_win_rate(&members[i]);
        double pfsp = (1 - p) * (1 - p);
        double even = p * (1 - p);

        league.pfsp_mix = 0;
        league.even = false;
        assert(fabs(league_weight(&league, i, NAN) - pfsp) < 1e-12);
        league.even = true;
        assert(fabs(league_weight(&league, i, NAN) - even) < 1e-12);
        league.even = false;

        league.pfsp_mix = 1;
        assert(fabs(league_weight(&league, i, NAN) - pfsp) < 1e-12);
        league.pfsp_mix = 0.25;
        assert(fabs(league_weight(&league, i, NAN) - (0.25 * pfsp + 0.75 * even)) < 1e-12);
    }

    // The blend gives the hardest member more share than even alone does.
    league.pfsp_mix = 0.5;
    double mixed = league_weight(&league, 0, NAN) / league_weight(&league, 1, NAN);
    league.pfsp_mix = 0;
    league.even = true;
    double even = league_weight(&league, 0, NAN) / league_weight(&league, 1, NAN);
    printf("pfsp_mix 0.5 hard/even share %.3f vs even %.3f\n", mixed, even);
    assert(mixed > even);
}

// Samples rows of a decoder output with cols columns through head and
// returns the decoded values.
static std::vector<float> sample_values(const std::vector<float>& dec_host, int rows,
        int cols, ValueHead head) {
    Prec dec = {.data = upload_prec(dec_host), .shape = {rows, cols}};
    precision_t* mask = upload_prec(std::vector<float>(rows * TEST_CLASSES, 1.0f));
    precision_t* out = upload_prec(std::vector<float>(2 * rows, 0.0f));
    float* actions = NULL;
    int* act_sizes = NULL;
    curandStatePhilox4_32_10_t* rng = NULL;
    assert(cudaMalloc((void**)&actions, 2 * rows * sizeof(float)) == cudaSuccess);
    assert(cudaMalloc((void**)&act_sizes, sizeof(int)) == cudaSuccess);
    assert(cudaMalloc((void**)&rng, rows * sizeof(*rng)) == cudaSuccess);
    int classes = TEST_CLASSES;
    cudaMemcpy(act_sizes, &classes, sizeof(int), cudaMemcpyHostToDevice);
    rng_init<<<grid_size(rows), BLOCK_SIZE>>>(rng, 7, rows);
    sample_logits<<<grid_size(rows), BLOCK_SIZE>>>(dec, Prec{}, act_sizes,
        actions, actions + rows, out, out + rows, rng, mask, TEST_CLASSES, head);
    assert(cudaDeviceSynchronize() == cudaSuccess);

    float* wide = NULL;
    assert(cudaMalloc((void**)&wide, rows * sizeof(float)) == cudaSuccess);
    cast<<<grid_size(rows), BLOCK_SIZE>>>(wide, out + rows, rows);
    std::vector<float> values = download(wide, rows);
    for (void* ptr : {(void*)dec.data, (void*)mask, (void*)out, (void*)actions,
            (void*)act_sizes, (void*)rng, (void*)wide}) {
        cudaFree(ptr);
    }
    return values;
}

// A layer_norm = 1, value_bins = 51 learner can run legacy opponents and
// teachers with layer_norm = 0 and a scalar value head.
static void test_hist_policy_head() {
    ValueHead learner = {.bins = 51, .min = -2.0f, .max = 3.0f};
    bool ln;
    ValueHead head;

    // -1 inherits the learner's layer_norm and bins.
    assert(hist_policy_head(true, learner, -1, -1, &ln, &head));
    assert(ln && head.bins == 51 && head.min == -2.0f && head.max == 3.0f);
    // 0 selects no layer_norm and a scalar value; the range still follows the learner.
    assert(hist_policy_head(true, learner, 0, 0, &ln, &head));
    assert(!ln && head.bins == 0 && value_cols(head) == 1 && head.min == -2.0f);
    // A scalar learner can face a binned opponent.
    ValueHead scalar = {.bins = 0, .min = -1.0f, .max = 1.0f};
    assert(hist_policy_head(false, scalar, 1, 21, &ln, &head));
    assert(ln && head.bins == 21 && value_cols(head) == 21);
    // Unsupported options and an empty bin range fail.
    assert(!hist_policy_head(true, learner, 2, -1, &ln, &head));
    assert(!hist_policy_head(true, learner, -2, -1, &ln, &head));
    assert(!hist_policy_head(true, learner, -1, 1, &ln, &head));
    assert(!hist_policy_head(true, learner, -1, -2, &ln, &head));
    assert(!hist_policy_head(true, learner, -1, VALUE_MAX_BINS + 1, &ln, &head));
    assert(hist_policy_head(true, learner, -1, VALUE_MAX_BINS, &ln, &head));
    ValueHead empty = {.bins = 0, .min = 1.0f, .max = 1.0f};
    assert(!hist_policy_head(false, empty, -1, 51, &ln, &head));

    // Built archs carry the resolved shape.
    ArchConfig config = {.input_size = OBS_SIZE, .hidden_size = 16, .num_layers = 1,
        .output_size = TEST_CLASSES, .value_cols = value_cols(learner), .horizon = 4,
        .continuous = false, .layer_norm = true};
    Arch learner_arch = build_arch(config);
    assert(hist_policy_head(true, learner, 0, 0, &ln, &head));
    config.layer_norm = ln;
    config.value_cols = value_cols(head);
    Arch legacy_arch = build_arch(config);
    assert(learner_arch.network.layer_norm && learner_arch.decoder.value_cols == 51);
    assert(!legacy_arch.network.layer_norm && legacy_arch.decoder.value_cols == 1);

    // Sampling reads each policy's own value columns after its logits.
    const int rows = 9;
    int scalar_cols = TEST_CLASSES + 1;
    std::vector<float> legacy(rows * scalar_cols);
    for (int i = 0; i < (int)legacy.size(); i++) {
        legacy[i] = 0.6f * sinf(0.29f * i);
    }
    legacy = round_prec(legacy);
    std::vector<float> values = sample_values(legacy, rows, scalar_cols, head);
    for (int r = 0; r < rows; r++) {
        assert(values[r] == legacy[r * scalar_cols + TEST_CLASSES]);
    }
    int binned_cols = TEST_CLASSES + learner.bins;
    std::vector<float> binned(rows * binned_cols, 0.0f);
    for (int r = 0; r < rows; r++) {
        binned[r * binned_cols + TEST_CLASSES + learner.bins - 1] = 30.0f;
    }
    values = sample_values(binned, rows, binned_cols, learner);
    float top = learner.min + (learner.bins - 0.5f) * (learner.max - learner.min) / learner.bins;
    for (int r = 0; r < rows; r++) {
        assert(fabsf(values[r] - top) < kTolerance * fabsf(top) + 1e-3f);
    }
    printf("hist policy head: ok\n");
}

int main() {
    Harness h;
    harness_init(&h);
    test_norm_adv(&h);
    test_norm_adv_degenerate(&h);
    test_teacher_coef();
    test_pfsp_mix();
    test_hist_policy_head();
    printf("test_norm_adv: ok\n");
    return 0;
}
