#include "klpo.cuh"

#ifdef PUF_CONDITIONAL_GROUPS
__global__ void KlpoRecord(Prec decoder, const float* actions, float* behavior) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < decoder.shape[0]) {
        ConditionalLogProb(decoder.data + row * decoder.shape[1],
            actions + row * NUM_ATNS, behavior + row * ConditionalWidth());
    }
}

__global__ void KlpoPrepare(const precision_t* rewards, const precision_t* dones,
        float* targets, float* priorities, float* counts, int rows, int horizon,
        const int* ends, int buffer_rows, int learner_rows,
        const float* gamma, bool whole_match, float alpha) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    int n[2];
    int physical = row / learner_rows * buffer_rows + row % learner_rows;
    KlpoTargets(rewards + row * horizon, dones + row * horizon,
        targets + row * horizon, horizon, ends[physical], *gamma, whole_match, n);
    float mass = 0;
    for (int t = 0; t < horizon; ++t) {
        float value = targets[row * horizon + t];
        if (isfinite(value)) {
            mass += fabsf(value);
        }
    }
    priorities[row] = n[0] ? powf(mass + 1e-6f, alpha) : 0;
    atomicAdd(counts, (float)n[0]);
    atomicAdd(counts + 1, (float)n[1]);
}

// One decision per thread, exact conditional centering, no auxiliary sampling
// or device-to-host scalar reads. Reuses the native decoder/backward buffers.
__global__ void KlpoLoss(Prec decoder, const float* behavior, const float* targets,
        const float* actions, const float* importance, const float* counts,
        int rows, float beta, float* gradients, float* value_gradients, float* partials) {
    __shared__ float block_losses[LOSS_N][PPO_THREADS];
    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + tid;
    int samples = decoder.shape[0], horizon = decoder.shape[1];
    constexpr int width = ConditionalWidth();
    for (int loss = 0; loss < LOSS_N; ++loss) {
        block_losses[loss][tid] = 0;
    }
    if (index < samples * horizon) {
        float* grad = gradients + index * width;
        value_gradients[index] = 0;
        if (isfinite(targets[index])) {
            float p[width];
            ConditionalLogProb(decoder.data + index * (width + 1),
                actions + index * NUM_ATNS, p);
            KlpoScore score = KlpoConditionalScore(p, behavior + index * width,
                actions + index * NUM_ATNS, grad);
            float weight = importance[index / horizon] * rows / (samples * counts[0]);
            float feedback = targets[index] - beta * score.log_ratio;
            for (int col = 0; col < width; ++col) {
                grad[col] *= -feedback * weight;
            }
            float loss = -feedback * score.centered * weight;
            block_losses[LOSS_PG][tid] = loss;
            block_losses[LOSS_TOTAL][tid] = loss;
            float diagnostic_weight = importance[index / horizon] * rows / (samples * counts[1]);
            block_losses[LOSS_ENT][tid] = score.entropy * diagnostic_weight;
            block_losses[LOSS_APPROX_KL][tid] = score.kl * diagnostic_weight;
            block_losses[LOSS_IMP][tid] = diagnostic_weight;
        } else {
            for (int col = 0; col < width; ++col) {
                grad[col] = 0;
            }
        }
    }
    block_reduce_sum(&block_losses[0][0], partials + blockIdx.x * LOSS_N,
        tid, PPO_THREADS, LOSS_N);
}
#endif
