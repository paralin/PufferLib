#pragma once

// Replay owns only the current rollout. Rows are already compacted to learners.
struct Replay {
    Prec advantages, returns, state;
    Float probabilities, cdf, importance;
    RolloutBuf batch;
    int* indices;
    float* beta;
};

void RegisterReplay(Replay* replay, Allocator* alloc, int rows, int samples,
        int horizon, int inputs, int actions, int mask, int layers, int hidden) {
    replay->advantages = {.shape = {rows, horizon}};
    replay->returns = {.shape = {rows, horizon}};
    replay->state = {.shape = {layers, samples, hidden}};
    replay->probabilities = {.shape = {rows}};
    replay->cdf = {.shape = {rows}};
    replay->importance = {.shape = {samples}};
    alloc_register(alloc, &replay->advantages);
    alloc_register(alloc, &replay->returns);
    alloc_register(alloc, &replay->state);
    alloc_register(alloc, &replay->probabilities);
    alloc_register(alloc, &replay->cdf);
    alloc_register(alloc, &replay->importance);
    register_rollout_buffers(&replay->batch, alloc, samples, horizon, inputs, actions, mask);
    cudaMalloc((void**)&replay->indices, samples * sizeof(int));
    cudaMalloc((void**)&replay->beta, sizeof(float));
}

__global__ void ReplayWeights(const precision_t* advantages, float* weights,
        int rows, int horizon, float alpha) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) {
        return;
    }
    float sum = 0;
    for (int t = 0; t < horizon; ++t) {
        sum += fabsf(to_float(advantages[row * horizon + t]));
    }
    weights[row] = powf(sum + 1e-6f, alpha);
}

__global__ void ReplayCDF(float* probabilities, float* cdf, int rows) {
    float sum = 0;
    for (int row = 0; row < rows; ++row) {
        sum += probabilities[row];
    }
    float cumulative = 0;
    for (int row = 0; row < rows; ++row) {
        probabilities[row] /= sum;
        cumulative += probabilities[row];
        cdf[row] = cumulative;
    }
}

__global__ void ReplaySample(int* indices, float* importance, const float* cdf,
        const float* probabilities, int rows, int samples, const float* beta,
        unsigned long long seed, const long* offset) {
    int sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= samples) {
        return;
    }
    curandStatePhilox4_32_10_t rng;
    curand_init(seed, (unsigned long long)*offset + sample, 0, &rng);
    float draw = curand_uniform(&rng) * cdf[rows - 1];
    int low = 0, high = rows - 1;
    while (low < high) {
        int mid = (low + high) / 2;
        if (cdf[mid] < draw) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    indices[sample] = low;
    importance[sample] = powf(probabilities[low] * rows, -*beta);
}

__global__ void ReplayAdvance(long* offset, int samples) {
    *offset += samples;
}

template <typename T>
__global__ void ReplayGather(T* dst, const T* src, const int* indices,
        int samples, int width) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < samples * width) {
        dst[index] = src[indices[index / width] * width + index % width];
    }
}

__global__ void ReplayGatherState(Prec dst, Prec src, const int* indices) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int layers = dst.shape[0], samples = dst.shape[1], hidden = dst.shape[2];
    if (index < layers * samples * hidden) {
        int layer = index / (samples * hidden);
        int row = index / hidden % samples;
        int col = index % hidden;
        dst.data[index] = src.data[(layer * src.shape[1] + indices[row]) * hidden + col];
    }
}

void GatherReplay(Replay* replay, const RolloutBuf& source, Prec state,
        cudaStream_t stream) {
    int samples = replay->importance.shape[0];
    int horizon = source.observations.shape[1];
    const Prec* src[] = {&source.observations, &source.logprobs, &source.values,
        &source.rewards, &source.terminals, &source.action_mask};
    Prec* dst[] = {&replay->batch.observations, &replay->batch.logprobs, &replay->batch.values,
        &replay->batch.rewards, &replay->batch.terminals, &replay->batch.action_mask};
    for (int i = 0; i < 6; ++i) {
        int width = src[i]->shape[1] * (src[i]->shape[2] > 0 ? src[i]->shape[2] : 1);
        ReplayGather<<<grid_size(samples * width), BLOCK_SIZE, 0, stream>>>(
            dst[i]->data, src[i]->data, replay->indices, samples, width);
    }
    int action_width = horizon * source.actions.shape[2];
    ReplayGather<<<grid_size(samples * action_width), BLOCK_SIZE, 0, stream>>>(
        replay->batch.actions.data, source.actions.data, replay->indices, samples, action_width);
    ReplayGatherState<<<grid_size(numel(replay->state.shape)), BLOCK_SIZE, 0, stream>>>(
        replay->state, state, replay->indices);
}
