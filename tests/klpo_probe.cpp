#include <cmath>
#ifdef KLPO_HIP
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>
using curandStatePhilox4_32_10_t = hiprandStatePhilox4_32_10_t;
#define curand_uniform hiprand_uniform
#else
#define __device__
struct curandStatePhilox4_32_10_t {};
float curand_uniform(curandStatePhilox4_32_10_t*) { return 0.5f; }
#endif
#define NUM_ATNS 15
#define PUF_CONDITIONAL_GROUPS {3, 3, 5, 2, 2}
#define PUF_CONDITIONAL_TIMES 10
using precision_t = float;
__device__ float to_float(float x) { return x; }
#include "conditional.cuh"
#include "klpo.cuh"

__device__ void Score(const float* p, const float* q, const float* action,
        float* gradient, float* stats) {
    float pl[ConditionalWidth()], ql[ConditionalWidth()];
    ConditionalLogProb(p, action, pl);
    ConditionalLogProb(q, action, ql);
    KlpoScore s = KlpoConditionalScore(pl, ql, action, gradient);
    stats[0] = s.centered;
    stats[1] = s.log_ratio;
    stats[2] = s.entropy;
    stats[3] = s.kl;
}
#ifdef KLPO_HIP
__global__ void Probe(const float* input, float* output) {
    Score(input, input + 576, input + 1152, output, output + 576);
}
extern "C" int score(const float* p, const float* q, const float* actions,
        float* gradient, float* stats) {
    float *input, *output;
    if (hipMalloc(&input, 1167 * sizeof(float)) != hipSuccess) return 1;
    if (hipMalloc(&output, 580 * sizeof(float)) != hipSuccess) return 2;
    hipMemcpy(input, p, 576 * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(input + 576, q, 576 * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(input + 1152, actions, 15 * sizeof(float), hipMemcpyHostToDevice);
    Probe<<<1, 1>>>(input, output);
    auto status = hipDeviceSynchronize();
    hipMemcpy(gradient, output, 576 * sizeof(float), hipMemcpyDeviceToHost);
    hipMemcpy(stats, output + 576, 4 * sizeof(float), hipMemcpyDeviceToHost);
    hipFree(input);
    hipFree(output);
    return status != hipSuccess;
}
#else
extern "C" int score(const float* p, const float* q, const float* actions,
        float* gradient, float* stats) {
    Score(p, q, actions, gradient, stats);
    return 0;
}
extern "C" void targets(const float* rewards, const float* dones, float* result,
        int horizon, int end_step, float gamma, bool whole_match, int* counts) {
    KlpoTargets(rewards, dones, result, horizon, end_step, gamma, whole_match, counts);
}
#endif
