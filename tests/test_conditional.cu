#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define NUM_ATNS 15
#define PUF_CONDITIONAL_GROUPS {3, 3, 5, 2, 2}
#define PUF_CONDITIONAL_TIMES 10
using precision_t = float;
__device__ float to_float(float value) { return value; }
#include "conditional.cuh"

constexpr int kCases = 8;
constexpr int kWidth = ConditionalWidth();
constexpr float kPolicyDerivative = 0.7f;
constexpr float kEntropyDerivative = -0.03f;

__global__ void Evaluate(const float* logits, const float* actions,
        float* gradients, float* scores) {
    int row = threadIdx.x;
    if (row >= kCases) {
        return;
    }
    float* gradient = gradients + row * kWidth;
    scores[2 * row] = ConditionalLogProb(logits + row * kWidth,
        actions + row * NUM_ATNS, gradient);
    scores[2 * row + 1] = ConditionalGradient(gradient, actions + row * NUM_ATNS,
        kPolicyDerivative, kEntropyDerivative);
}

std::vector<double> Probabilities(const std::vector<double>& logits, int offset, int count) {
    double total = 0;
    std::vector<double> result(count);
    for (int i = 0; i < count; ++i) {
        result[i] = std::exp(logits[offset + i]);
        total += result[i];
    }
    for (double& value : result) {
        value /= total;
    }
    return result;
}

// Enumerate the joint distribution independently of the device's chain-rule gradient.
double Reference(const std::vector<double>& logits, const float* actions,
        double* log_probability, double* entropy) {
    *log_probability = 0;
    *entropy = 0;
    int offset = 0;
    for (int group = 0; group < kConditionalGroupCount; ++group) {
        int size = kConditionalGroups[group];
        int final_base = offset + size;
        int time_base = final_base + size * size;
        auto initials = Probabilities(logits, offset, size);
        for (int initial = 0; initial < size; ++initial) {
            auto finals = Probabilities(logits, final_base + initial * size, size);
            for (int final = 0; final < size; ++final) {
                auto times = Probabilities(logits,
                    time_base + (initial * size + final) * kConditionalTimes,
                    kConditionalTimes);
                for (int time = 0; time < kConditionalTimes; ++time) {
                    double mass = initials[initial] * finals[final] * times[time];
                    *entropy -= mass * std::log(mass);
                    if (initial == actions[3 * group] && final == actions[3 * group + 1]
                            && time == actions[3 * group + 2]) {
                        *log_probability += std::log(mass);
                    }
                }
            }
        }
        offset = time_base + size * size * kConditionalTimes;
    }
    return kPolicyDerivative * *log_probability + kEntropyDerivative * *entropy;
}

void Check(cudaError_t status) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
        std::exit(1);
    }
}

int main() {
    float *logits, *actions, *gradients, *scores;
    Check(cudaMallocManaged(&logits, kCases * kWidth * sizeof(float)));
    Check(cudaMallocManaged(&actions, kCases * NUM_ATNS * sizeof(float)));
    Check(cudaMallocManaged(&gradients, kCases * kWidth * sizeof(float)));
    Check(cudaMallocManaged(&scores, kCases * 2 * sizeof(float)));
    for (int row = 0; row < kCases; ++row) {
        for (int col = 0; col < kWidth; ++col) {
            logits[row * kWidth + col] = row == 0 ? 0 :
                4 * std::sin(float(col * 13 + row * 7));
        }
        for (int group = 0; group < kConditionalGroupCount; ++group) {
            actions[row * NUM_ATNS + 3 * group] = (row + group) % kConditionalGroups[group];
            actions[row * NUM_ATNS + 3 * group + 1] = (row * 3 + group) % kConditionalGroups[group];
            actions[row * NUM_ATNS + 3 * group + 2] = (row * 7 + group) % kConditionalTimes;
        }
    }
    Evaluate<<<1, kCases>>>(logits, actions, gradients, scores);
    Check(cudaGetLastError());
    Check(cudaDeviceSynchronize());
    double gradient_error = 0, score_error = 0;
    for (int row = 0; row < kCases; ++row) {
        std::vector<double> input(logits + row * kWidth, logits + (row + 1) * kWidth);
        const float* action = actions + row * NUM_ATNS;
        double logp, entropy;
        Reference(input, action, &logp, &entropy);
        score_error = std::fmax(score_error, std::fabs(logp - scores[2 * row]));
        score_error = std::fmax(score_error, std::fabs(entropy - scores[2 * row + 1]));
        for (int col = 0; col < kWidth; ++col) {
            constexpr double epsilon = 1e-4;
            input[col] += epsilon;
            double plus = Reference(input, action, &logp, &entropy);
            input[col] -= 2 * epsilon;
            double minus = Reference(input, action, &logp, &entropy);
            input[col] += epsilon;
            double expected = (plus - minus) / (2 * epsilon);
            gradient_error = std::fmax(gradient_error,
                std::fabs(expected - gradients[row * kWidth + col]));
        }
    }
    std::printf("conditional cases=%d logits=%d gradient_max_error=%.9g score_max_error=%.9g\n",
        kCases, kWidth, gradient_error, score_error);
    Check(cudaFree(logits));
    Check(cudaFree(actions));
    Check(cudaFree(gradients));
    Check(cudaFree(scores));
    return gradient_error < 2e-5 && score_error < 2e-5 ? 0 : 1;
}
