#pragma once

// Conditional controls use p(initial) p(final|initial) p(time|initial,final).
// Each group stores S + S*S + S*S*K logits; the environment supplies S and K.
#ifdef PUF_CONDITIONAL_GROUPS
constexpr int kConditionalGroups[] = PUF_CONDITIONAL_GROUPS;
constexpr int kConditionalGroupCount = sizeof(kConditionalGroups) / sizeof(int);
constexpr int kConditionalTimes = PUF_CONDITIONAL_TIMES;
static_assert(NUM_ATNS == 3 * kConditionalGroupCount);

constexpr int ConditionalWidth() {
    int width = 0;
    for (int size : kConditionalGroups) {
        width += size + size * size * (1 + kConditionalTimes);
    }
    return width;
}

constexpr int ConditionalMaxGroup() {
    int maximum = 0;
    for (int size : kConditionalGroups) {
        maximum = size > maximum ? size : maximum;
    }
    return maximum;
}

// ConditionalGroupSize returns an environment-defined control alphabet size.
__device__ int ConditionalGroupSize(int group) {
    constexpr int sizes[] = PUF_CONDITIONAL_GROUPS;
    return sizes[group];
}

// ConditionalTable writes normalized log probabilities for one table.
__device__ void ConditionalTable(const precision_t* logits, float* logps, int n) {
    float maximum = -INFINITY;
    for (int i = 0; i < n; ++i) {
        maximum = fmaxf(maximum, to_float(logits[i]));
    }
    float sum = 0;
    for (int i = 0; i < n; ++i) {
        sum += expf(to_float(logits[i]) - maximum);
    }
    float normalizer = maximum + logf(sum);
    for (int i = 0; i < n; ++i) {
        logps[i] = to_float(logits[i]) - normalizer;
    }
}

// ConditionalLogProb caches every table and scores the sampled factor chain.
__device__ float ConditionalLogProb(const precision_t* logits,
        const float* actions, float* logps) {
    float result = 0;
    int offset = 0;
    for (int group = 0; group < kConditionalGroupCount; ++group) {
        int size = ConditionalGroupSize(group);
        int final_base = offset + size;
        int time_base = final_base + size * size;
        ConditionalTable(logits + offset, logps + offset, size);
        for (int initial = 0; initial < size; ++initial) {
            int row = final_base + initial * size;
            ConditionalTable(logits + row, logps + row, size);
            for (int final = 0; final < size; ++final) {
                row = time_base + (initial * size + final) * kConditionalTimes;
                ConditionalTable(logits + row, logps + row, kConditionalTimes);
            }
        }
        int initial = (int)actions[3 * group];
        int final = (int)actions[3 * group + 1];
        int time = (int)actions[3 * group + 2];
        result += logps[offset + initial] + logps[final_base + initial * size + final]
            + logps[time_base + (initial * size + final) * kConditionalTimes + time];
        offset = time_base + size * size * kConditionalTimes;
    }
    return result;
}

// ConditionalDraw samples one table and accumulates its selected log mass.
__device__ int ConditionalDraw(const precision_t* logits, int n,
        curandStatePhilox4_32_10_t* rng, float* logp) {
    constexpr int cache_size = ConditionalMaxGroup() > kConditionalTimes
        ? ConditionalMaxGroup() : kConditionalTimes;
    float cache[cache_size];
    ConditionalTable(logits, cache, n);
    float draw = curand_uniform(rng);
    float cumulative = 0;
    int selected = n - 1;
    for (int i = 0; i < n; ++i) {
        cumulative += expf(cache[i]);
        if (draw < cumulative) {
            selected = i;
            break;
        }
    }
    *logp += cache[selected];
    return selected;
}

// ConditionalSample writes the complete factor chain and its log probability.
__device__ float ConditionalSample(const precision_t* logits, float* actions,
        curandStatePhilox4_32_10_t* rng) {
    float logp = 0;
    int offset = 0;
    for (int group = 0; group < kConditionalGroupCount; ++group) {
        int size = ConditionalGroupSize(group);
        int initial = ConditionalDraw(logits + offset, size, rng, &logp);
        int final_base = offset + size;
        int final = ConditionalDraw(logits + final_base + initial * size, size, rng, &logp);
        int time_base = final_base + size * size;
        int time = ConditionalDraw(logits + time_base + (initial * size + final)
            * kConditionalTimes, kConditionalTimes, rng, &logp);
        actions[3 * group] = initial;
        actions[3 * group + 1] = final;
        actions[3 * group + 2] = time;
        offset = time_base + size * size * kConditionalTimes;
    }
    return logp;
}

// ConditionalEntropy returns the entropy of one normalized categorical table.
__device__ float ConditionalEntropy(const float* logps, int size) {
    float result = 0;
    for (int i = 0; i < size; ++i) {
        result -= expf(logps[i]) * logps[i];
    }
    return result;
}

// ConditionalGradient replaces cached log probabilities with exact chain-rule
// gradients. Entropy includes every conditional table, weighted by its prefix.
__device__ float ConditionalGradient(float* logps, const float* actions,
        float d_logp, float d_entropy) {
    float total_entropy = 0;
    int offset = 0;
    for (int group = 0; group < kConditionalGroupCount; ++group) {
        int size = ConditionalGroupSize(group);
        int selected_initial = (int)actions[3 * group];
        int selected_final = (int)actions[3 * group + 1];
        int selected_time = (int)actions[3 * group + 2];
        int final_base = offset + size;
        int time_base = final_base + size * size;
        float initial_entropy = ConditionalEntropy(logps + offset, size);
        float downstream[ConditionalMaxGroup()] = {};
        float expected_downstream = 0;
        for (int initial = 0; initial < size; ++initial) {
            float initial_probability = expf(logps[offset + initial]);
            int final_row = final_base + initial * size;
            float final_entropy = ConditionalEntropy(logps + final_row, size);
            float time_entropies[ConditionalMaxGroup()] = {};
            float expected_time_entropy = 0;
            for (int final = 0; final < size; ++final) {
                float final_probability = expf(logps[final_row + final]);
                int time_row = time_base + (initial * size + final) * kConditionalTimes;
                float time_entropy = ConditionalEntropy(logps + time_row, kConditionalTimes);
                time_entropies[final] = time_entropy;
                expected_time_entropy += final_probability * time_entropy;
                float prefix = initial_probability * final_probability;
                for (int time = 0; time < kConditionalTimes; ++time) {
                    float lp = logps[time_row + time];
                    float probability = expf(lp);
                    float pg = initial == selected_initial && final == selected_final
                        ? d_logp * ((time == selected_time ? 1.0f : 0.0f) - probability) : 0;
                    logps[time_row + time] = pg
                        + d_entropy * prefix * probability * (-time_entropy - lp);
                }
            }
            downstream[initial] = final_entropy + expected_time_entropy;
            expected_downstream += initial_probability * downstream[initial];
            for (int final = 0; final < size; ++final) {
                float lp = logps[final_row + final];
                float probability = expf(lp);
                float pg = initial == selected_initial
                    ? d_logp * ((final == selected_final ? 1.0f : 0.0f) - probability) : 0;
                logps[final_row + final] = pg + d_entropy * initial_probability * probability
                    * (-final_entropy - lp + time_entropies[final] - expected_time_entropy);
            }
        }
        total_entropy += initial_entropy + expected_downstream;
        for (int initial = 0; initial < size; ++initial) {
            float lp = logps[offset + initial];
            float probability = expf(lp);
            logps[offset + initial] = d_logp
                * ((initial == selected_initial ? 1.0f : 0.0f) - probability)
                + d_entropy * probability * (-initial_entropy - lp
                    + downstream[initial] - expected_downstream);
        }
        offset = time_base + size * size * kConditionalTimes;
    }
    return total_entropy;
}
#endif
