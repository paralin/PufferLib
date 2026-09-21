#pragma once

// KLPO's token-regression score uses the historical sampler, without an
// action importance ratio: (R - beta log(p/q)) stop-grad times centered log p.
// Conditional controls are scored after merging encodings of the same schedule.
#ifdef PUF_CONDITIONAL_GROUPS
__device__ float KlpoLogAdd(float a, float b) {
    float maximum = fmaxf(a, b);
    return maximum == -INFINITY ? maximum : maximum + log1pf(expf(-fabsf(a - b)));
}

// Only constants have aliases: switching at zero executes the final control,
// and identical initial/final controls execute a constant at every switch time.
__device__ void KlpoConstants(const float* lp, int size, float* constants) {
    int times = size + size * size;
    for (int final = 0; final < size; ++final) {
        float mass = -INFINITY;
        for (int initial = 0; initial < size; ++initial) {
            float prefix = lp[initial] + lp[size + initial * size + final];
            int count = initial == final ? kConditionalTimes : 1;
            for (int time = 0; time < count; ++time) {
                mass = KlpoLogAdd(mass, prefix
                    + lp[times + (initial * size + final) * kConditionalTimes + time]);
            }
        }
        constants[final] = mass;
    }
}

struct KlpoScore {
    float centered, log_ratio, entropy, kl;
};

// Writes d(centered log p)/d(logits). The quotient posterior distributes each
// schedule's derivative across its raw aliases, then each softmax is chained.
__device__ KlpoScore KlpoConditionalScore(const float* p, const float* q,
        const float* actions, float* gradient) {
    KlpoScore score = {};
    int offset = 0;
    for (int group = 0; group < kConditionalGroupCount; ++group) {
        int size = ConditionalGroupSize(group);
        int times = size + size * size;
        float pc[ConditionalMaxGroup()], qc[ConditionalMaxGroup()];
        KlpoConstants(p + offset, size, pc);
        KlpoConstants(q + offset, size, qc);
        int si = (int)actions[3 * group], sf = (int)actions[3 * group + 1];
        int sk = (int)actions[3 * group + 2];
        bool constant = si == sf || sk == 0;
        int selected_time = times + (si * size + sf) * kConditionalTimes + sk;
        float selected_p = constant ? pc[sf] : p[offset + si]
            + p[offset + size + si * size + sf] + p[offset + selected_time];
        float selected_q = constant ? qc[sf] : q[offset + si]
            + q[offset + size + si * size + sf] + q[offset + selected_time];
        score.centered += selected_p;
        score.log_ratio += selected_p - selected_q;
        float total_weight = 0;
        for (int initial = 0; initial < size; ++initial) {
            float initial_weight = 0;
            int final_row = offset + size + initial * size;
            for (int final = 0; final < size; ++final) {
                int time_row = offset + times + (initial * size + final) * kConditionalTimes;
                float final_weight = 0;
                for (int time = 0; time < kConditionalTimes; ++time) {
                    float pa = p[offset + initial] + p[final_row + final] + p[time_row + time];
                    float qa = q[offset + initial] + q[final_row + final] + q[time_row + time];
                    bool alias = initial == final || time == 0;
                    float pm = alias ? pc[final] : pa;
                    float qm = alias ? qc[final] : qa;
                    bool selected = constant ? alias && final == sf
                        : !alias && initial == si && final == sf && time == sk;
                    float weight = ((selected ? 1.0f : 0.0f) - expf(qm)) * expf(pa - pm);
                    gradient[time_row + time] = weight;
                    final_weight += weight;
                    score.centered -= expf(qa) * pm;
                    score.entropy -= expf(pa) * pm;
                    score.kl += expf(qa) * (qm - pm);
                }
                for (int time = 0; time < kConditionalTimes; ++time) {
                    gradient[time_row + time] -= expf(p[time_row + time]) * final_weight;
                }
                gradient[final_row + final] = final_weight;
                initial_weight += final_weight;
            }
            for (int final = 0; final < size; ++final) {
                gradient[final_row + final] -= expf(p[final_row + final]) * initial_weight;
            }
            gradient[offset + initial] = initial_weight;
            total_weight += initial_weight;
        }
        for (int initial = 0; initial < size; ++initial) {
            gradient[offset + initial] -= expf(p[offset + initial]) * total_weight;
        }
        offset += times + size * size * kConditionalTimes;
    }
    return score;
}
#endif

// Rewards/dones at t describe action t-1. Collection begins at a real reset
// and drains through the terminal at end_step. Only padding receives NaN.
__device__ void KlpoTargets(const precision_t* rewards, const precision_t* dones,
        float* targets, int horizon, int end_step, float gamma, bool whole_match, int* counts) {
    for (int t = 0; t < horizon; ++t) {
        targets[t] = NAN;
    }
    int start = 0;
    counts[0] = counts[1] = 0;
    for (int end = 1; end <= end_step; ++end) {
        if (to_float(dones[end]) == 0) {
            continue;
        }
        float value = 0;
        for (int t = end - 1; t >= start; --t) {
            value = to_float(rewards[t + 1]) + (whole_match ? 1.0f : gamma) * value;
            targets[t] = value;
        }
        if (whole_match) {
            for (int t = start; t < end; ++t) {
                targets[t] = value;
            }
        }
        counts[0] += 1;
        counts[1] += end - start;
        start = end;
    }
}
