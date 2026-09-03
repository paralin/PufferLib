#ifndef LLB_SHM_H
#define LLB_SHM_H

#include <stdint.h>
#include <stdatomic.h>

#define LLB_SHM_MAGIC UINT64_C(0x4c4c4253484d3031)
#define LLB_SHM_VERSION 4u
#define LLB_OBS_SIZE 66u
#define LLB_NUM_ACTIONS 15u
#define LLB_RING_CAPACITY 64u

enum LlbCommand { LLB_RESET = 0, LLB_STEP = 1 };

typedef struct {
    _Atomic uint64_t sequence;
    uint32_t command;
    uint32_t reserved;
    float actions[LLB_NUM_ACTIONS];
} LlbAction;

typedef struct {
    _Atomic uint64_t sequence;
    float observations[LLB_OBS_SIZE];
    float reward;
    uint32_t terminated;
    uint32_t truncated;
    uint32_t frame;
} LlbResult;

typedef struct {
    uint64_t magic;
    uint32_t version;
    uint32_t reserved;
    LlbAction actions[LLB_RING_CAPACITY];
    LlbResult results[LLB_RING_CAPACITY];
} LlbShm;

#endif
