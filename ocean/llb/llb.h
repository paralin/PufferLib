#ifndef LLB_ENV_H
#define LLB_ENV_H

#include "llb_shm.h"

#ifndef LLB_FRAME_STACK
#define LLB_FRAME_STACK 1u
#endif
#define LLB_POLICY_OBS_SIZE (LLB_OBS_SIZE * LLB_FRAME_STACK)

typedef struct LlbLog {
    float score;
    float episode_length;
    float terminal_hp_margin;
    float wins;
    float losses;
    float n;
} Log;

typedef struct LlbEnv LlbEnv;
struct LlbEnv {
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    unsigned char* truncations;
    Log log;
    int num_agents;
    unsigned int rng;
    unsigned int slot;
    int action_position;
    int result_position;
    int fd;
    LlbShm* shm;
};
typedef LlbEnv Env;

void c_reset(Env* env);
void c_step(Env* env);
void c_close(Env* env);
void c_render(Env* env);
void llb_init(Env* env);

#endif
