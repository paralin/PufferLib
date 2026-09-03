#include "llb.h"

#define OBS_SIZE LLB_POLICY_OBS_SIZE
#define NUM_ATNS LLB_NUM_ACTIONS
#define ACT_SIZES {3, 3, 10, 3, 3, 10, 5, 5, 10, 2, 2, 10, 2, 2, 10}
#define OBS_TENSOR_T FloatTensor
#define Env LlbEnv
#include "vecenv.h"
#include "llb.c"

void my_init(Env* env, Dict* kwargs) {
    (void)kwargs;
    env->num_agents = 1;
    memset(&env->log, 0, sizeof(env->log));
    llb_init(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "terminal_hp_margin", log->terminal_hp_margin);
    dict_set(out, "wins", log->wins);
    dict_set(out, "losses", log->losses);
    dict_set(out, "n", log->n);
}
