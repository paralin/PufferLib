// Puffer static env binding over the play-llb-sim Rust kernel cdylib.
#include <dlfcn.h>

#define OBS_SIZE 66
#define NUM_ATNS 15
#define ACT_SIZES {3, 3, 10, 3, 3, 10, 5, 5, 10, 2, 2, 10, 2, 2, 10}
#define OBS_TENSOR_T FloatTensor

typedef struct LlbRustLog {
    float score;
    float episode_length;
    float terminal_hp_margin;
    float wins;
    float losses;
    float n;
} Log;

typedef struct LlbRustEnv LlbRustEnv;
struct LlbRustEnv {
    void* observations;
    float* actions;
    float* rewards;
    float* terminals;
    Log log;
    int num_agents;
    unsigned int rng;
    void* handle;
};
typedef LlbRustEnv Env;

void c_reset(Env* env);
void c_step(Env* env);
void c_close(Env* env);
void c_render(Env* env);

#include "vecenv.h"

typedef struct PufferEnv PufferEnv;
typedef PufferEnv* (*puffer_env_new_auto_fn)(unsigned long n, unsigned int seed,
    int cpu_seat, int auto_reset);
typedef int (*puffer_env_free_fn)(PufferEnv* env);
typedef float* (*puffer_env_obs_ptr_fn)(PufferEnv* env);
typedef unsigned char* (*puffer_env_actions_ptr_fn)(PufferEnv* env);
typedef float* (*puffer_env_rewards_ptr_fn)(PufferEnv* env);
typedef unsigned char* (*puffer_env_terminated_ptr_fn)(PufferEnv* env);
typedef unsigned char* (*puffer_env_truncated_ptr_fn)(PufferEnv* env);
typedef int (*puffer_env_reset_fn)(PufferEnv* env);
typedef int (*puffer_env_reset_one_fn)(PufferEnv* env, unsigned long slot);
typedef int (*puffer_env_step_fn)(PufferEnv* env);

static puffer_env_new_auto_fn llb_puffer_env_new_auto;
static puffer_env_free_fn llb_puffer_env_free;
static puffer_env_obs_ptr_fn llb_puffer_env_obs_ptr;
static puffer_env_actions_ptr_fn llb_puffer_env_actions_ptr;
static puffer_env_rewards_ptr_fn llb_puffer_env_rewards_ptr;
static puffer_env_terminated_ptr_fn llb_puffer_env_terminated_ptr;
static puffer_env_truncated_ptr_fn llb_puffer_env_truncated_ptr;
static puffer_env_reset_fn llb_puffer_env_reset;
static puffer_env_reset_one_fn llb_puffer_env_reset_one;
static puffer_env_step_fn llb_puffer_env_step;

static void* llb_rust_library(void) {
    static void* library;
    static int resolved;
    if (resolved) {
        return library;
    }
    resolved = 1;
    const char* path = getenv("LLB_RUST_LIB");
    if (path == NULL || path[0] == '\0') {
        fprintf(stderr, "llb-rust: LLB_RUST_LIB must name the built libplay_llb_sim cdylib\n");
        return NULL;
    }
    library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (library == NULL) {
        fprintf(stderr, "llb-rust: dlopen(%s) failed: %s\n", path, dlerror());
        return NULL;
    }
    struct { const char* name; void** target; } symbols[] = {
        {"puffer_env_new_auto", (void**)&llb_puffer_env_new_auto},
        {"puffer_env_free", (void**)&llb_puffer_env_free},
        {"puffer_env_obs_ptr", (void**)&llb_puffer_env_obs_ptr},
        {"puffer_env_actions_ptr", (void**)&llb_puffer_env_actions_ptr},
        {"puffer_env_rewards_ptr", (void**)&llb_puffer_env_rewards_ptr},
        {"puffer_env_terminated_ptr", (void**)&llb_puffer_env_terminated_ptr},
        {"puffer_env_truncated_ptr", (void**)&llb_puffer_env_truncated_ptr},
        {"puffer_env_reset", (void**)&llb_puffer_env_reset},
        {"puffer_env_reset_one", (void**)&llb_puffer_env_reset_one},
        {"puffer_env_step", (void**)&llb_puffer_env_step},
    };
    for (size_t i = 0; i < sizeof(symbols) / sizeof(symbols[0]); i++) {
        void* symbol = dlsym(library, symbols[i].name);
        if (symbol == NULL) {
            fprintf(stderr, "llb-rust: missing symbol %s: %s\n", symbols[i].name, dlerror());
            dlclose(library);
            library = NULL;
            return NULL;
        }
        *symbols[i].target = symbol;
    }
    return library;
}

static void llb_copy_obs(Env* env) {
    memcpy(env->observations, llb_puffer_env_obs_ptr(env->handle), OBS_SIZE * sizeof(float));
}

void c_reset(Env* env) {
    llb_puffer_env_reset(env->handle);
    *env->rewards = 0.0f;
    *env->terminals = 0.0f;
    llb_copy_obs(env);
}

void c_step(Env* env) {
    unsigned char* actions = llb_puffer_env_actions_ptr(env->handle);
    for (int i = 0; i < NUM_ATNS; i++) {
        actions[i] = (unsigned char)env->actions[i];
    }
    llb_puffer_env_step(env->handle);

    float reward = *llb_puffer_env_rewards_ptr(env->handle);
    unsigned char terminated = *llb_puffer_env_terminated_ptr(env->handle);
    unsigned char truncated = *llb_puffer_env_truncated_ptr(env->handle);

    llb_copy_obs(env);
    *env->rewards = reward;
    *env->terminals = (terminated || truncated) ? 1.0f : 0.0f;

    env->log.score += reward;
    env->log.episode_length += 1.0f;
    if (terminated || truncated) {
        float margin = env->observations[10] - env->observations[28];
        env->log.terminal_hp_margin += margin;
        if (margin > 0.0f) {
            env->log.wins += 1.0f;
        } else if (margin < 0.0f) {
            env->log.losses += 1.0f;
        }
        env->log.n += 1.0f;
        llb_puffer_env_reset_one(env->handle, 0);
        llb_copy_obs(env);
    }
}

void c_close(Env* env) {
    if (env->handle == NULL) {
        return;
    }
    llb_puffer_env_free(env->handle);
    env->handle = NULL;
}

void c_render(Env* env) {
    (void)env;
}

void my_init(Env* env, Dict* kwargs) {
    (void)kwargs;
    env->num_agents = 1;
    memset(&env->log, 0, sizeof(env->log));
    env->handle = NULL;
    if (llb_rust_library() == NULL) {
        abort();
    }
    env->handle = llb_puffer_env_new_auto(1, env->rng, 0, 0);
    if (env->handle == NULL) {
        fprintf(stderr, "llb-rust: puffer_env_new_auto failed\n");
        abort();
    }
    c_reset(env);
}

void my_log(Log* log, Dict* out) {
    dict_set(out, "score", log->score);
    dict_set(out, "episode_length", log->episode_length);
    dict_set(out, "terminal_hp_margin", log->terminal_hp_margin);
    dict_set(out, "wins", log->wins);
    dict_set(out, "losses", log->losses);
    dict_set(out, "n", log->n);
}
