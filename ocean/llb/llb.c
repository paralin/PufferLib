#define _GNU_SOURCE
#include "llb.h"

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>
#include <sys/mman.h>
#include <time.h>
#include <sys/stat.h>
#include <unistd.h>

static void llb_fail(const char* operation, const Env* env) {
    fprintf(stderr, "llb shm slot %u %s failed: %s\n", env->slot, operation,
            strerror(errno));
    abort();
}

/* Busy sched_yield loops from every rollout thread starve the batched
 * host's service threads (measured: 91 ms/window service under load).
 * Brief spins keep the fast path tight; a short sleep releases the core
 * so the producer can run. */
static void llb_backoff(int* spins) {
    if (*spins < 64) {
        sched_yield();
        (*spins)++;
    } else {
        const struct timespec pause = {0, 250000}; /* 250 us */
        nanosleep(&pause, NULL);
    }
}

static void llb_wait_action(LlbAction* entry, uint64_t sequence,
                            const Env* env) {
    int spins = 0;
    while (atomic_load_explicit(&entry->sequence, memory_order_acquire)
           != sequence) {
        llb_backoff(&spins);
        if (env->shm->magic != LLB_SHM_MAGIC) {
            fprintf(stderr, "llb shm slot %u disappeared\n", env->slot);
            abort();
        }
    }
}

static void llb_wait_result(LlbResult* entry, uint64_t sequence,
                            const Env* env) {
    int spins = 0;
    while (atomic_load_explicit(&entry->sequence, memory_order_acquire)
           != sequence) {
        llb_backoff(&spins);
        if (env->shm->magic != LLB_SHM_MAGIC) {
            fprintf(stderr, "llb shm slot %u disappeared\n", env->slot);
            abort();
        }
    }
}

static int llb_exchange(Env* env, uint32_t command) {
    if (command != LLB_RESET && command != LLB_STEP) {
        fprintf(stderr, "llb shm slot %u unknown command %u\n",
                env->slot, command);
        abort();
    }
    uint64_t action_pos = (uint64_t)env->action_position;
    uint64_t result_pos = (uint64_t)env->result_position;
    LlbAction* action = &env->shm->actions[action_pos % LLB_RING_CAPACITY];
    LlbResult* result = &env->shm->results[result_pos % LLB_RING_CAPACITY];

    llb_wait_action(action, action_pos, env);
    action->command = command;
    if (command == LLB_RESET) {
        memset(action->actions, 0, sizeof(action->actions));
    } else {
        memcpy(action->actions, env->actions, sizeof(action->actions));
    }
    atomic_store_explicit(&action->sequence, action_pos + 1,
                          memory_order_release);
    env->action_position++;

    llb_wait_result(result, result_pos + 1, env);
    memmove(env->observations, env->observations + LLB_OBS_SIZE,
            (LLB_POLICY_OBS_SIZE - LLB_OBS_SIZE) * sizeof(float));
    memcpy(env->observations + LLB_POLICY_OBS_SIZE - LLB_OBS_SIZE,
           result->observations, sizeof(result->observations));
    *env->rewards = result->reward;
    *env->terminals = (result->terminated || result->truncated) ? 1.0f : 0.0f;
    if (command == LLB_STEP) {
        env->log.score += result->reward;
        env->log.episode_length += 1.0f;
        if (result->terminated || result->truncated) {
            /* GymHost keeps agent and opponent HP at indices 10 and 28;
             * capture them before c_step exchanges the reset observation. */
            size_t latest = LLB_POLICY_OBS_SIZE - LLB_OBS_SIZE;
            float margin = env->observations[latest + 10]
                - env->observations[latest + 28];
            env->log.terminal_hp_margin += margin;
            if (margin > 0.0f) {
                env->log.wins += 1.0f;
            } else if (margin < 0.0f) {
                env->log.losses += 1.0f;
            }
            env->log.n += 1.0f;
        }
    }
    atomic_store_explicit(&result->sequence, result_pos + LLB_RING_CAPACITY,
                          memory_order_release);
    env->result_position++;
    return result->terminated || result->truncated;
}

void c_reset(Env* env) {
    *env->rewards = 0.0f;
    *env->terminals = 0.0f;
    llb_exchange(env, LLB_RESET);
    for (size_t frame = 0; frame + 1 < LLB_FRAME_STACK; frame++) {
        memcpy(env->observations + frame * LLB_OBS_SIZE,
               env->observations + LLB_POLICY_OBS_SIZE - LLB_OBS_SIZE,
               LLB_OBS_SIZE * sizeof(float));
    }
    *env->rewards = 0.0f;
    *env->terminals = 0.0f;
}

void c_step(Env* env) {
    int done = llb_exchange(env, LLB_STEP);
    if (done) {
        float reward = *env->rewards;
        float terminal = *env->terminals;
        llb_exchange(env, LLB_RESET);
        *env->rewards = reward;
        *env->terminals = terminal;
    }
}

void c_close(Env* env) {
    if (env->shm == NULL) {
        return;
    }
    size_t size = sizeof(*env->shm);
    if (munmap(env->shm, size) != 0) {
        llb_fail("unmap", env);
    }
    if (env->fd >= 0) {
        close(env->fd);
    }
    env->shm = NULL;
    env->fd = -1;
}

void c_render(Env* env) {
    (void)env;
}

static void llb_attach(Env* env) {
    const char* directory = getenv("LLB_SHM_DIR");
    if (directory == NULL || directory[0] == '\0') {
        fprintf(stderr, "LLB_SHM_DIR is required\n");
        abort();
    }
    char path[4096];
    int length = snprintf(path, sizeof(path), "%s/slot-%u.bin", directory,
                          env->slot);
    if (length < 0 || (size_t)length >= sizeof(path)) {
        fprintf(stderr, "LLB_SHM_DIR path is too long\n");
        abort();
    }
    env->fd = open(path, O_RDWR | O_CLOEXEC);
    if (env->fd < 0) {
        llb_fail("open", env);
    }
    struct stat status;
    if (fstat(env->fd, &status) != 0) {
        llb_fail("stat", env);
    }
    if ((uintmax_t)status.st_size != sizeof(*env->shm)) {
        fprintf(stderr, "llb shm slot %u has incompatible layout "
                "(got %jd bytes, need %zu)\n", env->slot,
                (intmax_t)status.st_size, sizeof(*env->shm));
        abort();
    }
    env->shm = mmap(NULL, sizeof(*env->shm), PROT_READ | PROT_WRITE,
                    MAP_SHARED, env->fd, 0);
    if (env->shm == MAP_FAILED) {
        env->shm = NULL;
        llb_fail("map", env);
    }
    if (env->shm->magic != LLB_SHM_MAGIC
        || env->shm->version != LLB_SHM_VERSION) {
        fprintf(stderr, "llb shm slot %u has incompatible header\n", env->slot);
        abort();
    }
}

/* Included by binding.c after vecenv.h defines Env and the static callbacks. */
void llb_init(Env* env) {
    env->slot = env->rng;
    env->action_position = 0;
    env->result_position = 0;
    env->fd = -1;
    env->shm = NULL;
    llb_attach(env);
}
