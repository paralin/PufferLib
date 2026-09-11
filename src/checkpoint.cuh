#pragma once

#include <filesystem>
#include <string>
#include <vector>

// A checkpoint directory is published only after every file is closed. Resume
// restores learner state and opponents, then starts fresh environment matches.
namespace checkpoint {
namespace fs = std::filesystem;

void Require(bool ok, const std::string& operation) {
    if (!ok) {
        fprintf(stderr, "checkpoint: %s\n", operation.c_str());
        exit(1);
    }
}

void DeviceFile(const fs::path& path, void* device, size_t bytes, bool write) {
    std::vector<unsigned char> buffer(bytes);
    if (write) {
        Require(cudaMemcpy(buffer.data(), device, bytes, cudaMemcpyDeviceToHost)
            == cudaSuccess, "read device state");
    }
    FILE* file = fopen(path.c_str(), write ? "wb" : "rb");
    Require(file != NULL, "open " + path.string());
    size_t count = write ? fwrite(buffer.data(), 1, bytes, file)
        : fread(buffer.data(), 1, bytes, file);
    Require(count == bytes, "incomplete " + path.string());
    if (!write) {
        Require(fgetc(file) == EOF, "trailing bytes in " + path.string());
    }
    Require(fclose(file) == 0, "close " + path.string());
    if (!write) {
        Require(cudaMemcpy(device, buffer.data(), bytes, cudaMemcpyHostToDevice)
            == cudaSuccess, "restore device state");
    }
}

void WriteIni(const fs::path& path, Ini* ini) {
    FILE* file = fopen(path.c_str(), "w");
    Require(file != NULL, "open " + path.string());
    puf_ini_write(file, ini);
    Require(!ferror(file), "write " + path.string());
    Require(fclose(file) == 0, "close " + path.string());
}

// WriteMetrics publishes one complete native log snapshot for external readers.
void WriteMetrics(const fs::path& path, Dict* values) {
    Ini progress = {};
    Dict* metrics = puf_ini_section(&progress, "metrics", 1);
    for (int i = 0; i < values->size; ++i) {
        dict_set(metrics, values->items[i].key, values->items[i].value);
    }
    WriteIni(path.string() + ".tmp", &progress);
    fs::rename(path.string() + ".tmp", path);
    puf_ini_free(&progress);
}

void CheckLayout(Dict* state, PuffeRL* p, bool write) {
    struct Field { const char* name; long value; };
    Field fields[] = {
        {"version", 1}, {"precision_bytes", sizeof(precision_t)},
        {"rng_state_bytes", sizeof(curandStatePhilox4_32_10_t)},
        {"parameters", numel(p->policies[0].master_weights.shape)},
        {"total_agents", p->vec->total_agents}, {"buffers", p->vec->buffers},
        {"learner_rows", p->vec->policy_layout[1]}, {"policies", p->num_policies},
        {"hidden_size", p->hypers.hidden_size}, {"layers", p->hypers.num_layers},
        {"horizon", p->hypers.horizon}, {"seed", p->seed},
#ifdef __HIP_PLATFORM_AMD__
        {"hip", 1},
#else
        {"hip", 0},
#endif
    };
    for (const Field& field : fields) {
        if (write) {
            dict_set(state, field.name, field.value);
        } else {
            Require(dict_get(state, field.name) == field.value,
                std::string("incompatible ") + field.name);
        }
    }
}

void PolicyFile(const fs::path& path, Policy* policy, cudaStream_t stream, bool write) {
    DeviceFile(path, policy->master_weights.data,
        numel(policy->master_weights.shape) * sizeof(float), write);
    if (!write && USE_BF16) {
        int n = numel(policy->param.shape);
        cast<<<grid_size(n), BLOCK_SIZE, 0, stream>>>(
            policy->param.data, policy->master_weights.data, n);
    }
}

void StateFiles(const fs::path& root, PuffeRL* p, bool write) {
    PolicyFile(root / "weights.f32", &p->policies[0], p->default_stream, write);
    DeviceFile(root / "momentum.f32", p->muon.mb.data,
        numel(p->muon.mb.shape) * sizeof(float), write);
    DeviceFile(root / "rng-offsets.bytes", p->rng_offset,
        (p->vec->buffers + 1) * sizeof(long), write);
    for (int buffer = 0; buffer < p->vec->buffers; ++buffer) {
        DeviceFile(root / ("rng-" + std::to_string(buffer) + ".bytes"), p->rng_states[buffer],
            p->vec->agents_per_buf * sizeof(curandStatePhilox4_32_10_t), write);
    }
    for (int policy = 1; policy < p->num_policies; ++policy) {
        PolicyFile(root / ("opponent-" + std::to_string(policy) + ".f32"),
            &p->policies[policy], p->default_stream, write);
    }
}

void Save(const char* weights_path, Ini* config, PuffeRL* p, Selfplay* pool) {
    Require(p->hypers.world_size == 1, "full-state checkpoints require one GPU");
    Require(cudaDeviceSynchronize() == cudaSuccess, "wait for checkpoint boundary");
    fs::path target(weights_path);
    target.replace_extension(".state");
    fs::path temporary = target.string() + ".tmp." + std::to_string(getpid());
    Require(!fs::exists(target), "checkpoint already exists: " + target.string());
    Require(fs::create_directory(temporary), "create " + temporary.string());
    Ini manifest = {};
    Dict* state = puf_ini_section(&manifest, "state", 1);
    CheckLayout(state, p, true);
    dict_set(state, "epoch", p->epoch);
    dict_set(state, "global_step", p->global_step);
    dict_set(state, "collected_steps", p->collected_steps);
    dict_set(state, "episodes", p->completed_episodes);
    dict_set(state, "pool_size", pool->pool_size);
    dict_set(state, "pool_rng", pool->rng);
    StateFiles(temporary, p, true);
    for (int i = 0; i < pool->pool_size; ++i) {
        fs::copy_file(pool->pool[i], temporary / ("pool-" + std::to_string(i) + ".f32"));
    }
    for (int i = 0; i < pool->num_hist; ++i) {
        std::string key = "opponent_step_" + std::to_string(i);
        dict_set(state, key.c_str(), pool->hist[i].opp_started_step);
    }
    WriteIni(temporary / "config.ini", config);
    WriteIni(temporary / "state.ini", &manifest);
    puf_ini_free(&manifest);
    fs::rename(temporary, target);
    fs::path actor = target.parent_path() / "actor.bin";
    fs::copy_file(weights_path, actor.string() + ".tmp", fs::copy_options::overwrite_existing);
    fs::rename(actor.string() + ".tmp", actor);
}

void Load(const char* directory, PuffeRL* p, Selfplay* pool) {
    Require(p->hypers.world_size == 1, "full-state resume requires one GPU");
    fs::path root(directory);
    Ini manifest = {};
    puf_ini_load_file(&manifest, (root / "state.ini").c_str());
    Dict* state = puf_ini_section(&manifest, "state", 0);
    CheckLayout(state, p, false);
    p->epoch = dict_get(state, "epoch");
    p->global_step = dict_get(state, "global_step");
    p->collected_steps = dict_get(state, "collected_steps");
    p->completed_episodes = dict_get(state, "episodes");
    Require(p->epoch >= 0 && p->global_step >= 0 && p->completed_episodes >= 0,
        "negative training counters");
    StateFiles(root, p, false);
    int count = dict_get(state, "pool_size");
    Require(count >= 0 && count <= pool->max_size, "incompatible opponent pool size");
    pool->pool_size = 0;
    for (int i = 0; i < count; ++i) {
        std::string path = fs::absolute(root / ("pool-" + std::to_string(i) + ".f32")).string();
        selfplay_add_checkpoint(pool, path.c_str());
    }
    pool->rng = dict_get(state, "pool_rng");
    for (int i = 0; i < pool->num_hist; ++i) {
        std::string key = "opponent_step_" + std::to_string(i);
        pool->hist[i].opp_started_step = dict_get(state, key.c_str());
    }
    p->last_log_step = p->global_step;
    Require(cudaDeviceSynchronize() == cudaSuccess, "finish restoring state");
    puf_ini_free(&manifest);
}
} // namespace checkpoint
