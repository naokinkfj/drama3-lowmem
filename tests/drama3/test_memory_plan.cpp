// Automatic expert placement: the fewest leading expert layers go to system
// RAM so that the estimated GPU peak fits the free VRAM or the user's budget.
#include "engine/community_models/fish_drama3/memory_plan.h"

#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using engine::models::fish_drama3::ExpertPlacementRequest;
using engine::models::fish_drama3::parse_cpu_expert_layers;
using engine::models::fish_drama3::plan_expert_placement;

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { ++failures; std::fprintf(stderr, "FAIL line %d: ", __LINE__); std::fprintf(stderr, __VA_ARGS__); std::fprintf(stderr, "\n"); } } while (0)

static constexpr int64_t GiB = 1024LL * 1024 * 1024;

static ExpertPlacementRequest uniform_request(int64_t free_gib) {
    ExpertPlacementRequest request;
    request.vram_known = true;
    request.free_vram_bytes = free_gib * GiB;
    request.expert_layer_bytes.assign(40, GiB);
    request.gpu_fixed_bytes = 6 * GiB;
    request.host_fixed_bytes = 3 * GiB;
    return request;
}

template <class F>
static bool throws(F && f) {
    try { f(); } catch (const std::runtime_error &) { return true; }
    return false;
}

int main() {
    // Enough VRAM for every layer keeps all experts on the GPU.
    auto plan = plan_expert_placement(uniform_request(48));
    CHECK(plan.cpu_expert_layers == 0, "expected 0 CPU layers, got %lld", (long long) plan.cpu_expert_layers);
    CHECK(plan.gpu_bytes == 46 * GiB, "expected 46 GiB on the GPU, got %lld", (long long) plan.gpu_bytes);
    CHECK(plan.host_bytes == 3 * GiB, "expected 3 GiB on the host, got %lld", (long long) plan.host_bytes);

    // 16 GiB free: 6 GiB fixed leaves room for 10 layers, so 30 go to the CPU.
    plan = plan_expert_placement(uniform_request(16));
    CHECK(plan.cpu_expert_layers == 30, "expected 30 CPU layers, got %lld", (long long) plan.cpu_expert_layers);
    CHECK(plan.gpu_bytes == 16 * GiB, "expected 16 GiB on the GPU, got %lld", (long long) plan.gpu_bytes);
    CHECK(plan.host_bytes == 33 * GiB, "expected 33 GiB on the host, got %lld", (long long) plan.host_bytes);
    CHECK(plan.budget_bytes == 16 * GiB, "expected a 16 GiB budget, got %lld", (long long) plan.budget_bytes);

    // A part GiB still needs a whole layer moved.
    auto request = uniform_request(16);
    request.free_vram_bytes = 16 * GiB - 1;
    plan = plan_expert_placement(request);
    CHECK(plan.cpu_expert_layers == 31, "expected 31 CPU layers, got %lld", (long long) plan.cpu_expert_layers);

    // A budget below the free VRAM wins; one above it does not.
    request = uniform_request(32);
    request.vram_budget_bytes = 16 * GiB;
    plan = plan_expert_placement(request);
    CHECK(plan.cpu_expert_layers == 30, "budget 16 of 32 free: expected 30, got %lld", (long long) plan.cpu_expert_layers);
    CHECK(plan.budget_bytes == 16 * GiB, "expected the budget to be 16 GiB, got %lld", (long long) plan.budget_bytes);
    request = uniform_request(16);
    request.vram_budget_bytes = 32 * GiB;
    plan = plan_expert_placement(request);
    CHECK(plan.cpu_expert_layers == 30, "budget 32 of 16 free: expected 30, got %lld", (long long) plan.cpu_expert_layers);

    // Leading layers are what moves. When they are small (FP8) and the rest
    // large (F16), more of them must move than a per-layer average suggests.
    request = uniform_request(16);
    request.expert_layer_bytes.assign(40, 2 * GiB);
    for (int layer = 0; layer < 20; ++layer) request.expert_layer_bytes[layer] = GiB;
    // GPU part with N leading layers moved: 6 + sum of layers N..39 must be <= 16.
    // Layers 20..39 alone are 40 GiB, so all 20 small and 15 large ones move.
    plan = plan_expert_placement(request);
    CHECK(plan.cpu_expert_layers == 35, "mixed sizes: expected 35, got %lld", (long long) plan.cpu_expert_layers);
    CHECK(plan.gpu_bytes == 16 * GiB, "mixed sizes: expected 16 GiB on the GPU, got %lld", (long long) plan.gpu_bytes);
    CHECK(plan.host_bytes == 53 * GiB, "mixed sizes: expected 53 GiB on the host, got %lld", (long long) plan.host_bytes);

    // Every expert on the CPU is the last resort; below that it is an error.
    plan = plan_expert_placement(uniform_request(6));
    CHECK(plan.cpu_expert_layers == 40, "expected all 40 layers on the CPU, got %lld", (long long) plan.cpu_expert_layers);
    CHECK(throws([] { plan_expert_placement(uniform_request(5)); }), "a budget below the fixed part must throw");

    // Unknown VRAM on a GPU backend cannot be planned.
    request = uniform_request(16);
    request.vram_known = false;
    CHECK(throws([&] { plan_expert_placement(request); }), "unknown VRAM must throw");

    // A CPU backend keeps every expert in RAM regardless of the numbers.
    request = uniform_request(0);
    request.gpu_backend = false;
    request.vram_known = false;
    plan = plan_expert_placement(request);
    CHECK(plan.cpu_expert_layers == 0, "CPU backend: expected 0, got %lld", (long long) plan.cpu_expert_layers);

    CHECK(!parse_cpu_expert_layers("auto").has_value(), "\"auto\" must parse to nullopt");
    CHECK(parse_cpu_expert_layers("9") == 9, "\"9\" must parse to 9");
    CHECK(throws([] { parse_cpu_expert_layers("nine"); }), "\"nine\" must throw");

    if (failures) {
        std::fprintf(stderr, "%d failure(s)\n", failures);
        return 1;
    }
    std::printf("test_memory_plan: ok\n");
    return 0;
}
