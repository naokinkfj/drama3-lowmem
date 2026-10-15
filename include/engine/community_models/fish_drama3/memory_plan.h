#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace engine::models::fish_drama3 {

// Inputs for choosing how many leading expert layers stay in system RAM when
// fish_drama3.cpu_expert_layers is "auto". All sizes are bytes.
struct ExpertPlacementRequest {
    bool gpu_backend = true;
    bool vram_known = false;
    int64_t free_vram_bytes = 0;
    std::optional<int64_t> vram_budget_bytes;
    // Expert gate/up/down bytes of each layer, in layer order. Layers can
    // differ (a package may keep only its first layers in FP8).
    std::vector<int64_t> expert_layer_bytes;
    // Peak GPU use without any expert bank: other weights, caches, graphs and
    // the safety margin.
    int64_t gpu_fixed_bytes = 0;
    // Host memory needed besides the CPU-resident expert banks.
    int64_t host_fixed_bytes = 0;
};

struct ExpertPlacement {
    int64_t cpu_expert_layers = 0;
    int64_t budget_bytes = 0;
    int64_t gpu_bytes = 0;
    int64_t host_bytes = 0;
};

// Keeps the fewest leading expert layers on the CPU such that the estimated
// GPU peak fits min(free VRAM, vram budget). Throws when the GPU memory cannot
// be read or when even all-CPU experts do not fit. A CPU backend keeps every
// layer on the CPU anyway, so it returns 0.
ExpertPlacement plan_expert_placement(const ExpertPlacementRequest & request);

// "auto" yields nullopt; an integer yields its value; anything else throws.
std::optional<int64_t> parse_cpu_expert_layers(const std::string & value);

}  // namespace engine::models::fish_drama3
