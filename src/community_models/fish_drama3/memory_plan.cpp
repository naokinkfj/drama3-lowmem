#include "engine/community_models/fish_drama3/memory_plan.h"

#include <algorithm>
#include <cstdio>
#include <numeric>
#include <stdexcept>

namespace engine::models::fish_drama3 {

namespace {

std::string gib(int64_t bytes) {
    char text[32];
    std::snprintf(text, sizeof(text), "%.1f GiB", static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0));
    return text;
}

}  // namespace

ExpertPlacement plan_expert_placement(const ExpertPlacementRequest & request) {
    const auto & layers = request.expert_layer_bytes;
    const int64_t all_experts = std::accumulate(layers.begin(), layers.end(), int64_t{0});
    ExpertPlacement plan;
    if (!request.gpu_backend) {
        plan.host_bytes = request.host_fixed_bytes + all_experts;
        return plan;
    }
    if (!request.vram_known) {
        throw std::runtime_error("Drama 3 cannot read the free GPU memory; set fish_drama3.cpu_expert_layers to a number");
    }
    plan.budget_bytes = request.vram_budget_bytes ? std::min(*request.vram_budget_bytes, request.free_vram_bytes)
                                                  : request.free_vram_bytes;
    // Move leading layers to the CPU until the rest fits.
    int64_t gpu_experts = all_experts;
    int64_t moved = 0;
    while (request.gpu_fixed_bytes + gpu_experts > plan.budget_bytes && moved < static_cast<int64_t>(layers.size())) {
        gpu_experts -= layers[static_cast<size_t>(moved)];
        ++moved;
    }
    if (request.gpu_fixed_bytes + gpu_experts > plan.budget_bytes) {
        throw std::runtime_error("Drama 3 needs about " + gib(request.gpu_fixed_bytes) +
                                 " of GPU memory even with every expert layer in system RAM, but only " +
                                 gib(plan.budget_bytes) + " is available; lower fish_drama3.context_length or free GPU memory");
    }
    plan.cpu_expert_layers = moved;
    plan.gpu_bytes = request.gpu_fixed_bytes + gpu_experts;
    plan.host_bytes = request.host_fixed_bytes + (all_experts - gpu_experts);
    return plan;
}

std::optional<int64_t> parse_cpu_expert_layers(const std::string & value) {
    if (value == "auto") return std::nullopt;
    size_t used = 0;
    int64_t layers = 0;
    try {
        layers = std::stoll(value, &used);
    } catch (const std::exception &) {
        used = 0;
    }
    if (used == 0 || used != value.size()) {
        throw std::runtime_error("fish_drama3.cpu_expert_layers must be auto or a number of layers, got '" + value + "'");
    }
    return layers;
}

}  // namespace engine::models::fish_drama3
