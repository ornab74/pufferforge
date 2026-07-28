#pragma once

#include <algorithm>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <vector>

namespace pufferforge {

struct EpisodeStats {
    std::uint64_t episodes = 0;
    double return_sum = 0.0;
    std::uint64_t length_sum = 0;
};

class LineWorldVec {
public:
    LineWorldVec(std::int64_t num_envs, std::int64_t world_size,
                 std::int64_t max_steps, std::uint64_t seed);

    void reset(std::uint64_t seed);
    void step(const std::int64_t* actions, std::int64_t count);

    [[nodiscard]] std::int64_t num_envs() const noexcept { return num_envs_; }
    [[nodiscard]] std::int64_t obs_size() const noexcept { return 4; }
    [[nodiscard]] std::int64_t num_actions() const noexcept { return 3; }
    [[nodiscard]] std::int64_t world_size() const noexcept { return world_size_; }

    [[nodiscard]] float* observations() noexcept { return observations_.data(); }
    [[nodiscard]] float* rewards() noexcept { return rewards_.data(); }
    [[nodiscard]] std::uint8_t* terminated() noexcept { return terminated_.data(); }
    [[nodiscard]] std::uint8_t* truncated() noexcept { return truncated_.data(); }

    EpisodeStats drain_stats();

private:
    void reset_one(std::int64_t i);
    void write_observation(std::int64_t i);

    std::int64_t num_envs_;
    std::int64_t world_size_;
    std::int64_t max_steps_;
    std::uint64_t seed_;
    std::vector<std::mt19937_64> rngs_;
    std::vector<std::int32_t> positions_;
    std::vector<std::int32_t> targets_;
    std::vector<std::int32_t> lengths_;
    std::vector<float> episode_returns_;
    std::vector<float> observations_;
    std::vector<float> rewards_;
    std::vector<std::uint8_t> terminated_;
    std::vector<std::uint8_t> truncated_;
    EpisodeStats stats_;
};

}  // namespace pufferforge
