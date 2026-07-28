#include "pufferforge/core.hpp"

#include <cmath>
#include <cstring>
#include <limits>
#include <string>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace pufferforge {

LineWorldVec::LineWorldVec(std::int64_t num_envs, std::int64_t world_size,
                           std::int64_t max_steps, std::uint64_t seed)
    : num_envs_(num_envs),
      world_size_(world_size),
      max_steps_(max_steps),
      seed_(seed),
      rngs_(static_cast<std::size_t>(num_envs)),
      positions_(static_cast<std::size_t>(num_envs)),
      targets_(static_cast<std::size_t>(num_envs)),
      lengths_(static_cast<std::size_t>(num_envs)),
      episode_returns_(static_cast<std::size_t>(num_envs)),
      observations_(static_cast<std::size_t>(num_envs * 4)),
      rewards_(static_cast<std::size_t>(num_envs)),
      terminated_(static_cast<std::size_t>(num_envs)),
      truncated_(static_cast<std::size_t>(num_envs)) {
    if (num_envs <= 0) throw std::invalid_argument("num_envs must be positive");
    if (world_size < 3) throw std::invalid_argument("world_size must be >= 3");
    if (max_steps <= 0) throw std::invalid_argument("max_steps must be positive");
    reset(seed);
}

void LineWorldVec::reset(std::uint64_t seed) {
    seed_ = seed;
    stats_ = {};
    for (std::int64_t i = 0; i < num_envs_; ++i) {
        rngs_[static_cast<std::size_t>(i)].seed(seed_ + 0x9E3779B97F4A7C15ULL * static_cast<std::uint64_t>(i + 1));
        reset_one(i);
        rewards_[static_cast<std::size_t>(i)] = 0.0F;
        terminated_[static_cast<std::size_t>(i)] = 0;
        truncated_[static_cast<std::size_t>(i)] = 0;
    }
}

void LineWorldVec::reset_one(std::int64_t i) {
    auto idx = static_cast<std::size_t>(i);
    std::uniform_int_distribution<std::int32_t> pos_dist(1, static_cast<std::int32_t>(world_size_ - 2));
    std::bernoulli_distribution side(0.5);
    positions_[idx] = pos_dist(rngs_[idx]);
    targets_[idx] = side(rngs_[idx]) ? 0 : static_cast<std::int32_t>(world_size_ - 1);
    lengths_[idx] = 0;
    episode_returns_[idx] = 0.0F;
    write_observation(i);
}

void LineWorldVec::write_observation(std::int64_t i) {
    auto idx = static_cast<std::size_t>(i);
    auto base = static_cast<std::size_t>(i * 4);
    const float denom = static_cast<float>(world_size_ - 1);
    const float pos = static_cast<float>(positions_[idx]) / denom;
    const float target = static_cast<float>(targets_[idx]) / denom;
    observations_[base + 0] = 2.0F * pos - 1.0F;
    observations_[base + 1] = 2.0F * target - 1.0F;
    observations_[base + 2] = target - pos;
    observations_[base + 3] = static_cast<float>(lengths_[idx]) / static_cast<float>(max_steps_);
}

void LineWorldVec::step(const std::int64_t* actions, std::int64_t count) {
    if (count != num_envs_) throw std::invalid_argument("actions length must equal num_envs");

    EpisodeStats local{};
#ifdef _OPENMP
#pragma omp parallel
    {
        EpisodeStats thread_stats{};
#pragma omp for schedule(static)
#endif
        for (std::int64_t i = 0; i < num_envs_; ++i) {
            const auto idx = static_cast<std::size_t>(i);
            const auto action = actions[idx];
            if (action < 0 || action >= 3) continue;

            terminated_[idx] = 0;
            truncated_[idx] = 0;
            rewards_[idx] = -0.01F;

            if (action == 0) positions_[idx] -= 1;
            if (action == 2) positions_[idx] += 1;
            positions_[idx] = std::clamp<std::int32_t>(positions_[idx], 0, static_cast<std::int32_t>(world_size_ - 1));
            lengths_[idx] += 1;

            const bool hit_target = positions_[idx] == targets_[idx];
            const bool timeout = lengths_[idx] >= max_steps_;
            if (hit_target) {
                rewards_[idx] = 1.0F;
                terminated_[idx] = 1;
            } else if (timeout) {
                rewards_[idx] = -0.25F;
                truncated_[idx] = 1;
            }

            episode_returns_[idx] += rewards_[idx];
            if (hit_target || timeout) {
#ifdef _OPENMP
                thread_stats.episodes += 1;
                thread_stats.return_sum += episode_returns_[idx];
                thread_stats.length_sum += static_cast<std::uint64_t>(lengths_[idx]);
#else
                local.episodes += 1;
                local.return_sum += episode_returns_[idx];
                local.length_sum += static_cast<std::uint64_t>(lengths_[idx]);
#endif
                reset_one(i);  // autoreset; done flags/reward remain from completed episode
            } else {
                write_observation(i);
            }
        }
#ifdef _OPENMP
#pragma omp critical
        {
            local.episodes += thread_stats.episodes;
            local.return_sum += thread_stats.return_sum;
            local.length_sum += thread_stats.length_sum;
        }
    }
#endif
    stats_.episodes += local.episodes;
    stats_.return_sum += local.return_sum;
    stats_.length_sum += local.length_sum;
}

EpisodeStats LineWorldVec::drain_stats() {
    EpisodeStats out = stats_;
    stats_ = {};
    return out;
}

}  // namespace pufferforge

namespace {

template <typename T>
py::array view_1d(T* ptr, py::ssize_t n, py::handle owner) {
    return py::array(py::dtype::of<T>(), {n}, {static_cast<py::ssize_t>(sizeof(T))}, ptr, owner);
}

template <typename T>
py::array view_2d(T* ptr, py::ssize_t rows, py::ssize_t cols, py::handle owner) {
    return py::array(py::dtype::of<T>(), {rows, cols},
                     {cols * static_cast<py::ssize_t>(sizeof(T)), static_cast<py::ssize_t>(sizeof(T))},
                     ptr, owner);
}

py::tuple compute_gae(
    py::array_t<float, py::array::c_style | py::array::forcecast> rewards,
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> dones,
    py::array_t<float, py::array::c_style | py::array::forcecast> values,
    py::array_t<float, py::array::c_style | py::array::forcecast> next_values,
    float gamma,
    float gae_lambda) {

    if (rewards.ndim() != 2 || dones.ndim() != 2 || values.ndim() != 2) {
        throw std::invalid_argument("rewards, dones, and values must be rank-2 [T, N]");
    }
    if (rewards.shape(0) != dones.shape(0) || rewards.shape(1) != dones.shape(1) ||
        rewards.shape(0) != values.shape(0) || rewards.shape(1) != values.shape(1)) {
        throw std::invalid_argument("rewards, dones, and values must have identical shapes");
    }
    if (next_values.ndim() != 1 || next_values.shape(0) != rewards.shape(1)) {
        throw std::invalid_argument("next_values must have shape [N]");
    }
    if (!(gamma >= 0.0F && gamma <= 1.0F && gae_lambda >= 0.0F && gae_lambda <= 1.0F)) {
        throw std::invalid_argument("gamma and gae_lambda must be in [0, 1]");
    }

    const py::ssize_t T = rewards.shape(0);
    const py::ssize_t N = rewards.shape(1);
    py::array_t<float> advantages({T, N});
    py::array_t<float> returns({T, N});

    auto r = rewards.unchecked<2>();
    auto d = dones.unchecked<2>();
    auto v = values.unchecked<2>();
    auto nv = next_values.unchecked<1>();
    auto adv = advantages.mutable_unchecked<2>();
    auto ret = returns.mutable_unchecked<2>();

    {
        py::gil_scoped_release release;
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (py::ssize_t n = 0; n < N; ++n) {
            float last_gae = 0.0F;
            for (py::ssize_t t = T; t-- > 0;) {
                const float next_value = (t == T - 1) ? nv(n) : v(t + 1, n);
                const float nonterminal = d(t, n) ? 0.0F : 1.0F;
                const float delta = r(t, n) + gamma * next_value * nonterminal - v(t, n);
                last_gae = delta + gamma * gae_lambda * nonterminal * last_gae;
                adv(t, n) = last_gae;
                ret(t, n) = last_gae + v(t, n);
            }
        }
    }

    return py::make_tuple(std::move(advantages), std::move(returns));
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "PufferForge native rollout and vector-environment core";

    py::class_<pufferforge::LineWorldVec>(m, "LineWorldVec")
        .def(py::init<std::int64_t, std::int64_t, std::int64_t, std::uint64_t>(),
             py::arg("num_envs"), py::arg("world_size") = 15,
             py::arg("max_steps") = 64, py::arg("seed") = 1)
        .def_property_readonly("num_envs", &pufferforge::LineWorldVec::num_envs)
        .def_property_readonly("obs_size", &pufferforge::LineWorldVec::obs_size)
        .def_property_readonly("num_actions", &pufferforge::LineWorldVec::num_actions)
        .def_property_readonly("world_size", &pufferforge::LineWorldVec::world_size)
        .def("reset", [](pufferforge::LineWorldVec& self, std::uint64_t seed) {
            self.reset(seed);
            return view_2d(self.observations(), self.num_envs(), self.obs_size(), py::cast(&self));
        }, py::arg("seed") = 1)
        .def("observations", [](pufferforge::LineWorldVec& self) {
            return view_2d(self.observations(), self.num_envs(), self.obs_size(), py::cast(&self));
        })
        .def("step", [](pufferforge::LineWorldVec& self,
                         py::array_t<std::int64_t, py::array::c_style | py::array::forcecast> actions) {
            if (actions.ndim() != 1) throw std::invalid_argument("actions must be rank-1");
            {
                py::gil_scoped_release release;
                self.step(actions.data(), actions.shape(0));
            }
            py::handle owner = py::cast(&self);
            auto obs = view_2d(self.observations(), self.num_envs(), self.obs_size(), owner);
            auto rewards = view_1d(self.rewards(), self.num_envs(), owner);
            auto terminated = view_1d(self.terminated(), self.num_envs(), owner);
            auto truncated = view_1d(self.truncated(), self.num_envs(), owner);
            return py::make_tuple(obs, rewards, terminated, truncated);
        })
        .def("drain_stats", [](pufferforge::LineWorldVec& self) {
            auto s = self.drain_stats();
            py::dict out;
            out["episodes"] = s.episodes;
            out["return_sum"] = s.return_sum;
            out["length_sum"] = s.length_sum;
            out["mean_return"] = s.episodes ? s.return_sum / static_cast<double>(s.episodes) : 0.0;
            out["mean_length"] = s.episodes ? static_cast<double>(s.length_sum) / static_cast<double>(s.episodes) : 0.0;
            return out;
        });

    m.def("compute_gae", &compute_gae,
          py::arg("rewards"), py::arg("dones"), py::arg("values"),
          py::arg("next_values"), py::arg("gamma") = 0.99F,
          py::arg("gae_lambda") = 0.95F);

#ifdef _OPENMP
    m.attr("openmp_enabled") = true;
    m.attr("openmp_max_threads") = omp_get_max_threads();
#else
    m.attr("openmp_enabled") = false;
    m.attr("openmp_max_threads") = 1;
#endif
}
