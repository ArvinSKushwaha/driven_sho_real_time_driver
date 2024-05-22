#pragma once

#include "fmt/core.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <execution>
#include <ranges>

namespace simulator {

template <typename T, size_t Rows, size_t Cols = Rows>
struct SimulatorState {
    static constexpr const auto iterator = std::views::iota(0ul, Rows *Cols);
    static const size_t rows = Rows, cols = Cols;
    static const size_t n_dim = 2;

    std::array<T, Rows * Cols * 2> pos;
    std::array<T, Rows * Cols * 2> vel;
    std::array<T, Rows * Cols * 2> acc;

    static inline size_t index(size_t i, size_t j, size_t k) { return 2 * (i * Cols + j) + k; }
    static inline std::array<size_t, 2> deindex(size_t k) { return {k % Cols, k / Cols}; }

    SimulatorState() {
        pos.fill(0.);
        vel.fill(0.);
        acc.fill(0.);
    }

    void copy_to(SimulatorState *other) const {
        std::copy(pos.begin(), pos.end(), other->pos.begin());
        std::copy(vel.begin(), vel.end(), other->vel.begin());
        std::copy(acc.begin(), acc.end(), other->acc.begin());
    }

    void compute_acc(SimulatorState *other) const {
        copy_to(other);

        std::for_each(std::execution::par_unseq, iterator.begin(), iterator.end(), [&](size_t k) {
            auto [i, j] = deindex(k);
            other->acc[index(i, j, 0)] = -4. * pos[index(i, j, 0)];
            other->acc[index(i, j, 1)] = -4. * pos[index(i, j, 1)];

            if (i > 0) {
                other->acc[index(i, j, 0)] += pos[index(i - 1, j, 0)];
                other->acc[index(i, j, 1)] += pos[index(i - 1, j, 1)];
            }

            if (j > 0) {
                other->acc[index(i, j, 0)] += pos[index(i, j - 1, 0)];
                other->acc[index(i, j, 1)] += pos[index(i, j - 1, 1)];
            }

            if (i < Cols - 1) {
                other->acc[index(i, j, 0)] += pos[index(i + 1, j, 0)];
                other->acc[index(i, j, 1)] += pos[index(i + 1, j, 1)];
            }

            if (j < Cols - 1) {
                other->acc[index(i, j, 0)] += pos[index(i, j + 1, 0)];
                other->acc[index(i, j, 1)] += pos[index(i, j + 1, 1)];
            }
        });
    }
};

template <typename T, size_t Rows, size_t Cols = Rows>
struct Simulator {
    static constexpr const auto iterator = std::views::iota(0ul, Rows *Cols);

    T stiffness;

    Simulator(T stiffness)
        : stiffness(stiffness), state(std::make_unique<SimulatorState<T, Rows, Cols>>()), next_state(nullptr) {}

    std::unique_ptr<SimulatorState<T, Rows, Cols>> state;
    std::unique_ptr<SimulatorState<T, Rows, Cols>> next_state;

    void update(T dt) {
        T stiffness = stiffness;
        const auto time_update = [&](T x, T v) { return x + v * dt; };

        if (next_state == nullptr) {
            next_state = std::make_unique<SimulatorState<T, Rows, Cols>>();
            state->copy_to(next_state.get());
        }

        // move positions
        std::transform(std::execution::par_unseq, next_state->pos.cbegin(), next_state->pos.cend(), state->vel.cbegin(),
                       state->pos.begin(), time_update);

        // compute accelerations
        state->compute_acc(next_state.get());

        // move velocities
        std::for_each(std::execution::par_unseq, iterator.begin(), iterator.end(), [&](size_t k) {
            next_state->vel[k] += (state->acc[k] + next_state->acc[k]) * dt / 2. * stiffness;
        });

        std::swap(state->pos, next_state->pos);
        std::swap(state->vel, next_state->vel);
        std::swap(state->acc, next_state->acc);
    }
};

} // namespace simulator
