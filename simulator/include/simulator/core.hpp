#pragma once

#include "fmt/core.h"
#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <numeric>

namespace simulator {

template <typename T, size_t Rows, size_t Cols = Rows>
struct SimulatorState {
    static const size_t rows = Rows, cols = Cols;
    static const size_t n_dim = 2;

    std::array<T, Rows * Cols * 2> pos;
    std::array<T, Rows * Cols * 2> vel;
    std::array<T, Rows * Cols * 2> acc;

    static inline size_t index(size_t i, size_t j, size_t k) { return 2 * (i * Cols + j) + k; }

    SimulatorState() {
        pos.fill(0.);
        vel.fill(0.);
        acc.fill(0.);
    }

    void copy_to(SimulatorState &other) const {
        std::copy(pos.begin(), pos.end(), other.pos.begin());
        std::copy(vel.begin(), vel.end(), other.vel.begin());
        std::copy(acc.begin(), acc.end(), other.acc.begin());
    }

    void compute_acc(SimulatorState &other) const {
        copy_to(other);

        for (size_t i = 0; i < Rows; i++) {
            for (size_t j = 0; j < Cols; j++) {
                other.acc[index(i, j, 0)] = -4. * pos[index(i, j, 0)];
                other.acc[index(i, j, 1)] = -4. * pos[index(i, j, 1)];

                if (i > 0) {
                    other.acc[index(i, j, 0)] += pos[index(i - 1, j, 0)];
                    other.acc[index(i, j, 1)] += pos[index(i - 1, j, 1)];
                }

                if (j > 0) {
                    other.acc[index(i, j, 0)] += pos[index(i, j - 1, 0)];
                    other.acc[index(i, j, 1)] += pos[index(i, j - 1, 1)];
                }

                if (i < Cols - 1) {
                    other.acc[index(i, j, 0)] += pos[index(i + 1, j, 0)];
                    other.acc[index(i, j, 1)] += pos[index(i + 1, j, 1)];
                }

                if (j < Cols - 1) {
                    other.acc[index(i, j, 0)] += pos[index(i, j + 1, 0)];
                    other.acc[index(i, j, 1)] += pos[index(i, j + 1, 1)];
                }
            }
        }
    }
};

template <typename T, size_t Rows, size_t Cols = Rows>
struct Simulator {
    T stiffness;

    SimulatorState<T, Rows, Cols> state;
    SimulatorState<T, Rows, Cols> next_state;

    void update(T dt) {
        // move positions
        for (size_t k = 0; k < state.pos.size(); k++) {
            state.pos[k] += state.vel[k] * dt;
        }

        // compute accelerations
        state.compute_acc(next_state);

        // move velocities
        for (size_t k = 0; k < state.pos.size(); k++) {
            next_state.vel[k] += (state.acc[k] + next_state.acc[k]) * dt / 2. * stiffness;
        }

        std::swap(state.pos, next_state.pos);
        std::swap(state.vel, next_state.vel);
        std::swap(state.acc, next_state.acc);
    }
};

} // namespace simulator
