#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <omp.h>

#include "fmt/core.h"

#include "simulator/morton.h"

namespace simulator {

template <typename T, size_t Rows, size_t Cols = Rows>
struct SimulatorState {
    constexpr static const auto SIM_MORTON_LUT = arbitrary_morton_table<std::max(Rows, Cols)>();
    static const size_t rows = Rows, cols = Cols;
    static const size_t n_dim = 2;

    std::array<T, Rows * Cols * 2> pos;
    std::array<T, Rows * Cols * 2> vel;
    std::array<T, Rows * Cols * 2> acc;

    // Standard Indexing
    // static inline size_t index(size_t i, size_t j, bool k) { return ((i * cols + j) << 1) + k; }
    // static inline std::array<size_t, 2> deindex(size_t k) { return {k / cols, k % cols}; }

    // Morton Indexing
    constexpr static inline size_t index(size_t i, size_t j, bool k) {
        return (SIM_MORTON_LUT[i] << 2) + (SIM_MORTON_LUT[j] << 1) + k;
    }

    SimulatorState() {
        pos.fill(0.);
        vel.fill(0.);
        acc.fill(0.);
    }

    void copy_to(SimulatorState *other) const {
        auto otherpos = other->pos.data();
        auto othervel = other->vel.data();
        auto otheracc = other->acc.data();

#pragma omp parallel for simd schedule(static, 1024)
        for (size_t k = 0; k < 2 * Rows * Cols; k++) {
            otherpos[k] = pos[k];
            othervel[k] = vel[k];
            otheracc[k] = acc[k];
        }
    }

    void compute_acc(SimulatorState *other) const {
        copy_to(other);
        auto otheracc = other->acc.data();

#pragma omp parallel for simd collapse(2) schedule(static, 1024)
        for (size_t i = 0; i < Rows; i++) {
            for (size_t j = 0; j < Rows; j++) {
                otheracc[index(i, j, 0)] = -4. * pos[index(i, j, 0)];
                otheracc[index(i, j, 1)] = -4. * pos[index(i, j, 1)];

                if (i > 0) {
                    otheracc[index(i, j, 0)] += pos[index(i - 1, j, 0)];
                    otheracc[index(i, j, 1)] += pos[index(i - 1, j, 1)];
                }

                if (j > 0) {
                    otheracc[index(i, j, 0)] += pos[index(i, j - 1, 0)];
                    otheracc[index(i, j, 1)] += pos[index(i, j - 1, 1)];
                }

                if (i < Cols - 1) {
                    otheracc[index(i, j, 0)] += pos[index(i + 1, j, 0)];
                    otheracc[index(i, j, 1)] += pos[index(i + 1, j, 1)];
                }

                if (j < Cols - 1) {
                    otheracc[index(i, j, 0)] += pos[index(i, j + 1, 0)];
                    otheracc[index(i, j, 1)] += pos[index(i, j + 1, 1)];
                }
            }
        }
    }
};

template <typename T, size_t Rows, size_t Cols = Rows>
struct Simulator {
    T stiffness;

    Simulator(T stiffness)
        : stiffness(stiffness), state(std::make_unique<SimulatorState<T, Rows, Cols>>()), next_state(nullptr) {}

    std::unique_ptr<SimulatorState<T, Rows, Cols>> state;
    std::unique_ptr<SimulatorState<T, Rows, Cols>> next_state;

    void update(T dt) {
        if (next_state == nullptr) {
            next_state = std::make_unique<SimulatorState<T, Rows, Cols>>();
            state->copy_to(next_state.get());
        }

        auto statepos = state->pos.data();
        auto statevel = state->vel.data();
        auto stateacc = state->acc.data();

        auto prevpos = next_state->pos.data();

        // move positions
#pragma omp parallel for simd schedule(static, 1024)
        for (size_t k = 0; k < 2 * Rows * Cols; k++) {
            statepos[k] = prevpos[k] + statevel[k] * dt;
        }

        // compute accelerations
        state->compute_acc(next_state.get());

        auto nextvel = next_state->vel.data();
        auto nextacc = next_state->acc.data();

        // move velocities
#pragma omp parallel for simd schedule(static, 1024)
        for (size_t k = 0; k < 2 * Rows * Cols; k++) {
            nextvel[k] = statevel[k] + (stateacc[k] + nextacc[k]) * dt / 2. * stiffness;
        }

        std::swap(state->pos, next_state->pos);
        std::swap(state->vel, next_state->vel);
        std::swap(state->acc, next_state->acc);
    }
};

} // namespace simulator
