#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <omp.h>

#include "fmt/core.h"

#include "simulator/morton.h"

namespace simulator {

template <typename T, size_t Rows, size_t Cols = Rows>
struct Simulator;

template <typename T, size_t Rows, size_t Cols = Rows>
struct SimulatorState {
    constexpr static const auto SIM_MORTON_LUT = arbitrary_morton_table<std::max(Rows, Cols)>();
    static const size_t rows = Rows, cols = Cols;
    static const size_t n_dim = 2;

    std::array<T, Rows * Cols * 2> pos;
    std::array<T, Rows * Cols * 2> vel;
    std::array<T, Rows * Cols * 2> acc;

    // Standard Indexing
    constexpr static inline size_t index(size_t i, size_t j, bool k) { return ((i * cols + j) << 1) + k; }
    constexpr static inline std::array<size_t, 2> deindex(size_t k) { return {k / cols, k % cols}; }

    // Morton Indexing
    // constexpr static inline size_t index(size_t i, size_t j, bool k) {
    //     return (SIM_MORTON_LUT[i] << 2) + (SIM_MORTON_LUT[j] << 1) + k;
    // }

    SimulatorState() {
        pos.fill(0.);
        vel.fill(0.);
        acc.fill(0.);
        tmp_acc.fill(0.);
    }

    void compute_acc() {
#pragma omp parallel for simd collapse(2) schedule(static, 1024)
        for (size_t i = 0; i < Rows; i++) {
            for (size_t j = 0; j < Rows; j++) {
                if (i > 0) {
                    tmp_acc[index(i, j, 0)] -= pos[index(i, j, 0)];
                    tmp_acc[index(i, j, 1)] -= pos[index(i, j, 1)];
                    tmp_acc[index(i, j, 0)] += pos[index(i - 1, j, 0)];
                    tmp_acc[index(i, j, 1)] += pos[index(i - 1, j, 1)];
                }

                if (j > 0) {
                    tmp_acc[index(i, j, 0)] -= pos[index(i, j, 0)];
                    tmp_acc[index(i, j, 1)] -= pos[index(i, j, 1)];
                    tmp_acc[index(i, j, 0)] += pos[index(i, j - 1, 0)];
                    tmp_acc[index(i, j, 1)] += pos[index(i, j - 1, 1)];
                }

                if (i < Cols - 1) {
                    tmp_acc[index(i, j, 0)] -= pos[index(i, j, 0)];
                    tmp_acc[index(i, j, 1)] -= pos[index(i, j, 1)];
                    tmp_acc[index(i, j, 0)] += pos[index(i + 1, j, 0)];
                    tmp_acc[index(i, j, 1)] += pos[index(i + 1, j, 1)];
                }

                if (j < Cols - 1) {
                    tmp_acc[index(i, j, 0)] -= pos[index(i, j, 0)];
                    tmp_acc[index(i, j, 1)] -= pos[index(i, j, 1)];
                    tmp_acc[index(i, j, 0)] += pos[index(i, j + 1, 0)];
                    tmp_acc[index(i, j, 1)] += pos[index(i, j + 1, 1)];
                }
            }
        }

        std::swap(acc, tmp_acc);
    }

    friend void Simulator<T, Rows, Cols>::update(T dt);

  private:
    std::array<T, Rows * Cols * 2> tmp_acc;
};

template <typename T, size_t Rows, size_t Cols>
struct Simulator {
    T stiffness;

    Simulator(T stiffness) : stiffness(stiffness), state(std::make_unique<SimulatorState<T, Rows, Cols>>()) {}

    std::unique_ptr<SimulatorState<T, Rows, Cols>> state;

    void update(T dt) {
        auto pos = state->pos.data();
        auto vel = state->vel.data();
        auto acc = state->acc.data();
        auto nextacc = state->tmp_acc.data();

        // move positions
#pragma omp parallel for simd schedule(static, 1024)
        for (size_t k = 0; k < 2 * Rows * Cols; k++) {
            pos[k] += vel[k] * dt;
        }

        // compute accelerations
        state->compute_acc();

        // move velocities
#pragma omp parallel for simd schedule(static, 1024)
        for (size_t k = 0; k < 2 * Rows * Cols; k++) {
            vel[k] += (acc[k] + nextacc[k]) * dt / 2. * stiffness;
        }
    }
};

} // namespace simulator
