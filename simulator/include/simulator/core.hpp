#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <valarray>

namespace simulator {

template <typename T>
struct OscillatorState {
    std::slice_array<T> positions;
    std::slice_array<T> velocities;
    std::slice_array<T> forces;
};

template <typename T, size_t Rows, size_t Cols = Rows>
struct SimulatorState {
    SimulatorState(T spring_constant) : spring_constant(spring_constant), data(0., Rows * Cols) {}

    OscillatorState<T> operator[](size_t i, size_t j) {
        return {
            .positions = data[std::slice(index(i, j), 2, 1)],
            .velocities = data[std::slice(index(i, j) + 2, 2, 1)],
            .forces = data[std::slice(index(i, j) + 4, 2, 1)],
        };
    }

    const size_t rows = Rows, cols = Cols;
    const size_t n_dim = 2;

    T spring_constant;
    std::valarray<T> data;

    static inline size_t index(size_t i, size_t j) { return i * Cols + j; }
    static inline std::array<size_t, 2> index(size_t i) { return std::array<size_t, 2>{i / Cols, i % Cols}; }

    void copy_to(SimulatorState<T, Rows, Cols> &state) const {
        state.spring_constant = spring_constant;
        std::copy(std::begin(data), std::end(data), std::begin(state.data));
    }
};

} // namespace simulator
