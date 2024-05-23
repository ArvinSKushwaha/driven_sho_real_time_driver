#pragma once

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace simulator {

constexpr auto morton() {
    std::array<uint16_t, 256> arr{};
    arr[0] = 0;
    arr[1] = 1;

    for (size_t k = 1; k < 8; k <<= 1) {
        for (size_t i = (1ul << k); i <= (1ul << (k << 1)) - 1ul; i++) {
            arr[i] = (arr[i / (1 << k)] << (2 * k)) | arr[i % (1 << k)];
        }
    }

    return arr;
}

constexpr const std::array<uint16_t, 256> MORTON_LUT = morton();

constexpr uint64_t morton_lookup(uint32_t i) {
    const uint32_t MASK_BYTE_0 = 0x000000FF;
    const uint32_t MASK_BYTE_1 = 0x0000FF00;
    const uint32_t MASK_BYTE_2 = 0x00FF0000;
    const uint32_t MASK_BYTE_3 = 0xFF000000;
    return uint64_t(MORTON_LUT[i & MASK_BYTE_0]) | uint64_t(MORTON_LUT[(i & MASK_BYTE_1) >> 8] << 16) |
           uint64_t(MORTON_LUT[(i & MASK_BYTE_2) >> 16]) << 32 | uint64_t(MORTON_LUT[(i & MASK_BYTE_3) >> 24]) << 48;
}

template <size_t N>
constexpr std::array<size_t, N> arbitrary_morton_table() {
    std::array<size_t, N> arr = {};

    for (size_t i = 0; i < N; i++) {
        arr[i] = morton_lookup(i);
    }

    return arr;
}

} // namespace simulator
