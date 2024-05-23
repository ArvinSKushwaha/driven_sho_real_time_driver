#include <algorithm>
#include <chrono>
#include <random>

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <tqdm.hpp>

#include "simulator/core.h"

int main() {
    const size_t N = 1024;
    const size_t T = 100000;

    std::random_device rd{};
    std::mt19937 rng{rd()};
    std::normal_distribution<double> dst;

    simulator::Simulator<double, N> simulator(1.0);
    std::for_each(simulator.state->vel.begin(), simulator.state->vel.end(), [&](auto &f) { f = dst(rng); });

    auto A = tq::trange(T);
    const auto start{std::chrono::high_resolution_clock::now()};
    for (size_t i = 0; i < T; i++) {
        simulator.update(1e-5);

        const auto now{std::chrono::high_resolution_clock::now()};
        A << fmt::format("{:5f} μs / step", (std::chrono::duration<double, std::micro>(now - start) / (i + 1)).count());
        A.update();
    }

    return 0;
}
