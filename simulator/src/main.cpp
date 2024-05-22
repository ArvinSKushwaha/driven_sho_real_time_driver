#include "simulator/core.h"
#include <algorithm>
#include <fmt/core.h>
#include <fmt/ranges.h>

#include <random>

int main() {
    std::random_device rd{};
    std::mt19937 rng{rd()};
    std::normal_distribution<double> dst;

    simulator::Simulator<double, 128> simulator(1.0);
    std::for_each(simulator.state->vel.begin(), simulator.state->vel.end(), [&](auto &f) { f = dst(rng); });

    for (int i = 0; i < 100000; i++) {
        fmt::println("{}", i);
        simulator.update(1e-5);
    }

    return 0;
}
