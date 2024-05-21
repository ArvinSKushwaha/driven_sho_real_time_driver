#include "simulator/core.hpp"
#include <fmt/core.h>
#include <fmt/ranges.h>

int main() {
    simulator::SimulatorState<double, 5, 5> simulator(1.0);
    simulator.data = 1.;

    simulator::SimulatorState<double, 5, 5> simulator2(0.);

    fmt::println("{}", simulator.spring_constant);
    fmt::println("{}", simulator.data);
    fmt::println("{}", simulator2.spring_constant);
    fmt::println("{}", simulator2.data);

    simulator.copy_to(simulator2);

    fmt::println("{}", simulator.spring_constant);
    fmt::println("{}", simulator.data);
    fmt::println("{}", simulator2.spring_constant);
    fmt::println("{}", simulator2.data);

    return 0;
}
