#include "tsim/execution_events.hpp"
#include <thread>
#include <vector>
int main() {
  try {
    auto& events = tsim::execution_events();
    std::vector<std::thread> workers;
    for (unsigned worker = 0; worker < 4; ++worker) workers.emplace_back([&, worker] {
      for (unsigned i = 0; i < 30; ++i) {
        auto parent = events.issue(1, "kernel", "worker" + std::to_string(worker), 0,
            {worker, 0, 0}, 0, i, i + 1, 1, 1, "quoted\"\n\\instruction", {2}, {3}, -1, 4, 4, 0, 1, 4);
        tsim::ExecutionEvents::Scope scope(events, parent);
        events.memory(0, 0xfedcba9876540000ull + worker, 4, "global", "load");
      }
    });
    for (auto& worker : workers) worker.join();
    return 0;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "%s\n", e.what());
    return 2;
  }
}
