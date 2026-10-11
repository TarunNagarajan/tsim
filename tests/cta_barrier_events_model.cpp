#include "tsim/execution_events.hpp"
#include <string>
#include <thread>
#include <vector>

int main(int argc, char**) {
  try {
    auto& events = tsim::execution_events();
    tsim::ExecutionEvents::LaunchScope launch(events, 1, "k", {4,1,1}, {64,1,1}, 4,4,0,0);
    std::vector<std::thread> workers;
    for (unsigned worker = 0; worker < 4; ++worker) workers.emplace_back([&, worker] {
      const std::array<unsigned,3> block{worker,0,0};
      auto instruction = [&](unsigned warp, unsigned pc, const std::string& text, bool red = false) {
        return events.issue(1,"k","k",0,block,warp,pc,pc+1,0xffffffff,0xffffffff,text,
                            red ? std::vector<uint32_t>{0} : std::vector<uint32_t>{},
                            red ? std::vector<uint32_t>{4} : std::vector<uint32_t>{},-1,4,0,0);
      };
      auto retire = [&](unsigned warp) {
        auto issue = instruction(warp,99,"ret;");
        tsim::ExecutionEvents::Scope scope(events,issue);
        return events.cta_exit(1,"k",block,warp,0xffffffff);
      };
      if (argc > 1) {
        // An implicit exit must never inherit the current outer instruction,
        // which could belong to a different warp, block or nested launch.
        {
          auto issue = instruction(0,3,"mov.u32 %r0, 1;");
          tsim::ExecutionEvents::Scope scope(events,issue);
          events.cta_exit(1,"k",block,1,0xffffffff,true);
        }
        retire(0);
        return;
      }
      const std::string red = "bar.red.popc.u32 %r0, 3, %p2;";
      unsigned long long a, b;
      {
        auto issue = instruction(0,10,red,true);
        tsim::ExecutionEvents::Scope scope(events,issue);
        a = events.cta_barrier_arrive(3,0,0,3,true,0xffffffff,17);
      }
      {
        auto issue = instruction(1,10,red,true);
        tsim::ExecutionEvents::Scope scope(events,issue);
        b = events.cta_barrier_arrive(3,0,0,3,true,0xffffffff,5);
        events.cta_barrier_release(1,"k",block,3,0,0,3,3,3,{a,b},{},22);
      }
      for (unsigned warp = 0; warp < 2; ++warp) {
        auto issue = instruction(warp,10,red,true);
        tsim::ExecutionEvents::Scope scope(events,issue);
        events.cta_barrier_result(warp ? b : a,0xffffffff,22);
      }
      {
        auto issue = instruction(0,11,"bar.sync 3;");
        tsim::ExecutionEvents::Scope scope(events,issue);
        a = events.cta_barrier_arrive(3,1,0,0,true,0xffffffff,0);
      }
      auto exited = retire(1);
      events.cta_barrier_release(1,"k",block,3,1,0,0,1,1,{a},{exited},0);
      retire(0);
    });
    for (auto& worker : workers) worker.join();
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr,"%s\n",error.what());
    return 2;
  }
}
