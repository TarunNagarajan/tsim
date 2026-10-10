#include "tsim/shared_race.hpp"
#include <cstdio>
#include <stdexcept>

void require(bool value, const char* name) { if (!value) throw std::runtime_error(name); }
int main() {
  using tsim::SharedRace;
  SharedRace model;
  model.reset(3, 16);
  require(!model.access(0, 0, 4, true), "initial write");
  require(model.access(1, 0, 4, false).has_value(), "unordered payload must race");
  model.reset(3, 16);
  model.access(0, 0, 4, true);
  require(!model.access(0, 4, 4, true, true, true), "release store");
  require(!model.access(1, 4, 4, false, true, false, true), "acquire load");
  require(!model.access(1, 0, 4, false), "acquired payload must be ordered");
  model.access(1, 8, 4, true, true, true);
  model.access(2, 8, 4, false, true, false, true);
  require(!model.access(2, 0, 4, false), "transitive synchronization");
  model.reset(3, 16);
  model.access(0, 0, 4, true);
  model.access(0, 4, 4, true, true);
  require(!model.access(1, 4, 4, false, true), "strong flag accesses alone must not race");
  require(model.access(1, 0, 4, false).has_value(), "relaxed flag must not order payload");
  model.reset(3, 16);
  model.access(0, 0, 4, true);
  model.barrier(0b011);
  require(!model.access(1, 0, 4, false), "barrier participant sees payload");
  require(model.access(2, 0, 4, false).has_value(), "nonparticipant remains unordered");
  model.reset(3, 16);
  model.access(0, 0, 4, true);
  model.arrive(0, 0);
  model.access(0, 4, 4, true); // after arrival, outside this barrier's release
  model.arrive(0, 1);
  model.complete(0, 0b010);
  require(!model.access(1, 0, 4, false), "waiter acquires pre-arrival payload");
  require(model.access(1, 4, 4, false).has_value(), "post-arrival writes remain unordered");
  model.reset(3, 16);
  model.arrive(0, 0);
  model.access(1, 0, 4, true);
  model.arrive(0, 1);
  model.complete(0, 0b010);
  require(model.access(0, 0, 4, false).has_value(), "arrive alone does not acquire completion");
  model.reset(3, 16);
  model.access(0, 0, 1, true);
  require(!model.access(1, 1, 1, true), "adjacent bytes do not overlap");
  require(model.access(2, 0, 1, false).has_value(), "earlier byte writer must not be lost");
  model.reset(3, 16);
  model.access(0, 4, 4, true, true, true);
  require(model.access(1, 4, 2, false, true).has_value(), "mixed-width strong operations are not morally strong");
  model.reset(3, 16);
  model.access(0, 0, 4, true);
  model.access(0, 4, 4, true, true, true);
  model.barrier(0b101);
  model.access(2, 4, 4, true, true); // unrelated relaxed writer invalidates the release.
  model.access(1, 4, 4, false, true, false, true);
  require(model.access(1, 0, 4, false).has_value(), "acquire must not inherit an overwritten release");
  model.reset(3, 16);
  model.access(0, 0, 4, true); // weak writer then strong writer, no publication
  model.access(0, 0, 4, true, true);
  require(model.access(1, 0, 4, false, true).has_value(), "strong writer must not hide prior weak writer");
  model.reset(3, 16);
  model.access(0, 0, 4, false); // weak reader then strong reader
  model.access(0, 0, 4, false, true);
  require(model.access(1, 0, 4, true, true).has_value(), "strong reader must not hide prior weak reader");
  model.reset(3, 16);
  model.access(0, 0, 4, true);
  require(!model.access(1, 0, 4, true, false, false, false, true), "optional equal-value suppression");
  require(!model.access(1, 0, 4, false), "suppressed writer must not reappear on own read");
  model.reset(4, 16);
  model.access(0, 0, 4, true);
  model.access(0, 4, 4, true, true, true);
  model.access(1, 4, 4, true, true, false, false, false, true);
  model.access(2, 4, 4, true, true, false, false, false, true);
  model.access(3, 4, 4, false, true, false, true);
  require(!model.access(3, 0, 4, false), "acquire inherits publication through relaxed RMW chain");
  require(model.access(1, 0, 4, false).has_value(), "relaxed RMW must not itself acquire payload");
  model.reset(4, 16);
  model.access(0, 0, 4, true);
  model.access(0, 4, 4, true, true, true);
  model.access(1, 8, 4, true);
  model.access(1, 4, 4, true, true, true, false, false, true);
  model.access(3, 4, 4, false, true, false, true);
  require(!model.access(3, 0, 4, false), "release RMW preserves origin publication");
  require(!model.access(3, 8, 4, false), "release RMW adds its own prior payload");
  model.reset(4, 16);
  model.access(0, 0, 4, true);
  model.access(0, 4, 4, true, true, true);
  model.access(1, 8, 4, true);
  model.access(1, 4, 4, true, true, true, false, false, true);
  model.access(1, 4, 4, true, true); // plain write extends only its own release
  model.access(3, 4, 4, false, true, false, true);
  require(!model.access(3, 8, 4, false), "plain write extends own release pattern");
  require(model.access(3, 0, 4, false).has_value(), "plain write must not carry someone else's RMW origin");
  model.reset(3, 16);
  model.access(0, 0, 4, true);
  model.access(0, 4, 4, true, true, true);
  model.access(1, 4, 4, true, true); // intervening writer is not an RMW
  model.access(0, 4, 4, true, true); // original producer extends its release
  model.access(2, 4, 4, false, true, false, true);
  require(!model.access(2, 0, 4, false), "own release pattern survives an intervening writer");
  model.reset(3, 16);
  model.access(0, 0, 4, true);
  model.access(0, 4, 4, true, true, true);
  model.access(2, 4, 4, false, true); // strong read starts an acquire pattern
  model.access(1, 4, 4, true, true); // overwrite before the acquire completes
  model.access(2, 4, 4, false, true, false, true);
  require(!model.access(2, 0, 4, false), "earlier strong read participates in later acquire pattern");
  model.reset(3, 16);
  model.access(0, 0, 4, true);
  model.access(0, 4, 4, true, true, true);
  model.access(0, 8, 4, true); // work after the release needs another release
  model.access(0, 4, 4, true, true);
  model.access(1, 4, 4, false, true, false, true);
  require(!model.access(1, 0, 4, false), "extended release publishes preceding work");
  require(model.access(1, 8, 4, false).has_value(), "extended release does not publish post-release work");
  std::puts("PASS shared race model: ordering, transitivity, barrier isolation, overlap and overwritten releases");
}
