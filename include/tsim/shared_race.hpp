#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tsim {
// A happens-before detector between warps of one CTA. It does not claim
// per-thread independent-scheduling coverage or cross-CTA/global-memory checks.
class SharedRace {
  static constexpr unsigned max_warps = 32;
  using Clock = std::array<uint64_t, max_warps>;
  struct Access {
    unsigned warp;
    uint64_t time, address;
    unsigned bytes;
    bool strong;
  };
  struct Byte {
    std::vector<Access> writers;
    std::vector<Access> readers;
  };
  struct Release { Clock clock; };
  std::vector<Byte> memory_;
  std::array<Clock, max_warps> clocks_{};
  std::map<std::pair<uint64_t, unsigned>, Release> releases_;
  std::array<std::map<std::pair<uint64_t, unsigned>, Clock>, max_warps> patterns_;
  std::array<std::map<std::pair<uint64_t, unsigned>, Clock>, max_warps> acquire_patterns_;
  std::map<unsigned, Clock> arrivals_;
  unsigned warps_ = 0;
  static bool same_location(const Access& a, const Access& b) {
    return a.address == b.address && a.bytes == b.bytes;
  }
  bool ordered(const Access& previous, const Access& current) const {
    return previous.warp == current.warp ||
           previous.time <= clocks_[current.warp][previous.warp] ||
           (previous.strong && current.strong && same_location(previous, current));
  }
  static void remember(std::vector<Access>& history, const Access& current) {
    // A later strong access must not erase an earlier weak access: moral
    // strength can exempt the later operation without exempting the earlier.
    // Retain the latest event per warp and moral-strength equivalence class.
    auto it = std::find_if(history.begin(), history.end(), [&](const Access& prior) {
      return prior.warp == current.warp && prior.strong == current.strong &&
             (!current.strong || same_location(prior, current));
    });
    if (it == history.end()) history.push_back(current);
    else *it = current;
  }
 public:
  struct Conflict { uint64_t address; unsigned other; bool prior_write, current_write; };
  void reset(unsigned warps, size_t bytes) {
    if (!warps || warps > max_warps) throw std::invalid_argument("shared race: unsupported warp count");
    warps_ = warps;
    memory_.clear(); memory_.resize(bytes);
    clocks_ = {};
    releases_.clear();
    for (auto& patterns : patterns_) patterns.clear();
    for (auto& patterns : acquire_patterns_) patterns.clear();
    arrivals_.clear();
    for (unsigned warp = 0; warp < warps; ++warp) clocks_[warp][warp] = 1;
  }
  void barrier(uint64_t participants) {
    for (unsigned warp = 0; warp < warps_; ++warp) if ((participants >> warp) & 1) arrive(16, warp);
    complete(16, participants);
  }
  void arrive(unsigned id, unsigned warp) {
    if (warp >= warps_) throw std::out_of_range("shared race: barrier warp");
    ++clocks_[warp][warp];
    auto& joined = arrivals_[id];
    for (unsigned other = 0; other < warps_; ++other)
      joined[other] = std::max(joined[other], clocks_[warp][other]);
  }
  void complete(unsigned id, uint64_t acquirers) {
    const Clock joined = arrivals_.at(id);
    for (unsigned warp = 0; warp < warps_; ++warp) if ((acquirers >> warp) & 1) {
      for (unsigned other = 0; other < warps_; ++other)
        clocks_[warp][other] = std::max(clocks_[warp][other], joined[other]);
      ++clocks_[warp][warp];
    }
    arrivals_.erase(id);
  }
  std::optional<Conflict> access(unsigned warp, uint64_t address, unsigned bytes, bool write,
                                 bool strong = false, bool release = false, bool acquire = false,
                                 bool silent_write = false, bool rmw = false) {
    if (warp >= warps_ || address > memory_.size() || bytes > memory_.size() - address)
      throw std::out_of_range("shared race: access outside shadow memory");
    const auto location = std::make_pair(address, bytes);
    if (!write && strong) {
      auto observed = releases_.find(location);
      if (observed != releases_.end()) {
        auto& pattern = acquire_patterns_[warp][location];
        for (unsigned other = 0; other < warps_; ++other)
          pattern[other] = std::max(pattern[other], observed->second.clock[other]);
      }
    }
    if (acquire) {
      auto it = acquire_patterns_[warp].find(location);
      if (it != acquire_patterns_[warp].end())
        for (unsigned other = 0; other < warps_; ++other)
          clocks_[warp][other] = std::max(clocks_[warp][other], it->second[other]);
    }
    const Access current{warp, ++clocks_[warp][warp], address, bytes, strong};
    std::optional<Conflict> conflict;
    for (uint64_t byte = address; byte < address + bytes; ++byte) {
      Byte& cell = memory_[byte];
      for (const auto& writer : cell.writers)
        if (!ordered(writer, current) && !(write && silent_write) && !conflict)
          conflict = Conflict{byte, writer.warp, true, write};
      if (write) {
        for (const auto& reader : cell.readers)
          if (!ordered(reader, current) && !silent_write && !conflict)
            conflict = Conflict{byte, reader.warp, false, true};
        // Preserve the inherited non-strict equal-value-store policy: its
        // replacement writer must not resurrect a suppressed writer conflict
        // on a following read. Strict checking never enables this branch.
        if (silent_write) cell.writers.clear();
        remember(cell.writers, current);
      } else {
        remember(cell.readers, current);
      }
    }
    if (write) {
      // Program-order release patterns and RMW observation chains are
      // different relations. A plain strong write can extend its own prior
      // release; an atomic RMW can additionally carry the publication it read.
      // Carrying it does not give a relaxed RMW acquire semantics.
      std::optional<Clock> candidate;
      auto merge_candidate = [&](const Clock& clock) {
        if (!candidate) candidate = Clock{};
        for (unsigned other = 0; other < warps_; ++other)
          (*candidate)[other] = std::max((*candidate)[other], clock[other]);
      };
      if (strong) {
        auto pattern = patterns_[warp].find(location);
        if (pattern != patterns_[warp].end()) merge_candidate(pattern->second);
        auto observed = releases_.find(location);
        if (rmw && observed != releases_.end()) merge_candidate(observed->second.clock);
      }
      for (auto it = releases_.begin(); it != releases_.end();) {
        uint64_t start = it->first.first, end = start + it->first.second;
        if (address < end && start < address + bytes) it = releases_.erase(it);
        else ++it;
      }
      if (release) {
        patterns_[warp][location] = clocks_[warp];
        merge_candidate(clocks_[warp]);
      }
      if (candidate) releases_[location] = Release{*candidate};
    }
    return conflict;
  }
};
}
