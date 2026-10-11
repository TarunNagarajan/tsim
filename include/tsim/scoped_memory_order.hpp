#pragma once
#include <algorithm>
#include <compare>
#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace tsim {
// Observed generic-proxy causality and byte conflicts. The caller supplies
// canonical namespaces and actual read-from identities; this is not a value
// simulator, a proxy/alias model or an enumeration of weak-memory outcomes.
class ScopedMemoryOrder {
 public:
  using Id = uint64_t;
  enum class Scope { Cta, Cluster, Gpu, System };
  enum class Sem { Weak, Relaxed, Acquire, Release, AcqRel };
  enum class Kind { Read, Write, Rmw };
  struct Actor {
    Id device, grid, cta, cluster, thread;
    bool host = false;
    auto operator<=>(const Actor&) const = default;
  };
  struct Location {
    Id name_space, address;
    auto operator<=>(const Location&) const = default;
  };
  struct Conflict { Id byte, previous_event, previous_actor; bool prior_write, current_write; };
  struct Result { Id event; std::optional<Conflict> conflict; };
 private:
  using Clock = std::map<Id, Id>;
  using MemoryKey = std::pair<Location, unsigned>;
  struct Endpoint { Id actor; Scope scope; Clock clock; std::optional<MemoryKey> memory; };
  struct Access {
    Id event, actor, time;
    Location location;
    unsigned bytes;
    bool strong, write;
    Scope scope;
  };
  struct Version { Access access; std::vector<Endpoint> publications; };
  struct Byte { std::vector<Access> readers, writers; Id latest = 0; };
  struct Thread {
    Actor owner;
    Clock clock;
    std::map<MemoryKey, std::vector<Endpoint>> releases, observations;
    std::vector<Endpoint> fences;
  };
  struct Barrier { Id device, grid, cta; std::map<Id, Clock> arrivals; };
  std::map<Id, Thread> threads_;
  std::map<Actor, Id> identities_;
  std::map<Location, Byte> memory_;
  std::map<Id, Version> versions_;
  std::map<Id, std::pair<Id, Id>> events_;
  std::vector<Endpoint> sc_fences_;
  std::map<Id, Barrier> barriers_;
  Id serial_ = 0;

  static void valid(Scope scope) {
    if (scope < Scope::Cta || scope > Scope::System) throw std::invalid_argument("unknown memory scope");
  }
  Thread& thread(Id actor) { return threads_.at(actor); }
  const Thread& thread(Id actor) const { return threads_.at(actor); }
  bool includes(Id actor, Scope scope, Id other) const {
    const Actor& a = thread(actor).owner; const Actor& b = thread(other).owner;
    if (scope == Scope::System) return true;
    if (a.host || b.host || a.device != b.device) return false;
    if (scope == Scope::Gpu) return true;
    if (a.grid != b.grid) return false;
    return scope == Scope::Cluster ? a.cluster == b.cluster : a.cta == b.cta;
  }
  bool mutual(Id a, Scope x, Id b, Scope y) const {
    return a == b || (includes(a, x, b) && includes(b, y, a));
  }
  static void join(Clock& into, const Clock& from) {
    for (auto [actor, time] : from) into[actor] = std::max(into[actor], time);
  }
  static Id value(const Clock& clock, Id actor) {
    auto it = clock.find(actor); return it == clock.end() ? 0 : it->second;
  }
  Endpoint endpoint(Id actor, Scope scope, std::optional<MemoryKey> memory = {}) const {
    return {actor, scope, thread(actor).clock, memory};
  }
  static void retain(std::vector<Endpoint>& into, const Endpoint& origin) {
    auto it = std::find_if(into.begin(), into.end(), [&](const Endpoint& p) {
      return p.actor == origin.actor && p.scope == origin.scope && p.memory == origin.memory;
    });
    if (it == into.end()) into.push_back(origin);
    else join(it->clock, origin.clock);
  }
  Id tick(Id actor) {
    auto& clock = thread(actor).clock;
    if (clock[actor] == std::numeric_limits<Id>::max() || serial_ == std::numeric_limits<Id>::max())
      throw std::overflow_error("memory-order event counter exhausted");
    events_[++serial_] = {actor, ++clock[actor]}; return serial_;
  }
  static bool same(const Access& a, const Access& b) {
    return a.location == b.location && a.bytes == b.bytes;
  }
  bool moral(const Access& a, const Access& b) const {
    return same(a, b) && (a.actor == b.actor || (a.strong && b.strong && mutual(a.actor, a.scope, b.actor, b.scope)));
  }
  bool ordered(const Access& prior, const Access& now) const {
    return prior.actor == now.actor || prior.time <= value(thread(now.actor).clock, prior.actor) || moral(prior, now);
  }
  static void remember(std::vector<Access>& history, const Access& now) {
    // Scope and exact-width strength classes must survive separately. In
    // particular a later GPU strong access cannot erase an earlier weak one.
    auto it = std::find_if(history.begin(), history.end(), [&](const Access& a) {
      return a.actor == now.actor && a.strong == now.strong &&
             (!now.strong || (same(a, now) && a.scope == now.scope));
    });
    if (it == history.end()) history.push_back(now); else *it = now;
  }
  void acquire(Id actor, Scope scope, const std::vector<Endpoint>& origins, std::optional<MemoryKey> memory = {}) {
    for (const Endpoint& p : origins)
      if ((!p.memory || !memory || p.memory == memory) && mutual(p.actor, p.scope, actor, scope)) join(thread(actor).clock, p.clock);
  }
 public:
  void add_actor(Id id, Actor actor) {
    if (threads_.contains(id)) throw std::invalid_argument("duplicate actor id");
    if (identities_.contains(actor)) throw std::invalid_argument("duplicate thread identity");
    threads_.emplace(id, Thread{actor, {{id, 1}}, {}, {}, {}});
    identities_.emplace(actor, id);
  }
  bool knows(Id actor, Id event) const {
    auto [source, time] = events_.at(event); return value(thread(actor).clock, source) >= time;
  }
  Id latest(Location location, unsigned bytes) const {
    if ((bytes != 1 && bytes != 2 && bytes != 4 && bytes != 8) || location.address % bytes ||
        bytes - 1 > std::numeric_limits<Id>::max() - location.address)
      throw std::invalid_argument("invalid scalar latest-version query");
    Id source = 0;
    for (unsigned b = 0; b < bytes; ++b) {
      auto it = memory_.find({location.name_space, location.address + b});
      Id next = it == memory_.end() ? 0 : it->second.latest;
      if (b && next != source) return 0;
      source = next;
    }
    return source;
  }
  Result access(Id actor, Location location, unsigned bytes, Kind kind, Sem sem = Sem::Weak,
                Scope scope = Scope::Gpu, std::optional<Id> observed = {}, bool did_write = true) {
    thread(actor); valid(scope);
    if (bytes != 1 && bytes != 2 && bytes != 4 && bytes != 8) throw std::invalid_argument("scalar memory width");
    if (location.address % bytes || bytes - 1 > std::numeric_limits<Id>::max() - location.address)
      throw std::invalid_argument("unaligned or overflowing scalar location");
    if (kind < Kind::Read || kind > Kind::Rmw || sem < Sem::Weak || sem > Sem::AcqRel ||
        (kind == Kind::Read && (sem == Sem::Release || sem == Sem::AcqRel)) ||
        (kind == Kind::Write && (sem == Sem::Acquire || sem == Sem::AcqRel)) ||
        (kind == Kind::Rmw && sem == Sem::Weak) || (kind != Kind::Rmw && !did_write) ||
        (kind == Kind::Write && observed)) throw std::invalid_argument("invalid memory operation semantics");
    const bool read = kind != Kind::Write, write = kind != Kind::Read && did_write;
    const bool strong = sem != Sem::Weak;
    const bool acq = sem == Sem::Acquire || sem == Sem::AcqRel;
    const bool rel = sem == Sem::Release || sem == Sem::AcqRel;
    // An explicit observed identity denotes a complete scalar read-from edge.
    // Zero is initialization. A fragmented latest read has no complete edge.
    Id source = read ? observed.value_or(latest(location, bytes)) : 0;
    if (!observed && source && (versions_.at(source).access.location != location || versions_.at(source).access.bytes != bytes)) source = 0;
    if (source && (!versions_.contains(source) || versions_.at(source).access.location != location ||
                   versions_.at(source).access.bytes != bytes)) throw std::invalid_argument("read-from location/width mismatch");
    Access current{0, actor, 0, location, bytes, strong, write, scope};
    std::vector<Endpoint> seen;
    if (read && strong && source && moral(versions_.at(source).access, current)) seen = versions_.at(source).publications;
    auto& state = thread(actor);
    const MemoryKey key{location, bytes};
    if (read && strong) for (const auto& p : seen) retain(state.observations[key], p);
    if (acq) acquire(actor, scope, state.observations[key], key);
    current.event = tick(actor); current.time = value(state.clock, actor);
    std::optional<Conflict> conflict;
    for (unsigned b = 0; b < bytes; ++b) {
      auto& cell = memory_[{location.name_space, location.address + b}];
      for (const Access& prior : cell.writers) if (!ordered(prior, current) && !conflict)
        conflict = Conflict{location.address + b, prior.event, prior.actor, true, write};
      if (write) {
        for (const Access& prior : cell.readers) if (!ordered(prior, current) && !conflict)
          conflict = Conflict{location.address + b, prior.event, prior.actor, false, true};
        remember(cell.writers, current); cell.latest = current.event;
      } else remember(cell.readers, current);
    }
    // Failed CAS is a read-only RMW: it may acquire and establish its own
    // release pattern, but cannot replace the publication/version it observed.
    if (rel) retain(state.releases[key], endpoint(actor, scope, key));
    if (write) {
      std::vector<Endpoint> publications;
      if (strong) {
        for (const auto& p : state.releases[key]) retain(publications, p);
        for (const auto& p : state.fences) retain(publications, p);
        if (kind == Kind::Rmw) for (const auto& p : seen) retain(publications, p);
      }
      versions_.emplace(current.event, Version{current, std::move(publications)});
    }
    return {current.event, conflict};
  }
  Id fence(Id actor, Scope scope, bool sequentially_consistent = false) {
    thread(actor); valid(scope);
    auto& state = thread(actor);
    for (const auto& [_, origins] : state.observations) acquire(actor, scope, origins);
    if (sequentially_consistent) for (const auto& p : sc_fences_)
      if (mutual(p.actor, p.scope, actor, scope)) join(state.clock, p.clock);
    Id event = tick(actor);
    retain(state.fences, endpoint(actor, scope));
    if (sequentially_consistent) retain(sc_fences_, endpoint(actor, scope));
    return event;
  }
  Id arrive(Id barrier, Id actor) {
    auto& state = thread(actor); const Actor& owner = state.owner;
    if (owner.host) throw std::invalid_argument("host CTA barrier participant");
    auto it = barriers_.find(barrier);
    if (it != barriers_.end() && (it->second.device != owner.device || it->second.grid != owner.grid ||
        it->second.cta != owner.cta || it->second.arrivals.contains(actor)))
      throw std::invalid_argument("barrier domain or duplicate arrival");
    Id event = tick(actor);
    if (it == barriers_.end()) it = barriers_.emplace(barrier, Barrier{owner.device, owner.grid, owner.cta, {}}).first;
    it->second.arrivals.emplace(actor, state.clock); return event;
  }
  void complete(Id barrier, const std::vector<Id>& waiters) {
    const auto& phase = barriers_.at(barrier);
    std::map<Id, bool> unique;
    for (Id actor : waiters) if (!phase.arrivals.contains(actor) || !unique.emplace(actor, true).second)
      throw std::invalid_argument("nonparticipant or duplicate CTA waiter");
    Clock joined; for (const auto& [_, clock] : phase.arrivals) join(joined, clock);
    for (Id actor : waiters) { join(thread(actor).clock, joined); tick(actor); }
    barriers_.erase(barrier);
  }
};
}
