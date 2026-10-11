#include "tsim/scoped_memory_order.hpp"
#include <array>
#include <cstdio>
#include <functional>
#include <map>
#include <set>
#include <string>

using Model = tsim::ScopedMemoryOrder;
using Scope = Model::Scope; using Sem = Model::Sem; using Kind = Model::Kind;
using Id = Model::Id; using Location = Model::Location;
unsigned checks = 0, relations = 0, scenarios = 0;
void require(bool condition, const char* name) {
  if (!condition) throw std::runtime_error(name);
  ++checks;
}
void reject(const char* name, const std::function<void()>& action) {
  try { action(); } catch (const std::invalid_argument&) { ++checks; return; }
  catch (const std::out_of_range&) { ++checks; return; }
  throw std::runtime_error(name);
}
// Independent reference: reverse reachability in an explicit causal DAG,
// without vector clocks, production scope helpers or publication tables.
struct Dag {
  std::map<Id, std::vector<Id>> parents;
  std::map<Id, Id> last;
  void add(Id event, Id actor, std::vector<Id> sync = {}) {
    if (last.contains(actor)) sync.push_back(last.at(actor));
    parents.emplace(event, std::move(sync)); last[actor] = event;
  }
  bool knows(Id actor, Id event) const {
    if (!last.contains(actor)) return false;
    std::vector<Id> pending{last.at(actor)}; std::set<Id> seen;
    while (!pending.empty()) {
      Id next = pending.back(); pending.pop_back();
      if (next == event) return true;
      if (!seen.insert(next).second) continue;
      for (Id prior : parents.at(next)) pending.push_back(prior);
    }
    return false;
  }
  void audit(const Model& model, unsigned actors) const {
    for (Id actor = 0; actor < actors; ++actor) for (const auto& [event, _] : parents) {
      if (model.knows(actor, event) != knows(actor, event)) throw std::runtime_error("independent DAG/vector-clock disagreement");
      ++relations;
    }
  }
};
Model model(unsigned count = 3) {
  Model m; for (unsigned i = 0; i < count; ++i) m.add_actor(i, {0, 1, i, i, 0}); return m;
}
constexpr Location payload{0, 0}, flag{0, 8}, other{0, 16};
struct Context { Model::Actor receiver; std::array<bool, 4> covers; };
const std::array<Context, 6> contexts{{
  {{0,1,0,0,1}, {true,true,true,true}},
  {{0,1,1,0,0}, {false,true,true,true}},
  {{0,1,1,1,0}, {false,false,true,true}},
  {{0,2,0,0,0}, {false,false,true,true}},
  {{1,1,0,0,0}, {false,false,false,true}},
  {{0,0,0,0,0,true}, {false,false,false,true}}
}};
bool pair(const Context& context, unsigned first, unsigned second) {
  return context.covers[first] && context.covers[second];
}

int main() {
  try {
    for (const auto& context : contexts) for (unsigned x = 0; x < 4; ++x) for (unsigned y = 0; y < 4; ++y) {
      Model m; m.add_actor(0,{0,1,0,0,0}); m.add_actor(1,context.receiver); Dag dag;
      auto p = m.access(0,payload,4,Kind::Write); dag.add(p.event,0);
      auto w = m.access(0,flag,4,Kind::Write,Sem::Release,Scope(x)); dag.add(w.event,0);
      auto r = m.access(1,flag,4,Kind::Read,Sem::Acquire,Scope(y));
      dag.add(r.event,1,pair(context,x,y) ? std::vector<Id>{w.event} : std::vector<Id>{});
      require(bool(r.conflict) != pair(context,x,y),"strong flag race requires mutual scope");
      auto data = m.access(1,payload,4,Kind::Read); dag.add(data.event,1);
      require(bool(data.conflict) != pair(context,x,y),"scoped release/acquire payload");
      dag.audit(m,2); ++scenarios;
    }
    // The first release and final acquire scopes are independent of the
    // scopes of their intervening strong write/read operations.
    for (const auto& context : contexts) for (unsigned x=0;x<4;++x) for (unsigned wscope=0;wscope<4;++wscope)
      for (unsigned rscope=0;rscope<4;++rscope) for (unsigned y=0;y<4;++y) {
        Model m; m.add_actor(0,{0,1,0,0,0}); m.add_actor(1,context.receiver); Dag dag;
        auto p=m.access(0,payload,4,Kind::Write); dag.add(p.event,0);
        auto release=m.access(0,flag,4,Kind::Write,Sem::Release,Scope(x)); dag.add(release.event,0);
        auto late=m.access(0,other,4,Kind::Write); dag.add(late.event,0);
        auto carrier=m.access(0,flag,4,Kind::Write,Sem::Relaxed,Scope(wscope)); dag.add(carrier.event,0);
        auto observed=m.access(1,flag,4,Kind::Read,Sem::Relaxed,Scope(rscope)); dag.add(observed.event,1);
        auto overwritten=m.access(0,flag,4,Kind::Write); dag.add(overwritten.event,0);
        auto acquire=m.access(1,flag,4,Kind::Read,Sem::Acquire,Scope(y));
        const bool sync=pair(context,wscope,rscope)&&pair(context,x,y);
        dag.add(acquire.event,1,sync?std::vector<Id>{release.event}:std::vector<Id>{});
        require(m.knows(1,p.event)==sync,"pattern endpoints retain first release scope");
        require(!m.knows(1,late.event),"release snapshot excludes post-release writes");
        dag.audit(m,2); ++scenarios;
      }
    for (const auto& context : contexts) for (unsigned x=0;x<4;++x) for (unsigned y=0;y<4;++y)
      for (bool first_sc : {false,true}) for (bool second_sc : {false,true}) {
      Model m; m.add_actor(0,{0,1,0,0,0}); m.add_actor(1,context.receiver); Dag dag;
      auto p=m.access(0,payload,4,Kind::Write); dag.add(p.event,0);
      Id first=m.fence(0,Scope(x),first_sc); dag.add(first,0);
      Id second=m.fence(1,Scope(y),second_sc); dag.add(second,1,first_sc&&second_sc&&pair(context,x,y)?std::vector<Id>{first}:std::vector<Id>{});
      auto r=m.access(1,payload,4,Kind::Read); dag.add(r.event,1);
      require(bool(r.conflict)!=(first_sc&&second_sc&&pair(context,x,y)),"SC fences synchronize only at mutual scope");
      dag.audit(m,2); ++scenarios;
    }
    for (const auto& context : contexts) for (unsigned x=0;x<4;++x) for (unsigned wscope=0;wscope<4;++wscope)
      for (unsigned rscope=0;rscope<4;++rscope) for (unsigned y=0;y<4;++y) {
        Model m; m.add_actor(0,{0,1,0,0,0}); m.add_actor(1,context.receiver); Dag dag;
        auto p=m.access(0,payload,4,Kind::Write); dag.add(p.event,0);
        Id release=m.fence(0,Scope(x)); dag.add(release,0);
        auto late=m.access(0,other,4,Kind::Write); dag.add(late.event,0);
        auto carrier=m.access(0,flag,4,Kind::Write,Sem::Relaxed,Scope(wscope)); dag.add(carrier.event,0);
        auto observed=m.access(1,flag,4,Kind::Read,Sem::Relaxed,Scope(rscope)); dag.add(observed.event,1);
        Id acquire=m.fence(1,Scope(y));
        const bool sync=pair(context,wscope,rscope)&&pair(context,x,y);
        dag.add(acquire,1,sync?std::vector<Id>{release}:std::vector<Id>{});
        require(m.knows(1,p.event)==sync,"fence pattern endpoint and carrier scopes");
        require(!m.knows(1,late.event),"fence pattern excludes post-fence payload");
        dag.audit(m,2); ++scenarios;
      }
    for (unsigned x=0;x<4;++x) for (unsigned relay=0;relay<4;++relay) for (unsigned y=0;y<4;++y) {
      auto m=model(); Dag dag;
      auto p=m.access(0,payload,4,Kind::Write); dag.add(p.event,0);
      auto w=m.access(0,flag,4,Kind::Write,Sem::Release,Scope(x)); dag.add(w.event,0);
      auto rmw=m.access(1,flag,4,Kind::Rmw,Sem::Relaxed,Scope(relay)); dag.add(rmw.event,1);
      require(!m.knows(1,p.event),"relaxed RMW carries without acquiring");
      auto r=m.access(2,flag,4,Kind::Read,Sem::Acquire,Scope(y));
      const bool sync=x>=2&&relay>=2&&y>=2;
      dag.add(r.event,2,sync?std::vector<Id>{w.event}:std::vector<Id>{});
      require(m.knows(2,p.event)==sync,"every RMW observation link needs mutual scope");
      dag.audit(m,3); ++scenarios;
    }
    {
      auto m=model(); auto p=m.access(0,payload,4,Kind::Write);
      m.fence(0,Scope::Gpu); m.access(0,flag,4,Kind::Write,Sem::Relaxed);
      m.access(1,flag,4,Kind::Read,Sem::Relaxed); m.fence(1,Scope::Gpu);
      require(m.knows(1,p.event)&&!m.access(1,payload,4,Kind::Read).conflict,"fence/strong-write/read/fence publication");
    }
    {
      auto m=model(); auto p=m.access(0,payload,4,Kind::Write); m.fence(0,Scope::Gpu);
      auto later=m.access(0,other,4,Kind::Write); m.access(0,flag,4,Kind::Write,Sem::Relaxed);
      m.access(1,flag,4,Kind::Read,Sem::Acquire);
      require(m.knows(1,p.event)&&!m.knows(1,later.event),"fence release snapshot excludes later payload");
    }
    {
      auto m=model(); auto p=m.access(0,payload,4,Kind::Write); m.access(0,flag,4,Kind::Write,Sem::Release);
      m.access(1,flag,4,Kind::Write,Sem::Relaxed); m.access(2,flag,4,Kind::Read,Sem::Acquire);
      require(!m.knows(2,p.event),"unrelated plain writer breaks observation chain");
    }
    {
      auto m=model(); auto p=m.access(0,payload,4,Kind::Write); m.access(0,flag,4,Kind::Write,Sem::Release);
      m.access(1,flag,4,Kind::Write,Sem::Relaxed); m.access(0,flag,4,Kind::Write,Sem::Relaxed);
      m.access(2,flag,4,Kind::Read,Sem::Acquire);
      require(m.knows(2,p.event),"own release pattern survives intervening writer");
    }
    {
      auto m=model(); auto p=m.access(0,payload,4,Kind::Write); m.access(0,flag,4,Kind::Write,Sem::Release);
      auto q=m.access(1,other,4,Kind::Write); m.access(1,flag,4,Kind::Rmw,Sem::Release);
      m.access(2,flag,4,Kind::Read,Sem::Acquire);
      require(m.knows(2,p.event)&&m.knows(2,q.event),"release RMW carries origin and own payload");
    }
    {
      auto m=model(); auto p=m.access(0,payload,4,Kind::Write); auto w=m.access(0,flag,4,Kind::Write,Sem::Release);
      auto cas=m.access(1,flag,4,Kind::Rmw,Sem::Acquire,Scope::Gpu,{},false);
      require(m.latest(flag,4)==w.event&&m.knows(1,p.event),"failed CAS acquires without replacing write version");
      require(m.access(2,flag,4,Kind::Write).conflict.has_value(),"failed CAS records a reader conflict");
      require(cas.event!=w.event,"failed CAS has its own event identity");
    }
    {
      auto m=model(); auto p=m.access(0,payload,4,Kind::Write); m.access(0,flag,4,Kind::Write,Sem::Release);
      m.access(0,flag,8,Kind::Write,Sem::Relaxed); m.access(1,flag,8,Kind::Read,Sem::Acquire);
      require(!m.knows(1,p.event),"wider strong write cannot extend narrower release pattern");
    }
    {
      auto m=model(); m.access(0,payload,8,Kind::Write,Sem::Relaxed);
      require(m.access(1,payload,4,Kind::Read,Sem::Acquire).conflict.has_value(),"mixed-size strong read remains a partial-overlap race");
      require(m.access(2,{0,4},4,Kind::Read).conflict.has_value(),"high-half read retains original byte writer");
      require(!m.access(1,{0,8},1,Kind::Write).conflict,"adjacent byte is independent");
    }
    {
      Model m; m.add_actor(0,{0,1,0,0,0}); m.add_actor(1,{0,1,0,0,1});
      auto p=m.access(0,payload,4,Kind::Write); m.access(0,{1,8},4,Kind::Write,Sem::Release,Scope::Cta);
      m.access(1,{1,8},4,Kind::Read,Sem::Acquire,Scope::Cta);
      require(m.knows(1,p.event),"causality closes across canonical state-space namespaces");
    }
    {
      auto m=model(); m.access(0,payload,4,Kind::Write); m.access(0,payload,4,Kind::Write,Sem::Relaxed);
      require(m.access(1,payload,4,Kind::Read,Sem::Relaxed).conflict.has_value(),"strong writer retains earlier weak history");
    }
    {
      auto m=model(); m.access(0,payload,4,Kind::Write,Sem::Relaxed,Scope::Cta);
      m.access(0,payload,4,Kind::Write,Sem::Relaxed,Scope::Gpu);
      require(m.access(1,payload,4,Kind::Read,Sem::Relaxed).conflict.has_value(),"wider-scope writer retains earlier narrow-scope conflict");
    }
    {
      auto m=model(); m.access(0,payload,4,Kind::Read,Sem::Relaxed,Scope::Cta);
      m.access(0,payload,4,Kind::Read,Sem::Relaxed,Scope::Gpu);
      require(m.access(1,payload,4,Kind::Write,Sem::Relaxed).conflict.has_value(),"wider-scope reader retains earlier narrow-scope conflict");
    }
    {
      Model m; m.add_actor(0,{0,1,0,0,0}); m.add_actor(1,{0,1,0,0,1});
      auto p=m.access(0,payload,4,Kind::Write); Id first=m.arrive(7,0);
      auto late=m.access(0,other,4,Kind::Write); m.arrive(7,1); m.complete(7,{1});
      require(m.knows(1,p.event)&&m.knows(1,first)&&!m.knows(1,late.event),"CTA waiter receives only arrival snapshot");
      auto q=m.access(1,{0,24},4,Kind::Write);
      require(!m.knows(0,q.event),"arrival-only participant does not acquire waiter clock");
      require(!m.access(1,payload,4,Kind::Read).conflict,"CTA barrier orders global payload");
      require(m.access(1,other,4,Kind::Read).conflict.has_value(),"post-arrival global write remains unordered");
      m.arrive(7,0); m.arrive(7,1); m.complete(7,{0,1});
      require(m.knows(0,q.event),"reused phase joins new arrivals");
    }
    {
      Model m; m.add_actor(0,{0,1,0,0,0}); m.add_actor(1,{0,1,0,0,1});
      m.access(0,payload,4,Kind::Write);
      require(m.access(1,payload,4,Kind::Read).conflict.has_value(),"different lanes are independent actors even within one warp");
      require(!m.access(0,payload,4,Kind::Read).conflict,"same thread remains ordered");
      require(!m.access(1,{1,0},4,Kind::Read).conflict,"distinct canonical namespaces do not alias");
    }
    {
      auto m=model(); auto p=m.access(0,payload,4,Kind::Write);
      m.access(0,flag,4,Kind::Write,Sem::Release); m.access(1,flag,4,Kind::Read,Sem::Acquire,Scope::Gpu,0);
      require(!m.knows(1,p.event),"initialization read does not invent publication from latest write");
    }
    {
      auto m=model(); auto p=m.access(0,payload,4,Kind::Write); auto old=m.access(0,flag,4,Kind::Write,Sem::Release);
      m.access(2,flag,4,Kind::Write,Sem::Relaxed); m.access(1,flag,4,Kind::Read,Sem::Acquire,Scope::Gpu,old.event);
      require(m.knows(1,p.event),"explicit observed write identity governs publication");
    }
    {
      auto m=model(); auto prior=m.access(0,payload,4,Kind::Write);
      reject("unknown actor",[&]{m.access(99,payload,4,Kind::Read);});
      reject("misaligned scalar",[&]{m.access(0,{0,1},4,Kind::Write);});
      reject("invalid width",[&]{m.access(0,payload,3,Kind::Write);});
      reject("unknown scope",[&]{m.access(0,payload,4,Kind::Write,Sem::Relaxed,Scope(99));});
      reject("read-from width mismatch",[&]{m.access(1,payload,8,Kind::Read,Sem::Acquire,Scope::Gpu,prior.event);});
      reject("read-from wrong address",[&]{m.access(1,flag,4,Kind::Read,Sem::Acquire,Scope::Gpu,prior.event);});
      reject("write cannot acquire",[&]{m.access(1,flag,4,Kind::Write,Sem::Acquire);});
      reject("weak RMW",[&]{m.access(1,flag,4,Kind::Rmw);});
      reject("duplicate actor id",[&]{m.add_actor(0,{0,2,0,0,0});});
      reject("duplicate thread identity",[&]{m.add_actor(7,{0,1,0,0,0});});
      require(m.latest(payload,4)==prior.event&&!m.knows(1,prior.event),"failed preflight preserves versions and clocks");
      m.arrive(9,0);
      reject("cross-CTA barrier join",[&]{m.arrive(9,1);});
      reject("duplicate arrival",[&]{m.arrive(9,0);});
      reject("nonparticipant waiter",[&]{m.complete(9,{1});}); m.complete(9,{0});
    }
    std::printf("PASS scoped_memory_order checks=%u scenarios=%u dag_relations=%u\n",checks,scenarios,relations);
    return 0;
  } catch (const std::exception& e) { std::fprintf(stderr,"FAIL scoped_memory_order: %s\n",e.what()); return 1; }
}
