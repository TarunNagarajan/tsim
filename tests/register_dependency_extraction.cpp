#include "vgpu/ptx/parser.hpp"
#include "vgpu/ptx/regalloc.hpp"
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

int main() {
  using namespace vgpu::ptx;
  unsigned checks = 0;
  auto check = [&](const std::string& instruction, const std::vector<std::string>& destination,
                   const std::vector<std::string>& source) {
    auto module = parse(".version 8.5\n.target sm_90\n.address_size 64\n"
      ".visible .entry k(){.reg .b32 %r<5>; .reg .b64 %rd<3>; .reg .pred %p<3>;\n" + instruction + "; ret; }");
    const auto& function = module.entries[0];
    if (function.body[0].text.empty()) throw std::runtime_error("Missing parsed instruction source text");
    std::vector<uint32_t> defs, uses;
    instr_registers(function.body[0], defs, uses);
    auto keys = [&](const std::vector<std::string>& names) {
      std::vector<uint32_t> result;
      for (const auto& name : names) result.push_back(function.reg_ids.at(name) * 2 + function.reg_wide.at(name));
      std::sort(result.begin(), result.end()); return result;
    };
    std::sort(defs.begin(), defs.end()); std::sort(uses.begin(), uses.end());
    if (defs != keys(destination) || uses != keys(source)) throw std::runtime_error("Wrong dependency operands: " + instruction);
    ++checks;
  };
  try {
    check("mov.b64 {%r1,_},%rd1", {"%r1"}, {"%rd1"});
    check("atom.global.add.u32 %r1,[%rd1],1", {"%r1"}, {"%rd1"});
    check("atom.global.add.u32 %r1,[%rd1],%r2", {"%r1"}, {"%rd1", "%r2"});
    check("atom.global.cas.b32 %r1,[%rd1],%r2,%r3", {"%r1"}, {"%rd1", "%r2", "%r3"});
    check("red.global.add.u32 [%rd1],%r2", {}, {"%rd1", "%r2"});
    check("setp.eq.u32 %p1,%r1,%r2", {"%p1"}, {"%r1", "%r2"});
    check("mad.lo.u32 %r1,%r2,%r3,1", {"%r1"}, {"%r2", "%r3"});
    check("@%p1 add.u32 %r1,%r2,%r3", {"%r1"}, {"%p1", "%r2", "%r3"});
    check("bar.warp.sync 0xffffffff", {}, {});
    check("fns.b32 %r1,%r2,%r3,%r4", {"%r1"}, {"%r2", "%r3", "%r4"});
    check("vote.sync.ballot.b32 %r1,%p1,%r4", {"%r1"}, {"%p1", "%r4"});
    check("redux.sync.add.u32 %r1,%r2,%r3", {"%r1"}, {"%r2", "%r3"});
    check("match.any.sync.b32 %r1,%r2,%r3", {"%r1"}, {"%r2", "%r3"});
    check("match.all.sync.b32 _|%p1,%r2,%r3", {"%p1"}, {"%r2", "%r3"});
    check("match.all.sync.b32 %r1|_,%r2,%r3", {"%r1"}, {"%r2", "%r3"});
    for (const std::string instruction : {"match.all.sync.b32 _|_,%r2,%r3",
                                          "match.all.sync.b32 %r1,%r2,%r3",
                                          "match.any.sync.b32 _,%r2,%r3"}) {
      bool rejected = false;
      try {
        parse(".version 8.5\n.target sm_90\n.address_size 64\n.visible .entry k(){"
              ".reg .b32 %r<5>; .reg .pred %p<3>; " + instruction + "; ret; }");
      } catch (const std::exception&) { rejected = true; }
      if (!rejected) throw std::runtime_error("Invalid match result syntax accepted: " + instruction);
      ++checks;
    }
    check("elect.sync %r1|%p1,%r2", {"%r1", "%p1"}, {"%r2"});
    check("bar.sync %r1,%r2", {}, {"%r1", "%r2"});
    check("bar.warp.sync %r1", {}, {"%r1"});
    check("mbarrier.init.shared.b64 [%rd1],%r2", {}, {"%rd1", "%r2"});
    check("mbarrier.try_wait.parity.shared.b64 %p1,[%rd1],%r2", {"%p1"}, {"%rd1", "%r2"});
    check("mapa.shared::cluster.u64 %rd1,%rd2,%r1", {"%rd1"}, {"%rd2", "%r1"});
    // An indirect call target is a register operand even though arguments and
    // return values use separate ABI slots and still require further capture.
    Instr call;
    OpCall operation; operation.indirect = true; operation.target_reg.id = 3; operation.target_reg.wide = true;
    call.op = operation;
    std::vector<uint32_t> defs, uses; instr_registers(call, defs, uses);
    if (!defs.empty() || uses != std::vector<uint32_t>{7}) throw std::runtime_error("Indirect target dependency missing");
    ++checks;
    operation.indirect = false; call.op = operation; defs.clear(); uses.clear(); instr_registers(call, defs, uses);
    if (!defs.empty() || !uses.empty()) throw std::runtime_error("Direct call has fabricated register target");
    ++checks;
    // AST references cover optional forms independently of parser defaults.
    // Inactive operands are deliberately populated, so an ungated collector
    // cannot pass by depending on default kNoReg initialization.
    auto reg = [](uint32_t id, bool wide = false) { return Reg{"reference", id, wide}; };
    auto operand = [&](uint32_t id, bool wide = false) -> Operand { return RegOperand{reg(id, wide)}; };
    auto address = [](uint32_t id, bool wide = false) {
      Addr at; at.base_kind = Addr::Base::Reg; at.base_id = id; at.base_wide = wide; return at;
    };
    auto ast = [&](const char* name, Op op, std::vector<uint32_t> expected_defs, std::vector<uint32_t> expected_uses) {
      Instr instruction; instruction.op = std::move(op);
      std::vector<uint32_t> actual_defs, actual_uses;
      instr_registers(instruction, actual_defs, actual_uses);
      auto sorted = [](auto& values) { std::sort(values.begin(), values.end()); };
      sorted(actual_defs); sorted(actual_uses); sorted(expected_defs); sorted(expected_uses);
      if (actual_defs != expected_defs || actual_uses != expected_uses)
        throw std::runtime_error(std::string("Wrong AST operand roles: ") + name);
      ++checks;
    };
    OpVote vote; vote.dst = reg(0); vote.src = reg(1); vote.members = operand(2);
    ast("legacy vote has no membership operand", vote, {0}, {2});
    vote.has_members = true; ast("synchronized vote reads membership", vote, {0}, {2, 4});
    OpShfl shuffle; shuffle.dst = reg(0); shuffle.a = operand(1); shuffle.b = ImmInt{1};
    shuffle.c = ImmInt{31}; shuffle.member_mask = operand(2); shuffle.has_members = false;
    ast("legacy shuffle has no membership operand", shuffle, {0}, {2});
    shuffle.has_members = true; ast("synchronized shuffle reads membership", shuffle, {0}, {2, 4});
    OpMbarrier barrier; barrier.addr = address(1, true); barrier.count = operand(2); barrier.state = operand(3, true);
    ast("barrier ignores absent count/state", barrier, {}, {3});
    barrier.have_count = true; ast("barrier reads explicit count", barrier, {}, {3, 4});
    barrier.have_count = false; barrier.have_state = true;
    ast("barrier reads explicit wide state", barrier, {}, {3, 7});
    OpMma mma; mma.d = {reg(0)}; mma.a = {reg(1)}; mma.b = {reg(2)}; mma.c = {reg(3)};
    mma.meta = operand(4); mma.selector = ImmInt{1};
    ast("dense MMA ignores sparse metadata", mma, {0}, {2, 4, 6});
    mma.sparse = true; ast("sparse MMA reads metadata and no immediate register", mma, {0}, {2, 4, 6, 8});
    OpWgmma wgmma; wgmma.d = {reg(0)}; wgmma.a = {reg(1)}; wgmma.a_desc = operand(2, true);
    wgmma.b_desc = operand(3, true); wgmma.scale_d = operand(4); wgmma.sp_meta = operand(5); wgmma.sp_sel = ImmInt{1};
    ast("dense WGMMA ignores sparse metadata", wgmma, {0}, {0, 2, 5, 7, 8});
    wgmma.sparse = true; ast("sparse WGMMA reads metadata", wgmma, {0}, {0, 2, 5, 7, 8, 10});
    OpCpAsyncGroup group; group.bar = address(1, true);
    ast("ordinary async commit has no barrier address", group, {}, {});
    group.kind = OpCpAsyncGroup::Kind::MbarrierArrive;
    ast("async barrier arrival reads address", group, {}, {3});
    OpStAsync store; store.addr = address(1); store.srcs = {operand(2)}; store.mbar = address(3, true);
    ast("async store reads data/destination/completion barrier", store, {}, {2, 4, 7});
    OpTensormapReplace replace; replace.addr = address(1, true); replace.value = operand(2, true);
    ast("tensor map replacement reads address and value", replace, {}, {3, 5});
    OpTcgen05 tensor; tensor.red = true;
    ast("discarded tensor reduction destination is no register", tensor, {}, {});
    tensor.red_dst = reg(1); ast("tensor reduction destination is defined", tensor, {2}, {});
    std::printf("PASS register dependency extraction checks=%u\n", checks);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what()); return 1;
  }
}
