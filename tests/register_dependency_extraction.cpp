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
    std::printf("PASS register dependency extraction checks=%u\n", checks);
    return 0;
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what()); return 1;
  }
}
