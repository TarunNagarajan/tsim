// PTX fence operand inspection and eager interpreter regressions. The Python
// driver supplies independently specified cases and NVIDIA assembler results.
#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/ptx/regalloc.hpp"
#include "vgpu/registry.hpp"
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

using namespace vgpu;
using namespace vgpu::ptx;
namespace {
unsigned checks = 0;
void check(bool ok, const std::string& why) {
  ++checks;
  if (!ok) throw std::runtime_error(why);
}
std::string module(const std::string& body) {
  return ".version 8.5\n.target sm_90\n.address_size 64\n"
         ".visible .entry k(.param .u64 out){ .reg .b64 %rd<8>; .reg .b32 %r<8>; "
         ".reg .pred %p<2>; " + body + " ret; }\n";
}
std::vector<uint8_t> arg(uint64_t value) {
  std::vector<uint8_t> bytes(8); std::memcpy(bytes.data(), &value, 8); return bytes;
}
void inspect(const char* path) {
  std::ifstream f(path);
  if (!f) throw std::runtime_error("missing PTX input");
  std::ostringstream text; text << f.rdbuf();
  auto m = parse(text.str());
  const Instr* found = nullptr;
  for (const auto& i : m.entries.at(0).body) if (std::holds_alternative<OpFence>(i.op)) {
    if (found) throw std::runtime_error("inspection requires exactly one fence");
    found = &i;
  }
  if (!found) throw std::runtime_error("fence missing from AST");
  const auto& op = std::get<OpFence>(found->op);
  std::vector<uint32_t> defs, uses; instr_registers(*found, defs, uses);
  const char* kinds[] = {"thread", "membar", "alias", "async", "tensormap", "mbarrier_init"};
  const char* sems[] = {"acq_rel", "sc", "acquire", "release", "bidirectional", "legacy_membar"};
  const char* scopes[] = {"none", "cta", "cluster", "gpu", "sys"};
  const char* spaces[] = {"all", "global", "shared::cta", "shared::cluster"};
  const char* bases[] = {"reg", "entry_param", "call_slot", "symbol"};
  std::cout << "{\"kind\":\"" << kinds[unsigned(op.kind)] << "\",\"sem\":\"" << sems[unsigned(op.sem)]
            << "\",\"scope\":\"" << scopes[unsigned(op.scope)] << "\",\"space\":\"" << spaces[unsigned(op.space)]
            << "\",\"range\":" << op.has_range << ",\"bytes\":" << op.bytes
            << ",\"base\":\"" << bases[unsigned(op.addr.base_kind)] << "\",\"wide\":" << op.addr.base_wide
            << ",\"offset\":" << op.addr.offset << ",\"defs\":" << defs.size() << ",\"uses\":" << uses.size()
            << ",\"address_used\":" << (op.has_range && op.addr.base_kind == Addr::Base::Reg &&
                std::count(uses.begin(), uses.end(), op.addr.base_id * 2 + unsigned(op.addr.base_wide)) == 1)
            << "}\n";
}
void runtime() {
  const auto profile = load_gpu("nvidia/h100");
  MemoryManager mem(1 << 20);
  const auto out = mem.alloc(192 * 4);
  // Only one lane in each CTA executes these fences. A fence must not wait
  // for the other warps/lanes or act as a barrier. Each lane owns its output.
  const auto diverged = parse(module(
      "ld.param.u64 %rd1, [out]; mov.u32 %r1, %tid.x; mov.u32 %r2, %ctaid.x; "
      "setp.eq.u32 %p1, %r1, 0; @!%p1 bra write; "
      "membar.cta; membar.gl; membar.sys; fence.cta; fence.acq_rel.gpu; fence.sc.sys; fence.sc.cluster; "
      "fence.proxy.alias; membar.proxy.alias; fence.proxy.async; fence.proxy.async.global; "
      "fence.proxy.async.shared::cta; fence.proxy.async.shared::cluster; fence.mbarrier_init.release.cluster; "
      "write: mad.lo.u32 %r3, %r2, 96, %r1; mul.wide.u32 %rd2, %r3, 4; add.u64 %rd3, %rd1, %rd2; "
      "mad.lo.u32 %r4, %r3, 3, 1; st.global.u32 [%rd3], %r4;"));
  for (auto scheduler : {exec::SchedulerKind::Deterministic, exec::SchedulerKind::Random,
                         exec::SchedulerKind::Adversarial}) {
    exec::LaunchConfig cfg; cfg.block = {96, 1, 1}; cfg.grid = {2, 1, 1};
    cfg.scheduler = scheduler; cfg.scheduler_seed = 751; cfg.max_steps = 1000;
    const uint8_t zero = 0; mem.fill(out, &zero, 1, 192 * 4);
    exec::launch(diverged.entries.at(0), cfg, {arg(out)}, mem, profile);
    std::vector<uint32_t> values(192); mem.read(out, values.data(), values.size() * 4);
    for (unsigned i = 0; i < values.size(); ++i) check(values[i] == i * 3 + 1, "divergent fence blocked or changed execution");
  }
  // A ranged proxy fence consumes the address without loading/storing the
  // range or requiring the other lanes to execute it.
  exec::LaunchConfig cfg; cfg.block = {32, 1, 1}; cfg.max_steps = 1000;
  for (const auto& prefix : {std::string("ld.param.u64 %rd1, [out];"),
                             std::string("ld.param.u64 %rd1, [out]; add.u64 %rd1, %rd1, 16;")}) {
    const uint8_t pattern = 0x5a; mem.fill(out, &pattern, 1, 192 * 4);
    auto m = parse(module(prefix + " fence.proxy.tensormap::generic.release.gpu; "
                         "fence.proxy.tensormap::generic.acquire.gpu [%rd1], 128;"));
    exec::launch(m.entries.at(0), cfg, {arg(out)}, mem, profile);
    std::vector<uint8_t> bytes(192 * 4); mem.read(out, bytes.data(), bytes.size());
    for (auto b : bytes) check(b == 0x5a, "proxy fence changed memory bytes");
  }
  const auto expect_error = [&](const std::string& body, Err code) {
    auto m = parse(module(body));
    try { exec::launch(m.entries.at(0), cfg, {arg(out)}, mem, profile); }
    catch (const Error& e) { check(e.code() == code, "wrong ranged-fence diagnostic: " + e.message()); return; }
    check(false, "ranged fence failed to diagnose invalid operand");
  };
  expect_error("fence.proxy.tensormap::generic.acquire.gpu [%rd1], 128;", Err::UninitializedRegister);
  expect_error("fence.proxy.tensormap::generic.acquire.gpu [%r1], 128;", Err::UninitializedRegister);
  // The engine's documented private VA windows, not physical GPU pointers.
  for (auto address : {uint64_t{0x6ffe00000000}, uint64_t{0x6fff00000000},
                       uint64_t{0x6ffd00000000}, uint64_t{UINT64_MAX - 32}})
    expect_error("mov.u64 %rd1, " + std::to_string(address) +
                 "; fence.proxy.tensormap::generic.acquire.gpu [%rd1], 128;", Err::InvalidValue);
  // A false predicate must suppress operand evaluation as well as ordering.
  auto skipped = parse(module("mov.u32 %r1, 0; setp.ne.u32 %p1, %r1, 0; "
                              "@%p1 fence.proxy.tensormap::generic.acquire.gpu [%rd1], 128;"));
  exec::launch(skipped.entries.at(0), cfg, {arg(out)}, mem, profile);
  check(true, "false-predicate ranged fence");
  std::cout << "PASS fence_runtime checks=" << checks << "\n";
}
} // namespace
int main(int argc, char** argv) {
  try {
    if (argc == 3 && std::string(argv[1]) == "inspect") inspect(argv[2]);
    else if (argc == 2 && std::string(argv[1]) == "runtime") runtime();
    else throw std::runtime_error("expected inspect PTX or runtime");
    return 0;
  } catch (const Error& e) { std::cerr << e.what() << '\n'; return 2; }
    catch (const std::exception& e) { std::cerr << "FAIL fence_semantics: " << e.what() << '\n'; return 1; }
}
