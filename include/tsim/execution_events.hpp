#pragma once
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <exception>
#include <iomanip>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tsim {
// Actual interpreter events, not a hardware-cycle prediction. The instruction
// parent is thread-local: CTA workers must never share a current instruction.
class ExecutionEvents {
  std::mutex mutex_;
  FILE* file_ = nullptr;
  unsigned long long sequence_ = 0, limit_ = 200000;
  bool truncated_ = false;
  static inline thread_local unsigned long long parent_ = 0;
  unsigned long long write(const std::string& body) {
    if (!file_) return 0;
    std::lock_guard<std::mutex> lock(mutex_);
    if (truncated_) return 0;
    std::string row;
    if (sequence_ >= limit_) {
      truncated_ = true;
      row = "{\"type\":\"truncated\",\"limit\":" + std::to_string(limit_) + "}\n";
    } else {
      row = "{\"sequence\":" + std::to_string(++sequence_) + "," + body + "}\n";
    }
    if (std::fwrite(row.data(), 1, row.size(), file_) != row.size() || std::fflush(file_))
      throw std::runtime_error("tsim: failed to write execution events");
    return truncated_ ? 0 : sequence_;
  }
 public:
  static std::string quoted(const std::string& text) {
    std::ostringstream s; s << '"';
    for (unsigned char c : text) {
      if (c == '"' || c == '\\') s << '\\' << c;
      else if (c < 32) s << "\\u" << std::hex << std::setw(4) << std::setfill('0') << unsigned(c) << std::dec;
      else s << c;
    }
    s << '"'; return s.str();
  }
  ExecutionEvents() {
    const char* path = std::getenv("TSIM_EVENTS");
    if (!path || !*path) return;
    if (const char* cap = std::getenv("TSIM_EVENTS_LIMIT")) {
      char* end = nullptr; errno = 0;
      auto n = std::strtoull(cap, &end, 10);
      if (errno || end == cap || *end || cap[0] < '0' || cap[0] > '9' || !n)
        throw std::runtime_error("tsim: invalid execution event limit");
      limit_ = n;
    }
    file_ = std::fopen(path, "a");
    if (!file_) throw std::runtime_error("tsim: cannot open execution events: " + std::string(std::strerror(errno)));
    try {
      write("\"type\":\"stream\",\"schema\":1,\"backend\":\"CPU PTX interpreter\","
            "\"execution_path\":\"reference\",\"address_encoding\":\"hexadecimal virtual address\","
            "\"memory_coverage\":\"ld/st and scalar atom/red; other paths emit gaps\","
            "\"dependency_schema\":2,"
            "\"cta_barrier_schema\":1,\"cta_barrier_coverage\":\"actual converged arrivals, releases, result reads and kernel exits\","
            "\"collective_schema\":2,\"collective_coverage\":\"executed groups, virtual data-source lanes and exact parked membership reads\","
            "\"dependency_coverage\":\"virtual registers and CC.CF; call ABI and other implicit state incomplete\","
            "\"hardware_cycles\":false");
    } catch (...) { std::fclose(file_); file_ = nullptr; throw; }
  }
  ~ExecutionEvents() {
    if (!file_) return;
    try { write("\"type\":\"stream_end\""); }
    catch (...) { std::fputs("tsim: failed to finish execution events\n", stderr); }
    if (std::fclose(file_)) std::fputs("tsim: failed to close execution events\n", stderr);
  }
  bool enabled() const { return file_ != nullptr; }
  unsigned long long issue(unsigned long long launch, const std::string& kernel,
      const std::string& function, unsigned long long frame,
      const std::array<unsigned, 3>& block, unsigned warp, size_t pc, size_t line,
      unsigned long long path_mask, unsigned long long execution_mask,
      const std::string& instruction, const std::vector<uint32_t>& defs,
      const std::vector<uint32_t>& uses, long long predicate,
      unsigned regs32, unsigned regs64, unsigned local_bytes,
      unsigned memory_accesses = 0, unsigned memory_width = 0,
      bool carry_read = false, bool carry_write = false) {
    std::ostringstream s;
    s << "\"type\":\"issue\",\"launch\":" << launch << ",\"kernel\":" << quoted(kernel)
      << ",\"function\":" << quoted(function) << ",\"frame\":" << frame
      << ",\"block\":[" << block[0] << ',' << block[1] << ',' << block[2]
      << "],\"warp\":" << warp << ",\"pc\":" << pc << ",\"ptx_line\":" << line
      << ",\"path_mask\":" << path_mask << ",\"execution_mask\":" << execution_mask
      << ",\"instruction\":" << quoted(instruction);
    auto regs = [&](const char* key, const std::vector<uint32_t>& values) {
      s << ",\"" << key << "\":[";
      for (size_t i = 0; i < values.size(); ++i) { if (i) s << ','; s << values[i]; }
      s << ']';
    };
    regs("defs", defs); regs("uses", uses);
    s << ",\"implicit_uses\":[" << (carry_read ? "\"CC.CF\"" : "") << ']'
      << ",\"implicit_defs\":[" << (carry_write ? "\"CC.CF\"" : "") << ']';
    s << ",\"predicate_use\":" << predicate
      << ",\"virtual_registers32\":" << regs32 << ",\"virtual_registers64\":" << regs64
      << ",\"declared_local_bytes\":" << local_bytes
      << ",\"expected_memory_accesses\":" << memory_accesses
      << ",\"expected_memory_width\":" << memory_width;
    return write(s.str());
  }
  void memory(unsigned lane, uint64_t address, unsigned bytes, const char* space,
              const char* operation, unsigned element = 0) {
    if (!parent_) return;
    std::ostringstream s;
    s << "\"type\":\"memory\",\"issue\":" << parent_ << ",\"lane\":" << lane
      << ",\"address\":\"0x" << std::hex << address << std::dec << "\",\"bytes\":" << bytes
      << ",\"space\":" << quoted(space) << ",\"operation\":" << quoted(operation)
      << ",\"element\":" << element << ",\"status\":\"completed\"";
    write(s.str());
  }
  void gap(const char* reason) {
    if (parent_) write("\"type\":\"coverage_gap\",\"issue\":" + std::to_string(parent_) + ",\"reason\":" + quoted(reason));
  }
  unsigned long long collective_group(const char* kind, unsigned long long mask,
      long long data_use, const std::vector<uint32_t>& data_defs) {
    if (!parent_) return 0;
    std::ostringstream s;
    s << "\"type\":\"collective_group\",\"issue\":" << parent_ << ",\"kind\":" << quoted(kind)
      << ",\"execution_mask\":" << mask << ",\"data_use\":" << data_use << ",\"data_defs\":[";
    for (size_t i = 0; i < data_defs.size(); ++i) { if (i) s << ','; s << data_defs[i]; }
    s << "]";
    return write(s.str());
  }
  void collective_transfer(unsigned long long group, unsigned destination_lane, unsigned long long source_mask) {
    if (!parent_ || !group) return;
    std::ostringstream s;
    s << "\"type\":\"collective_transfer\",\"issue\":" << parent_ << ",\"group\":" << group
      << ",\"destination_lane\":" << destination_lane << ",\"source_mask\":" << source_mask;
    write(s.str());
  }
  void collective_wait(long long member_use, unsigned long long member_lane_mask) {
    if (parent_) write("\"type\":\"collective_wait\",\"issue\":" + std::to_string(parent_) +
        ",\"member_use\":" + std::to_string(member_use) + ",\"member_lane_mask\":" + std::to_string(member_lane_mask));
  }
  void cta_barrier_wait(unsigned long long mask) {
    if (parent_) write("\"type\":\"cta_barrier_wait\",\"issue\":" + std::to_string(parent_) +
        ",\"execution_mask\":" + std::to_string(mask));
  }
  unsigned long long cta_barrier_arrive(unsigned id, unsigned long long phase, unsigned count,
      unsigned reduction, bool blocking, unsigned long long mask, unsigned votes) {
    if (!parent_) return 0;
    std::ostringstream s;
    s << "\"type\":\"cta_barrier_arrive\",\"issue\":" << parent_ << ",\"barrier\":" << id
      << ",\"phase\":" << phase << ",\"count\":" << count << ",\"reduction\":" << reduction
      << ",\"blocking\":" << (blocking ? "true" : "false") << ",\"execution_mask\":" << mask
      << ",\"true_votes\":" << votes;
    return write(s.str());
  }
  void cta_barrier_result(unsigned long long arrival, unsigned long long mask, unsigned long long result) {
    if (parent_) write("\"type\":\"cta_barrier_result\",\"issue\":" + std::to_string(parent_) +
        ",\"arrival\":" + std::to_string(arrival) + ",\"execution_mask\":" + std::to_string(mask) +
        ",\"result\":" + std::to_string(result));
  }
  unsigned long long cta_exit(unsigned long long launch, const std::string& kernel,
      const std::array<unsigned, 3>& block, unsigned warp, unsigned long long mask, bool implicit = false) {
    std::ostringstream s;
    s << "\"type\":\"cta_exit\",\"issue\":" << (implicit ? 0 : parent_) << ",\"launch\":" << launch
      << ",\"kernel\":" << quoted(kernel) << ",\"block\":[" << block[0] << ',' << block[1] << ',' << block[2]
      << "],\"warp\":" << warp << ",\"execution_mask\":" << mask;
    return write(s.str());
  }
  void cta_barrier_release(unsigned long long launch, const std::string& kernel,
      const std::array<unsigned, 3>& block, unsigned id, unsigned long long phase,
      unsigned count, unsigned reduction, unsigned long long participants, unsigned long long waiters,
      const std::vector<unsigned long long>& arrivals, const std::vector<unsigned long long>& exits,
      unsigned long long result) {
    std::ostringstream s;
    s << "\"type\":\"cta_barrier_release\",\"launch\":" << launch << ",\"kernel\":" << quoted(kernel)
      << ",\"block\":[" << block[0] << ',' << block[1] << ',' << block[2] << "],\"barrier\":" << id
      << ",\"phase\":" << phase << ",\"count\":" << count << ",\"reduction\":" << reduction
      << ",\"participants_mask\":" << participants << ",\"waiters_mask\":" << waiters
      << ",\"result\":" << result;
    auto references = [&](const char* key, const std::vector<unsigned long long>& values) {
      s << ",\"" << key << "\":[";
      for (size_t i = 0; i < values.size(); ++i) { if (i) s << ','; s << values[i]; }
      s << ']';
    };
    references("arrivals", arrivals); references("exits", exits);
    write(s.str());
  }
  class LaunchScope {
    ExecutionEvents& stream_;
    unsigned long long launch_;
    int exceptions_;
   public:
    LaunchScope(ExecutionEvents& stream, unsigned long long launch, const std::string& kernel,
                const std::array<unsigned, 3>& grid, const std::array<unsigned, 3>& block,
                unsigned regs, unsigned peak, unsigned static_shared, unsigned dynamic_shared)
        : stream_(stream), launch_(launch), exceptions_(std::uncaught_exceptions()) {
      if (!stream.enabled()) return;
      std::ostringstream s;
      s << "\"type\":\"launch\",\"launch\":" << launch << ",\"kernel\":" << quoted(kernel)
        << ",\"grid\":[" << grid[0] << ',' << grid[1] << ',' << grid[2]
        << "],\"block\":[" << block[0] << ',' << block[1] << ',' << block[2]
        << "],\"estimated_registers_per_thread\":" << regs << ",\"estimated_peak_live\":" << peak
        << ",\"register_allocation_provenance\":\"PTX liveness estimate, not NVIDIA ptxas\""
        << ",\"static_shared_bytes\":" << static_shared << ",\"dynamic_shared_bytes\":" << dynamic_shared;
      stream.write(s.str());
    }
    ~LaunchScope() noexcept(false) {
      if (!stream_.enabled()) return;
      bool fault = std::uncaught_exceptions() > exceptions_;
      try {
        stream_.write("\"type\":\"launch_end\",\"launch\":" + std::to_string(launch_) +
                      ",\"status\":\"" + (fault ? "fault" : "completed") + "\"");
      } catch (...) {
        if (!fault) throw;
        std::fputs("tsim: failed to write execution events while reporting a launch fault\n", stderr);
      }
    }
    LaunchScope(const LaunchScope&) = delete;
  };
  class Scope {
    ExecutionEvents& stream_;
    unsigned long long previous_, issue_;
    int exceptions_;
   public:
    Scope(ExecutionEvents& stream, unsigned long long issue)
        : stream_(stream), previous_(parent_), issue_(issue), exceptions_(std::uncaught_exceptions()) { parent_ = issue; }
    ~Scope() noexcept(false) {
      parent_ = previous_;
      if (!issue_) return;
      const bool fault = std::uncaught_exceptions() > exceptions_;
      try {
        stream_.write("\"type\":\"issue_end\",\"issue\":" + std::to_string(issue_) +
                      ",\"status\":\"" + (fault ? "fault" : "completed") + "\"");
      } catch (...) {
        if (!fault) throw;
        std::fputs("tsim: failed to write execution events while reporting a fault\n", stderr);
      }
    }
    Scope(const Scope&) = delete;
  };
};
inline ExecutionEvents& execution_events() { static ExecutionEvents stream; return stream; }
}
