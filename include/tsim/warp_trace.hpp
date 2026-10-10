#pragma once
#include <array>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

namespace tsim {
// A pre-issue snapshot from the actual PTX interpreter, including predication.
// Stream to disk with a hard cap, so teaching traces cannot exhaust host RAM.
class WarpTrace {
  std::mutex mutex_;
  FILE* file_ = nullptr;
  unsigned long long count_ = 0, limit_ = 200000;
  unsigned long long next_launch_ = 0;
  static void quoted(FILE* f, const std::string& s) {
    std::fputc('"', f);
    for (unsigned char c : s) {
      if (c == '"' || c == '\\') { std::fputc('\\', f); std::fputc(c, f); }
      else if (c < 32) std::fprintf(f, "\\u%04x", unsigned(c));
      else std::fputc(c, f);
    }
    std::fputc('"', f);
  }
 public:
  WarpTrace() {
    const char* path = std::getenv("TSIM_TRACE");
    if (!path || !*path) return;
    if (const char* cap = std::getenv("TSIM_TRACE_LIMIT")) {
      char* end = nullptr;
      errno = 0;
      auto parsed = std::strtoull(cap, &end, 10);
      if (!errno && end != cap && !*end && cap[0] != '-' && parsed > 0) limit_ = parsed;
    }
    file_ = std::fopen(path, "a");
    if (!file_) throw std::runtime_error("tsim: cannot open warp trace: " + std::string(std::strerror(errno)));
  }
  ~WarpTrace() { if (file_) std::fclose(file_); }
  unsigned long long begin_launch() {
    std::lock_guard<std::mutex> lock(mutex_);
    return ++next_launch_;
  }
  void issue(unsigned long long launch, const std::string& kernel, const std::string& function,
             const std::array<unsigned, 3>& block, unsigned warp,
             size_t pc, size_t line, unsigned long long path_mask,
             unsigned long long execution_mask, size_t paths, const std::string& instruction) {
    if (!file_) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (count_ >= limit_) {
      if (count_ == limit_) {
        std::fprintf(file_, "{\"type\":\"truncated\",\"limit\":%llu}\n", limit_);
        std::fflush(file_);
        ++count_;
      }
      return;
    }
    std::fprintf(file_, "{\"type\":\"issue\",\"sequence\":%llu,\"launch\":%llu,\"kernel\":", count_++, launch);
    quoted(file_, kernel);
    std::fputs(",\"function\":", file_); quoted(file_, function);
    std::fprintf(file_, ",\"block\":[%u,%u,%u],\"warp\":%u,\"pc\":%zu,\"ptx_line\":%zu,\"path_mask\":%llu,\"execution_mask\":%llu,\"paths\":%zu,\"instruction\":",
                 block[0], block[1], block[2], warp, pc, line, path_mask, execution_mask, paths);
    quoted(file_, instruction);
    std::fputs("}\n", file_);
    // Flush each event: preserve the last instruction even when a kernel faults.
    if (std::fflush(file_)) throw std::runtime_error("tsim: failed to write warp trace");
  }
};
inline WarpTrace& warp_trace() { static WarpTrace trace; return trace; }
}
