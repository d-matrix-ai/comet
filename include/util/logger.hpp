#pragma once

#define FMT_HEADER_ONLY
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include "spdlog/spdlog.h"
#include "spdlog/cfg/env.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/basic_file_sink.h"

#define COMET_LOG(level, ...) \
  ::logger::Logger::log(level, __VA_ARGS__)

#define STAT_LOG(level, ...) \
  ::logger::StatLogger::log(__VA_ARGS__)

namespace logger {
  // levels that map to spdlog
  enum LogLevel {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2, 
    WARN = 3, 
    ERROR = 4, 
    CRITICAL = 5, 
    OFF = 6
  };

  // static function wrapper around a spdlog logger
  class Logger {
    public:
      static void InitLogger(); // to stdout
      
      static void set_level(LogLevel level);
      
      // static void add_file_sink(std::string_view file_name);
      // static void add_rotating_file_sink(std::string_view file_name, size_t max_size, size_t max_files);

      static void add_sink(spdlog::sink_ptr sink);
      
      static void set_pattern(std::string_view pattern="", bool* env_overrode = nullptr);
      
      template <typename... ArgsT>
        static void log(LogLevel level, fmt::format_string<ArgsT...> fmt_str, ArgsT&&... args) {
          log_message(level, fmt_str, fmt::make_format_args(args...));
        }
    private:
      static void log_message(LogLevel level, fmt::string_view format, fmt::format_args args);
      // static std::shared_ptr<spdlog::logger> logger;
      // static std::vector<spdlog::sink_ptr> sinks;
    };

}
