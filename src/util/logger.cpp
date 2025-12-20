#include "util/logger.hpp"

namespace logger {
 // static logger
 static std::shared_ptr<spdlog::logger> logger;
 
  void Logger::InitLogger() {
    static auto tmp_logger = spdlog::stdout_color_st("CometLogger");
    tmp_logger->set_pattern("[%L] %v");
    logger = tmp_logger;
  }

  void Logger::log_message(LogLevel level, fmt::string_view format, fmt::format_args args) {
    logger->log(static_cast<spdlog::level::level_enum>(level), fmt::vformat(format, args));
  }

  void Logger::set_level(LogLevel level) {
    logger->set_level(static_cast<spdlog::level::level_enum>(level));
  }

  // void Logger::add_file_sink(std::string_view file_name) {
  //     auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(std::string(file_name), true);
  //     sinks.push_back(file_sink);
  // }

  // void Logger::add_rotating_file_sink(std::string_view file_name, size_t max_size, size_t max_files) {
  //     auto rotating_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(std::string(file_name), max_size, max_files);
  //     sinks.push_back(rotating_sink);
  // }

  // void Logger::add_sink(spdlog::sink_ptr sink) {
  //     sinks.push_back(sink);
  // }


}
