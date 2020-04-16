////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "logger.hpp"

#include <VistaBase/VistaStreamUtils.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <sstream>
#include <utility>

namespace cs::utils {

namespace {

// This class is used to intercept Vista messages.
template <spdlog::level::level_enum level>
class SpdlogBuffer : public std::streambuf {
 public:
  explicit SpdlogBuffer(std::shared_ptr<spdlog::logger> logger)
      : mLogger(std::move(logger)) {
  }

 private:
  // Whenever vista prints a '\n', a new log message is emitted.
  int_type overflow(int_type c) override {
    char_type ch = traits_type::to_char_type(c);
    if (ch == '\n') {
      mLogger->log(level, "{}", mLine);
      mLine = "";
    } else {
      mLine += ch;
    }

    return 0;
  }

  std::shared_ptr<spdlog::logger> mLogger;
  std::string                     mLine;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

class SignalSink : public spdlog::sinks::base_sink<std::mutex> {
 public:
  Signal<std::string, spdlog::level::level_enum, std::string> onLogMessage;

 protected:
  void sink_it_(spdlog::details::log_msg const& msg) override {
    onLogMessage.emit(std::string(msg.logger_name.begin(), msg.logger_name.end()), msg.level,
        std::string(msg.payload.begin(), msg.payload.end()));
  }

  void flush_() override {
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

spdlog::logger& logger() {
  static auto logger = createLogger("cs-utils");
  return *logger;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void initVistaLogger() {
  // This logger will be used by vista.
  static std::shared_ptr<spdlog::logger> vistaLogger = createLogger("vista");

  // Assign a custom log stream for vista's debug messages.
  static SpdlogBuffer<spdlog::level::debug> debugBuffer(vistaLogger);
  static std::ostream                       debugStream(&debugBuffer);
  vstr::SetDebugStream(&debugStream);

  // Assign a custom log stream for vista's info messages.
  static SpdlogBuffer<spdlog::level::info> infoBuffer(vistaLogger);
  static std::ostream                      infoStream(&infoBuffer);
  vstr::SetOutStream(&infoStream);

  // Assign a custom log stream for vista's warnings.
  static SpdlogBuffer<spdlog::level::warn> warnBuffer(vistaLogger);
  static std::ostream                      warnStream(&warnBuffer);
  vstr::SetWarnStream(&warnStream);

  // Assign a custom log stream for vista's errors.
  static SpdlogBuffer<spdlog::level::err> errBuffer(vistaLogger);
  static std::ostream                     errStream(&errBuffer);
  vstr::SetErrStream(&errStream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

Signal<std::string, spdlog::level::level_enum, std::string> const& onLogMessage() {
  return std::dynamic_pointer_cast<SignalSink>(getLoggerSignalSink())->onLogMessage;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<spdlog::logger> createLogger(std::string const& name) {
  size_t const prefixLength = 20;

  // Append some ... to the name of the logger to make the output more readable.
  std::string paddedName = name + " ";
  while (paddedName.size() < prefixLength) {
    paddedName += ".";
  }
  paddedName.back() = ' ';

  std::vector<spdlog::sink_ptr> sinks = {
      getLoggerSignalSink(), getLoggerCoutSink(), getLoggerFileSink()};
  auto logger = std::make_unique<spdlog::logger>(paddedName, sinks.begin(), sinks.end());

  // See https://github.com/gabime/spdlog/wiki/3.-Custom-formatting for formatting options.
  logger->set_pattern("%^[%L] %n%$%v"); // NOLINT(clang-analyzer-cplusplus.Move)
  logger->set_level(spdlog::level::trace);

  return logger;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

spdlog::sink_ptr getLoggerSignalSink() {
  static auto sink = std::make_shared<SignalSink>();
  return sink;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

spdlog::sink_ptr getLoggerCoutSink() {
  static auto sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  return sink;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

spdlog::sink_ptr getLoggerFileSink() {
  static auto sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("cosmoscout.log", true);
  return sink;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::utils
