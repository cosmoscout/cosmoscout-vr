////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "logger.hpp"

#include <VistaBase/VistaStreamUtils.h>
#include <sstream>

namespace cs::utils::logger {

namespace {

// This class is used to intercept Vista messages.
template <spdlog::level::level_enum level>
class SpdlogBuffer : public std::streambuf {
 public:
  SpdlogBuffer(std::shared_ptr<spdlog::logger> const& logger)
      : mLogger(logger) {
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

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

void init() {
  spdlog::set_default_logger(createLogger("cs-utils"));

  // This logger will be used by vista.
  static auto vistaLogger = createLogger("vista");

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

std::shared_ptr<spdlog::logger> createLogger(std::string const& name) {
  // Setup our sinks. The logger will print to the console and store it's messages in a file called
  // cosmoscout.log.
  static auto fileSink =
      std::make_shared<spdlog::sinks::basic_file_sink_mt>("cosmoscout.log", true);
  static auto coutSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  static std::vector<spdlog::sink_ptr> sinks = {coutSink, fileSink};

  // Append some ... to the name of the logger to make the output more readable.
  std::string paddedName = name + " ";
  while (paddedName.length() < 20) {
    paddedName += ".";
  }
  paddedName.back() = ' ';

  auto logger = std::make_shared<spdlog::logger>(paddedName, sinks.begin(), sinks.end());

  // TODO: Make log level configurable.
  logger->set_level(spdlog::level::trace);

  // See https://github.com/gabime/spdlog/wiki/3.-Custom-formatting for formatting options.
  logger->set_pattern("%^[%L] %n%$%v");

  return logger;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::utils::logger
