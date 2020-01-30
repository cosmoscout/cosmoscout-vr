////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "logger.hpp"

#include <VistaBase/VistaStreamUtils.h>
#include <sstream>

namespace cs::utils::logger {

template <spdlog::level::level_enum level>
class SpdlogBuffer : public std::stringbuf {
 public:
  SpdlogBuffer(std::shared_ptr<spdlog::logger> const& logger)
      : mLogger(logger) {
  }

 private:
  virtual int sync() override {
    mLogger->log(level, "{}", str());
    return 0;
  }

  int overflow(int ch) override {
    return callback_(&ch, 1, user_data_); // returns the number of characters successfully written.
  }

  std::shared_ptr<spdlog::logger> mLogger;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

void init() {
  spdlog::set_default_logger(createLogger("cs-utils"));

  static auto vistaLogger = createLogger("vista");

  static SpdlogBuffer<spdlog::level::debug> debugBuffer(vistaLogger);
  static std::ostream                       debugStream(&debugBuffer);
  vstr::SetDebugStream(&debugStream);

  static SpdlogBuffer<spdlog::level::info> infoBuffer(vistaLogger);
  static std::ostream                      infoStream(&infoBuffer);
  vstr::SetOutStream(&infoStream);

  static SpdlogBuffer<spdlog::level::warn> warnBuffer(vistaLogger);
  static std::ostream                      warnStream(&warnBuffer);
  vstr::SetWarnStream(&warnStream);

  static SpdlogBuffer<spdlog::level::err> errBuffer(vistaLogger);
  static std::ostream                     errStream(&errBuffer);
  vstr::SetErrStream(&errStream);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::shared_ptr<spdlog::logger> createLogger(std::string const& name) {
  static auto coutSink = std::make_shared<spdlog::sinks::basic_file_sink_mt>("cosmoscout.log");
  static auto fileSink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();

  // We create a colored console logger which can be used from multiple threads. We may consider
  // logging to files in the future.
  std::vector<spdlog::sink_ptr> sinks = {coutSink, fileSink};
  auto logger = std::make_shared<spdlog::logger>(name, sinks.begin(), sinks.end());

  // TODO: Make log level configurable.
  logger->set_level(spdlog::level::trace);

  // See https://github.com/gabime/spdlog/wiki/3.-Custom-formatting for formatting options.
  logger->set_pattern("%^[%L] %-17!n:%$ %v");

  return logger;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::utils::logger
