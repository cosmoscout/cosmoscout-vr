////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_UTILS_LOGGER_HPP
#define CS_UTILS_LOGGER_HPP

#include "cs_utils_export.hpp"

#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__)
#define CS_ALWAYS_INLINE inline __attribute__((__always_inline__))
#elif defined(_MSC_VER)
#define CS_ALWAYS_INLINE __forceinline
#else
#define CS_ALWAYS_INLINE inline
#endif

/// CosmoScout VR uses spdlog for logging. You can print messages simply with code like this:
/// spdlog::info("The awnser is {}!", 42);
/// spdlog uses the fmt library for formatting. You can have a look at their READMEs for examples:
///   https://github.com/gabime/spdlog
///   https://github.com/fmtlib/fmt
/// For each library / plugin of CosmoScout VR a separate default logger is created. This way, the
/// name of the library / plugin can be automatically prepended to the message.

/// Here are the available log levels of spdlog with some hints on when you should use them.
/// spdlog::critical(...)
///
/// spdlog::error(...)
///   Use this for errors which cannot be recovered at runtime abd which likely lead to a crash of
///   the application.
///
/// spdlog::warn(...)
///   Use this for issues which will not lead to a crash but which may limit functionality in some
///   way.
///
/// spdlog::info(...)
///   Use this to inform the user on progress of operations or other events.
///
/// spdlog::debug(...)
///   Use this for information which is meant for the application developper and not for the user.
///
/// spdlog::trace(...)
///   You may use this for temporay debugging of code flow. These should not stay in the code.

/// Here are some guidelines for well-formatted log messages.
/// * Start with capital letter.
/// * No leading "Warning:" or "WARN:" or similar.
/// * End trace, debug and info messages with a dot; warning, error and critical messages with an
///   exclamation mark.
/// * Use ... at the end of your message only if there will be another message indicating that the
///   action has been completed.
///   An example would be:
///     [I][csp-stars] Loading 112236 Stars...
///     [I][csp-stars] Loading done.
/// * Standard form: is <event>: <reason>. <hint>
///   An example would be:
///     [I][csp-stars] Failed to open Tycho catalogue file: File "../main.dat" does not exits! Did
///                    you download all required datasets?
namespace cs::utils::logger {

/// This creates the default logger for "cs-utils" and is called at startup by the main() method.
CS_UTILS_EXPORT void init();

CS_UTILS_EXPORT spdlog::sink_ptr getCoutSink();
CS_UTILS_EXPORT spdlog::sink_ptr getFileSink();

/// Call this method once from your plugin in order to setup the default logger. The given name will
/// be shown together with the log level in each message.
CS_ALWAYS_INLINE void init(std::string const& name) {

  // We create a colored console logger which can be used from multiple threads. We may consider
  // logging to files in the future.
  std::vector<spdlog::sink_ptr> sinks = {getCoutSink(), getFileSink()};
  auto logger =
      std::make_shared<spdlog::logger>(name, sinks.begin(), sinks.end());

  // TODO: Make log level configurable.
  logger->set_level(spdlog::level::trace);

  // See https://github.com/gabime/spdlog/wiki/3.-Custom-formatting for formatting options.
  logger->set_pattern("%^[%L] %-17!n:%$ %v");

  // Since spdlog has a default logger for each shared library, this method "inline". This way it
  // will setup the logger for your plugin when it's called from your code.
  spdlog::set_default_logger(logger);
}

} // namespace cs::utils::logger

#endif // CS_UTILS_LOGGER_HPP
