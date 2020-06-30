////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_UTILS_LOGGER_HPP
#define CS_UTILS_LOGGER_HPP

#include "cs_utils_export.hpp"

#include "Signal.hpp"

#include <spdlog/spdlog.h>

/// CosmoScout VR uses spdlog for logging. You can print messages simply with code like this:
/// logger().info("The awnser is {}!", 42);
/// spdlog uses the fmt library for formatting. You can have a look at their READMEs for examples:
///   https://github.com/gabime/spdlog
///   https://github.com/fmtlib/fmt
/// For each library / plugin of CosmoScout VR, the logger() method returns a separate default
/// logger. This way, the name of the library / plugin can be automatically prepended to the
/// message.

/// Here are the available log levels of spdlog with some hints on when you should use them.
/// logger().critical(...)
///   Use this for errors which certainly lead to a crash and which also may have corrupted
///   something for the user.
///
/// logger().error(...)
///   Use this for errors which cannot be recovered at runtime and which likely lead to a crash of
///   the application.
///
/// logger().warn(...)
///   Use this for issues which will not lead to a crash but which may limit functionality in some
///   way.
///
/// logger().info(...)
///   Use this to inform the user on progress of operations or other events.
///
/// logger().debug(...)
///   Use this for information which is meant for the application developer and not for the user.
///
/// logger().trace(...)
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
namespace cs::utils {

/// This creates the default singleton logger for "cs-utils" when called for the first time and
/// returns it.
CS_UTILS_EXPORT spdlog::logger& logger();

/// This creates the default logger for vista and is called at startup by the main() method.
CS_UTILS_EXPORT void initVistaLogger();

/// This signal is emitted whenever a message is logged with spdlog. The first argument is the
/// logger's name, the second the log level, the last argument is the message.
CS_UTILS_EXPORT Signal<std::string, spdlog::level::level_enum, std::string> const& onLogMessage();

/// Call this method once from your plugin in order to create a new logger. The given name will
/// be shown together with the log level in each message. The logger will print to the console and
/// store it's messages in a file called cosmoscout.log. If you want to, you could store the
/// returned logger in function static variable in order to create a singleton.
CS_UTILS_EXPORT std::unique_ptr<spdlog::logger> createLogger(std::string const& name);

/// Adjust the log level for each sink seperately.
CS_UTILS_EXPORT spdlog::sink_ptr getLoggerSignalSink();
CS_UTILS_EXPORT spdlog::sink_ptr getLoggerCoutSink();
CS_UTILS_EXPORT spdlog::sink_ptr getLoggerFileSink();

} // namespace cs::utils

#endif // CS_UTILS_LOGGER_HPP
