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

/// CosmoScout VR uses spdlog for logging. You can print messages simply with code like this:
/// spdlog::info("The awnser is {}!", 42);
/// spdlog uses the fmt library for formatting. You can have a look at their READMEs for examples:
///   https://github.com/gabime/spdlog
///   https://github.com/fmtlib/fmt
/// For each library / plugin of CosmoScout VR a separate default logger is created. This way, the
/// name of the library / plugin can be automatically prepended to the message.

/// Here are the available log levels of spdlog with some hints on when you should use them.
/// spdlog::critical(...)
///   Use this for errors which certainly lead to a crash and which also may have corrupted
///   something for the user.
///
/// spdlog::error(...)
///   Use this for errors which cannot be recovered at runtime and which likely lead to a crash of
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
///   Use this for information which is meant for the application developer and not for the user.
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

/// Call this method once from your plugin in order to create a new logger. The given name will
/// be shown together with the log level in each message. The logger will print to the console and
/// store it's messages in a file called cosmoscout.log. You need to call
/// spdlog::set_default_logger(logger); in order to be able to use the global spdlog::info() etc.
/// methods.
CS_UTILS_EXPORT std::shared_ptr<spdlog::logger> createLogger(std::string const& name);

} // namespace cs::utils::logger

#endif // CS_UTILS_LOGGER_HPP
