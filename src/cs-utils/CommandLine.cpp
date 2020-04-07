////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "CommandLine.hpp"

#include "utils.hpp"

#include <algorithm>
#include <iomanip>
#include <sstream>

namespace cs::utils {

////////////////////////////////////////////////////////////////////////////////////////////////////

CommandLine::CommandLine(std::string description)
    : mDescription(std::move(description)) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void CommandLine::addArgument(
    std::vector<std::string> const& flags, Value value, std::string const& help) {

  mArguments.emplace_back(Argument{flags, value, help});
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void CommandLine::printHelp(std::ostream& os) const {

  // Print the general description.
  os << mDescription << std::endl;

  // Find the argument with the longest combined flag length (in order to align the help messages).
  uint32_t maxFlagLength = 0;

  for (auto const& argument : mArguments) {
    uint32_t flagLength = 0;
    for (auto const& flag : argument.mFlags) {
      // Plus comma and space.
      flagLength += static_cast<uint32_t>(flag.size()) + 2;
    }

    maxFlagLength = std::max(maxFlagLength, flagLength);
  }

  // Now print each argument.
  for (auto const& argument : mArguments) {

    std::string flags;
    for (auto const& flag : argument.mFlags) {
      flags += flag + ", ";
    }

    // Remove last comma and space and add padding according to the longest flags in order to align
    // the help messages.
    std::stringstream sstr;
    sstr << std::left << std::setw(maxFlagLength) << flags.substr(0, flags.size() - 2);

    // Print the help for each argument. This is a bit more involved since we do line wrapping for
    // long descriptions.
    size_t currentSpacePos  = 0;
    size_t currentLineWidth = 0;
    while (currentSpacePos != std::string::npos) {
      size_t nextSpacePos = argument.mHelp.find_first_of(' ', currentSpacePos + 1);
      sstr << argument.mHelp.substr(currentSpacePos, nextSpacePos - currentSpacePos);
      currentLineWidth += nextSpacePos - currentSpacePos;
      currentSpacePos = nextSpacePos;

      const int32_t MAX_LINE_WIDTH = 60;
      if (currentLineWidth > MAX_LINE_WIDTH) {
        os << sstr.str() << std::endl;
        sstr = std::stringstream();
        sstr << std::left << std::setw(maxFlagLength - 1) << " ";
        currentLineWidth = 0;
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void CommandLine::parse(std::vector<std::string> args) const {

  // Skip the first argument (name of the program).
  size_t i = 0;
  while (i < args.size()) {

    // We assume that the entire argument is an argument flag.
    std::string flag(args[i]);
    std::string value;
    bool        valueIsSeperate = false;

    // If there is an '=' in the flag, the part after the '=' is actually the value.
    size_t equalPos = flag.find('=');
    if (equalPos != std::string::npos) {
      value = flag.substr(equalPos + 1);
      flag  = flag.substr(0, equalPos);
    }
    // Else the following argument is the value.
    else if (i + 1 < args.size()) {
      value           = args[i + 1];
      valueIsSeperate = true;
    }

    // Search for an argument with the provided flag.
    bool foundArgument = false;

    for (auto const& argument : mArguments) {
      if (utils::contains(argument.mFlags, flag)) {
        foundArgument = true;

        // In the case of booleans, there must not be a value present. So if the value is neither
        // 'true' nor 'false' it is considered to be the next argument.
        if (std::holds_alternative<bool*>(argument.mValue)) {
          if (!value.empty() && value != "true" && value != "false") {
            valueIsSeperate = false;
          }
          *std::get<bool*>(argument.mValue) = (value != "false");
        }
        // In all other cases there must be a value.
        else if (value.empty()) {
          throw std::runtime_error(
              "Failed to parse command line arguments: Missing value for argument \"" + flag +
              "\"!");
        }
        // For a std::string, we take the entire value.
        else if (std::holds_alternative<std::string*>(argument.mValue)) {
          *std::get<std::string*>(argument.mValue) = value;
        }
        // In all other cases we use a std::stringstream to convert the value.
        else {
          std::visit(
              [&value](auto&& arg) {
                std::stringstream sstr(value);
                sstr >> *arg;
              },
              argument.mValue);
        }

        break;
      }
    }

    if (foundArgument && valueIsSeperate) {
      ++i;
    }

    ++i;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::utils
