////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "utils.hpp"

#include <VistaKernel/DisplayManager/VistaDisplayManager.h>
#include <VistaKernel/DisplayManager/VistaProjection.h>
#include <VistaKernel/DisplayManager/VistaViewport.h>
#include <VistaKernel/VistaSystem.h>

#include <array>
#include <iostream>
#include <memory>

namespace cs::utils {

////////////////////////////////////////////////////////////////////////////////////////////////////

bool contains(std::string const& string, std::string const& value) {
  return string.find(value) != std::string::npos;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool startsWith(std::string const& string, std::string const& prefix) {
  return string.find(prefix) == 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool endsWith(std::string const& string, std::string const& postfix) {
  return std::mismatch(postfix.rbegin(), postfix.rend(), string.rbegin()).first == postfix.rend();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void replaceString(
    std::string& sInput, std::string const& sPlaceHolder, std::string const& sNewString) {
  std::size_t pos = 0;

  while ((pos = sInput.find(sPlaceHolder, pos)) != std::string::npos) {
    sInput.replace(pos, sPlaceHolder.length(), sNewString);

    pos += sNewString.length();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
std::string toString(float const& v) {
  std::ostringstream oss;
  oss.precision(std::numeric_limits<float>::max_digits10);
  oss << v;
  return oss.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
std::string toString(double const& v) {
  std::ostringstream oss;
  oss.precision(std::numeric_limits<double>::max_digits10);
  oss << v;
  return oss.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
std::string toString(std::string const& v) {
  std::string tmp(v);

  replaceString(tmp, "\"", "\\\"");
  replaceString(tmp, "\n", "\\n");

  return "\"" + tmp + "\"";
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string toString(char const* v) {
  return toString(std::string(v));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<std::string> splitString(std::string const& s, char delim) {
  size_t start = 0;
  size_t end   = s.find_first_of(delim);

  std::vector<std::string> output;

  while (end <= std::string::npos) {
    output.emplace_back(s.substr(start, end - start));

    if (end == std::string::npos) {
      break;
    }

    start = end + 1;
    end   = s.find_first_of(delim, start);
  }

  return output;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float getCurrentFarClipDistance() {
  double nearPlane{};
  double farPlane{};
  GetVistaSystem()
      ->GetDisplayManager()
      ->GetCurrentRenderInfo()
      ->m_pViewport->GetProjection()
      ->GetProjectionProperties()
      ->GetClippingRange(nearPlane, farPlane);
  return static_cast<float>(farPlane);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#ifdef __linux__
#define CS_POPEN popen
#define CS_CLOSE pclose
#else
#define CS_POPEN _popen
#define CS_CLOSE _pclose
#endif

std::string exec(std::string const& cmd) {
  std::array<char, 128>                      buffer{};
  std::string                                result;
  std::unique_ptr<FILE, decltype(&CS_CLOSE)> pipe(CS_POPEN(cmd.c_str(), "r"), CS_CLOSE);
  if (!pipe) {
    throw std::runtime_error("popen() failed!");
  }
  while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) != nullptr) {
    result += buffer.data();
  }
  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::utils
