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

#include <boost/filesystem.hpp>
#include <iostream>

namespace cs::utils {

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
  std::vector<std::string> elems;

  std::stringstream ss(s);
  std::string       item;

  while (std::getline(ss, item, delim)) {
    elems.push_back(item);
  }

  return elems;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float getCurrentFarClipDistance() {
  double near, far;
  GetVistaSystem()
      ->GetDisplayManager()
      ->GetCurrentRenderInfo()
      ->m_pViewport->GetProjection()
      ->GetProjectionProperties()
      ->GetClippingRange(near, far);
  return (float)far;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::utils
