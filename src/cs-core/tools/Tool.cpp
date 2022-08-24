////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Tool.hpp"

namespace cs::core::tools {

Tool::Tool(std::string objectName)
    : mObjectName(std::move(objectName)) {
}

void Tool::update() {
}

void Tool::setObjectName(std::string const& name) {
  mObjectName = name;
}

std::string const& Tool::getObjectName() const {
  return mObjectName;
}

} // namespace cs::core::tools
