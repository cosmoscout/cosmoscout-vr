////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Tool.hpp"

namespace cs::core::tools {

Tool::Tool(std::string objectName)
    : mObjectName(std::move(objectName)) {
}

void Tool::update() {
}

void Tool::setObjectName(std::string name) {
  mObjectName = std::move(name);
}

std::string const& Tool::getObjectName() const {
  return mObjectName;
}

} // namespace cs::core::tools
