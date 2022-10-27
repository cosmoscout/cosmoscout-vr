////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_NODE_EDITOR_CONNECTION_HPP
#define CSL_NODE_EDITOR_CONNECTION_HPP

#include "csl_node_editor_export.hpp"

#include <any>

namespace csl::nodeeditor {

struct CSL_NODE_EDITOR_EXPORT Connection {
  std::any mValue;
  bool     mDirty;
};

} // namespace csl::nodeeditor

#endif // CSL_NODE_EDITOR_CONNECTION_HPP
