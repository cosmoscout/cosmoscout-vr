////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_NODE_EDITOR_NODE_HPP
#define CSL_NODE_EDITOR_NODE_HPP

#include "csl_node_editor_export.hpp"

namespace csl::nodeeditor {

class CSL_NODE_EDITOR_EXPORT Node {
 public:
  virtual void process();

 private:
};

} // namespace csl::nodeeditor

#endif // CSL_NODE_EDITOR_NODE_HPP
