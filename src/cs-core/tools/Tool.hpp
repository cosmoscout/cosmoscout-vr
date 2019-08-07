////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_CORE_TOOLS_TOOL_HPP
#define CS_CORE_TOOLS_TOOL_HPP

#include "../../cs-utils/Property.hpp"
#include "cs_core_export.hpp"

namespace cs::core::tools {

/// This is the base interface for all tools.
class CS_CORE_EXPORT Tool {
 public:
  /// If set to true the tool will be deleted in the next update cycle.
  cs::utils::Property<bool> pShouldDelete = false;

  virtual ~Tool() {
  }

  virtual void update() {
  }
};

} // namespace cs::core::tools

#endif // CS_CORE_TOOLS_TOOL_HPP
