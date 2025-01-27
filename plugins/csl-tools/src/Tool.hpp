////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_TOOLS_TOOL_HPP
#define CSL_TOOLS_TOOL_HPP

#include "../../../src/cs-utils/Property.hpp"
#include "csl_tools_export.hpp"

namespace csl::tools {

/// This is the base interface for all tools. Tools are meant to be things which are created by the
/// user at runtime somewhere in the universe in order to measure something. Usually they are
/// placed on planetary surfaces.
/// If you instantiate derived classes somewhere in your plugin code, you should store them in a
/// std::list<std::shared_ptr<csl::tools::Tool>> mTools;
/// Then, once each frame you should check the Tool's pShouldDelete property and remove it from the
/// list if necessary.
///
/// for (auto it = mTools.begin(); it != mTools.end();) {
///   if ((*it)->pShouldDelete.get()) {
///     it = mTools.erase(it);
///   } else {
///     (*it)->update();
///     ++it;
///   }
/// }
class CSL_TOOLS_EXPORT Tool {
 public:
  /// If set to true the tool will be deleted in the next update cycle.
  cs::utils::Property<bool> pShouldDelete = false;

  Tool() = default;
  explicit Tool(std::string objectName);

  Tool(Tool const& other)     = default;
  Tool(Tool&& other) noexcept = default;

  Tool& operator=(Tool const& other)     = default;
  Tool& operator=(Tool&& other) noexcept = default;

  virtual ~Tool() = default;

  virtual void update();

  virtual void       setObjectName(std::string name);
  std::string const& getObjectName() const;

 private:
  std::string mObjectName;
};

} // namespace csl::tools

#endif // CSL_TOOLS_TOOL_HPP
