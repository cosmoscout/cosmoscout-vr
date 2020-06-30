////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_FLY_TO_LOCATIONS_PLUGIN_HPP
#define CSP_FLY_TO_LOCATIONS_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-core/Settings.hpp"

#include <map>
#include <optional>
#include <string>
#include <vector>

namespace csp::flytolocations {

/// This is a very simple plugin with no configuration options. It just shows the bookmarks in the
/// sidebar. It will only show bookmarks which have an associated location. Bookmarks which have an
/// icon and only a center & frame will be shown in a grid of buttons. Bookmarks which have an
/// additional position will be shown in a list when a body with the same center name is currently
/// active.
class Plugin : public cs::core::PluginBase {
 public:
  void init() override;
  void deInit() override;

 private:
  void onAddBookmark(uint32_t bookmarkID, cs::core::Settings::Bookmark const& bookmark);
  int  mActiveBodyConnection        = -1;
  int  mOnBookmarkAddedConnection   = -1;
  int  mOnBookmarkRemovedConnection = -1;
};

} // namespace csp::flytolocations

#endif // CSP_FLY_TO_LOCATIONS_PLUGIN_HPP
