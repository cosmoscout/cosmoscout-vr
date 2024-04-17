////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_GUIDED_TOUR_PLUGIN_HPP
#define CSP_GUIDED_TOUR_PLUGIN_HPP

#include "../../../src/cs-core/PluginBase.hpp"

#include <glm/glm.hpp>

#include <list>
#include <string>
#include <vector>

class VistaOpenGLNode;
class VistaTransformNode;

namespace cs::gui {
class WorldSpaceGuiArea;
class GuiItem;
} // namespace cs::gui

namespace csp::guidedtour {
/// This plugin allows to add custom HTML content to a sidebar-tob, to a floating window or to any
/// position in space.

class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {

    struct CheckPointSettings {
      /// The SPICE center and frame names.
      std::string mObject;

      /// The position of the item, elevation is relative to the surface height.
      double mLongitude{};
      double mLatitude{};
      double mElevation{};

      /// Size of the item. The item will scale based on the observer distance.
      double mScale{};

      /// Size of the item in pixels.
      uint32_t mWidth{};
      uint32_t mHeight{};

      /// The actual File path.
      std::string mFile;
    };

    struct TourSettings {

      std::string                     mName;
      std::vector<CheckPointSettings> mCheckpoints;
    };
    /// These items will be placed somewhere on a celestial body.

    std::vector<TourSettings> mTours;
  };

  void init() override;
  void update() override;
  void deInit() override;
  void loadTour(std::string const& tourName);

 private:
  void onLoad();
  void onSave();
  void unload(Settings const& pluginSettings);
  void setTour(std::string const& tourName);
  void loadCheckpoints();

  struct CPItem {
    std::unique_ptr<cs::gui::WorldSpaceGuiArea> mGuiArea;
    std::unique_ptr<cs::gui::GuiItem>           mGuiItem;
    std::unique_ptr<VistaTransformNode>         mAnchor;
    std::unique_ptr<VistaTransformNode>         mTransform;
    std::unique_ptr<VistaOpenGLNode>            mGuiNode;
    double                                      mScale = 1.0;
    glm::dvec3                                  mPosition;
    std::string                                 mObjectName;
  };

  Settings mPluginSettings;

  std::list<CPItem> mCPItems;
  std::string       mCurrentTour = "none";

  int mOnLoadConnection = -1;
  int mOnSaveConnection = -1;
};

} // namespace csp::guidedtour

#endif // CSP_GUIDED_TOUR_PLUGIN_HPP
