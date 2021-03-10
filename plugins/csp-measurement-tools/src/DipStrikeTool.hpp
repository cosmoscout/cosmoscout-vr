////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_MEASUREMENT_TOOLS_DIP_STRIKE_HPP
#define CSP_MEASUREMENT_TOOLS_DIP_STRIKE_HPP

#include "../../../src/cs-core/tools/MultiPointTool.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>

#include <glm/glm.hpp>
#include <vector>

namespace cs::scene {
class CelestialAnchorNode;
}

namespace cs::gui {
class GuiItem;
class WorldSpaceGuiArea;
} // namespace cs::gui

class VistaBufferObject;
class VistaGLSLShader;
class VistaOpenGLNode;
class VistaVertexArrayObject;
class VistaTransformNode;

namespace csp::measurementtools {

/// The dip and strike tool is used to measure the steepness and orientation of slopes. It uses a
/// set of points on the surface to generate a plane that has the lowest sum of squared distances to
/// all points.
/// The dip (steepness) is given in degrees from 0° to 90° and the strike (orientation) is also
/// given in degrees, where at 0° the peak is in the east and at 90° the peak is in the north.
class DipStrikeTool : public IVistaOpenGLDraw, public cs::core::tools::MultiPointTool {
 public:
  /// This text is shown on the ui and can be edited by the user.
  cs::utils::Property<std::string> pText    = std::string("Dip & Strike");
  cs::utils::Property<float>       pSize    = 1.5F;
  cs::utils::Property<float>       pOpacity = 0.5F;

  DipStrikeTool(std::shared_ptr<cs::core::InputManager> const& pInputManager,
      std::shared_ptr<cs::core::SolarSystem> const&            pSolarSystem,
      std::shared_ptr<cs::core::Settings> const&               settings,
      std::shared_ptr<cs::core::TimeControl> const& pTimeControl, std::string const& sCenter,
      std::string const& sFrame);

  DipStrikeTool(DipStrikeTool const& other) = delete;
  DipStrikeTool(DipStrikeTool&& other)      = delete;

  DipStrikeTool& operator=(DipStrikeTool const& other) = delete;
  DipStrikeTool& operator=(DipStrikeTool&& other) = delete;

  ~DipStrikeTool() override;

  // Gets or sets the SPICE center name for all points.
  void setCenterName(std::string const& name) override;

  /// Gets or sets the SPICE frame name for all points.
  void setFrameName(std::string const& name) override;

  /// Called from Tools class.
  void update() override;

  /// Inherited from IVistaOpenGLDraw.
  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 private:
  void calculateDipAndStrike();

  /// Returns the interpolated position in cartesian coordinates the fourth component is height
  /// above the surface.
  glm::dvec4 getInterpolatedPosBetweenTwoMarks(cs::core::tools::DeletableMark const& pMark1,
      cs::core::tools::DeletableMark const& pMark2, double value);

  /// These are called by the base class MultiPointTool.
  void onPointMoved() override;
  void onPointAdded() override;
  void onPointRemoved(int index) override;

  std::shared_ptr<cs::scene::CelestialAnchorNode> mGuiAnchor;
  std::shared_ptr<cs::scene::CelestialAnchorNode> mPlaneAnchor;

  std::unique_ptr<cs::gui::WorldSpaceGuiArea> mGuiArea;
  std::unique_ptr<cs::gui::GuiItem>           mGuiItem;
  std::unique_ptr<VistaTransformNode>         mGuiTransform;
  std::unique_ptr<VistaOpenGLNode>            mGuiOpenGLNode;
  std::unique_ptr<VistaOpenGLNode>            mPlaneOpenGLNode;

  VistaVertexArrayObject mVAO;
  VistaBufferObject      mVBO;
  VistaGLSLShader        mShader;

  struct {
    uint32_t modelViewMatrix  = 0;
    uint32_t projectionMatrix = 0;
    uint32_t opacity          = 0;
    uint32_t farClip          = 0;
  } mUniforms;

  bool      mVerticesDirty = false;
  double    mSize{};
  glm::vec3 mNormal = glm::vec3(0.0), mMip = glm::vec3(0.0);
  float     mOffset{};

  int mTextConnection  = -1;
  int mScaleConnection = -1;

  static const int   RESOLUTION;
  static const char* SHADER_VERT;
  static const char* SHADER_FRAG;
};

} // namespace csp::measurementtools

#endif // CSP_MEASUREMENT_TOOLS_DIP_STRIKE_HPP
