////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_MEASUREMENT_TOOLS_ELLIPSE_HPP
#define CSP_MEASUREMENT_TOOLS_ELLIPSE_HPP

#include "FlagTool.hpp"

#include <array>

namespace csp::measurementtools {

/// The ellipse tool uses three points on the surface to draw an ellipse. A center point and two
/// points through which the edge has to go through.
class EllipseTool : public IVistaOpenGLDraw, public cs::core::tools::Tool {
 public:
  /// The ellipse and all handels are drawn with this color.
  cs::utils::Property<glm::vec3> pColor = glm::vec3(0.75, 0.75, 1.0);

  EllipseTool(std::shared_ptr<cs::core::InputManager> const& pInputManager,
      std::shared_ptr<cs::core::SolarSystem> const&          pSolarSystem,
      std::shared_ptr<cs::core::Settings> const&             settings,
      std::shared_ptr<cs::core::TimeControl> const& pTimeControl, std::string const& sCenter,
      std::string const& sFrame);

  EllipseTool(EllipseTool const& other) = delete;
  EllipseTool(EllipseTool&& other)      = delete;

  EllipseTool& operator=(EllipseTool const& other) = delete;
  EllipseTool& operator=(EllipseTool&& other) = delete;

  ~EllipseTool() override;

  /// Gets or sets the SPICE center name for all three handles.
  void               setCenterName(std::string const& name);
  std::string const& getCenterName() const;

  /// Gets or sets the SPICE frame name for all three handles.
  void               setFrameName(std::string const& name);
  std::string const& getFrameName() const;

  FlagTool const&              getCenterHandle() const;
  cs::core::tools::Mark const& getFirstHandle() const;
  cs::core::tools::Mark const& getSecondHandle() const;
  FlagTool&                    getCenterHandle();
  cs::core::tools::Mark&       getFirstHandle();
  cs::core::tools::Mark&       getSecondHandle();

  /// Called from Tools class.
  void update() override;

  /// Inherited from IVistaOpenGLDraw.
  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

  void setNumSamples(int const& numSamples);

 private:
  void calculateVertices();

  std::shared_ptr<cs::core::SolarSystem> mSolarSystem;
  std::shared_ptr<cs::core::Settings>    mSettings;
  std::shared_ptr<cs::core::TimeControl> mTimeControl;

  std::shared_ptr<cs::scene::CelestialAnchorNode> mAnchor;

  bool mVerticesDirty = false;
  bool mFirstUpdate   = true;

  FlagTool                                              mCenterHandle;
  std::array<glm::dvec3, 2>                             mAxes;
  std::array<std::unique_ptr<cs::core::tools::Mark>, 2> mHandles;
  std::array<int, 2>                                    mHandleConnections{};

  std::unique_ptr<VistaOpenGLNode> mOpenGLNode;

  VistaVertexArrayObject mVAO;
  VistaBufferObject      mVBO;
  VistaGLSLShader        mShader;

  struct {
    uint32_t modelViewMatrix  = 0;
    uint32_t projectionMatrix = 0;
    uint32_t color            = 0;
    uint32_t farClip          = 0;
  } mUniforms;

  int mScaleConnection = -1;
  int mNumSamples      = 360;

  static const char* SHADER_VERT;
  static const char* SHADER_FRAG;
};
} // namespace csp::measurementtools
#endif // CSP_MEASUREMENT_TOOLS_ELLIPSE_HPP
