////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_MEASUREMENT_TOOLS_ELLIPSE_HPP
#define CSP_MEASUREMENT_TOOLS_ELLIPSE_HPP

#include "FlagTool.hpp"

#include <array>

namespace csp::measurementtools {

/// The ellipse tool uses three points on the surface to draw an ellipse. A center point and two
/// points through which the edge has to go through.
class EllipseTool : public IVistaOpenGLDraw, public csl::tools::Tool {
 public:
  /// The ellipse and all handels are drawn with this color.
  cs::utils::Property<glm::vec3> pColor = glm::vec3(0.75, 0.75, 1.0);

  EllipseTool(std::shared_ptr<cs::core::InputManager> pInputManager,
      std::shared_ptr<cs::core::SolarSystem>          pSolarSystem,
      std::shared_ptr<cs::core::Settings> settings, std::string objectName);

  EllipseTool(EllipseTool const& other) = delete;
  EllipseTool(EllipseTool&& other)      = delete;

  EllipseTool& operator=(EllipseTool const& other) = delete;
  EllipseTool& operator=(EllipseTool&& other) = delete;

  ~EllipseTool() override;

  // Assigns all points to a new celestial object.
  void setObjectName(std::string name) override;

  FlagTool const&         getCenterHandle() const;
  csl::tools::Mark const& getFirstHandle() const;
  csl::tools::Mark const& getSecondHandle() const;
  FlagTool&               getCenterHandle();
  csl::tools::Mark&       getFirstHandle();
  csl::tools::Mark&       getSecondHandle();

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

  std::unique_ptr<VistaTransformNode> mAnchor;

  bool mVerticesDirty = false;
  bool mFirstUpdate   = true;

  FlagTool                                         mCenterHandle;
  std::array<glm::dvec3, 2>                        mAxes;
  std::array<std::unique_ptr<csl::tools::Mark>, 2> mHandles;
  std::array<int, 2>                               mHandleConnections{};

  std::unique_ptr<VistaOpenGLNode> mOpenGLNode;

  VistaVertexArrayObject mVAO;
  VistaBufferObject      mVBO;
  VistaGLSLShader        mShader;

  struct {
    uint32_t modelViewMatrix  = 0;
    uint32_t projectionMatrix = 0;
    uint32_t color            = 0;
  } mUniforms;

  int mScaleConnection = -1;
  int mNumSamples      = 360;

  static const char* SHADER_VERT;
  static const char* SHADER_FRAG;
};
} // namespace csp::measurementtools
#endif // CSP_MEASUREMENT_TOOLS_ELLIPSE_HPP
