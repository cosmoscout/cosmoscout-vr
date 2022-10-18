////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_MEASUREMENT_TOOLS_PATH_HPP
#define CSP_MEASUREMENT_TOOLS_PATH_HPP

#include "../../csl-tools/src/MultiPointTool.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>

#include <glm/glm.hpp>
#include <vector>

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

/// The path tool is used to measure the distance and height along a path of lines.
class PathTool : public IVistaOpenGLDraw, public csl::tools::MultiPointTool {
 public:
  /// This text is shown on the ui and can be edited by the user.
  cs::utils::Property<std::string> pText = std::string("Path");

  PathTool(std::shared_ptr<cs::core::InputManager> pInputManager,
      std::shared_ptr<cs::core::SolarSystem>       pSolarSystem,
      std::shared_ptr<cs::core::Settings> settings, std::string objectName);

  PathTool(PathTool const& other) = delete;
  PathTool(PathTool&& other)      = delete;

  PathTool& operator=(PathTool const& other) = delete;
  PathTool& operator=(PathTool&& other) = delete;

  ~PathTool() override;

  /// Called from Tools class.
  void update() override;

  /// Inherited from IVistaOpenGLDraw.
  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

  void setNumSamples(int const& numSamples);

 private:
  void updateLineVertices();

  /// Returns the interpolated position in cartesian coordinates. The fourth component is height
  /// above the surface.
  glm::dvec4 getInterpolatedPosBetweenTwoMarks(csl::tools::DeletableMark const& l0,
      csl::tools::DeletableMark const& l1, double value, double const& scale);

  /// These are called by the base class MultiPointTool.
  void onPointMoved() override;
  void onPointAdded() override;
  void onPointRemoved(int index) override;

  std::unique_ptr<VistaTransformNode>         mGuiAnchor;
  std::unique_ptr<VistaTransformNode>         mGuiTransform;
  std::unique_ptr<VistaOpenGLNode>            mGuiOpenGLNode;
  std::unique_ptr<VistaOpenGLNode>            mPathOpenGLNode;
  std::unique_ptr<cs::gui::WorldSpaceGuiArea> mGuiArea;
  std::unique_ptr<cs::gui::GuiItem>           mGuiItem;

  VistaVertexArrayObject mVAO;
  VistaBufferObject      mVBO;
  VistaGLSLShader        mShader;

  struct {
    uint32_t modelViewMatrix  = 0;
    uint32_t projectionMatrix = 0;
    uint32_t color            = 0;
  } mUniforms;

  std::vector<glm::dvec3> mSampledPositions;
  glm::dvec3              mPosition{};
  size_t                  mIndexCount    = 0;
  bool                    mVerticesDirty = false;

  int mScaleConnection = -1;
  int mTextConnection  = -1;
  int mNumSamples      = 256;

  static const char* SHADER_VERT;
  static const char* SHADER_FRAG;
};

} // namespace csp::measurementtools

#endif // CSP_MEASUREMENT_TOOLS_PATH_HPP
