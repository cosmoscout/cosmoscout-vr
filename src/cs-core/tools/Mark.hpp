////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_CORE_TOOLS_MARK_HPP
#define CS_CORE_TOOLS_MARK_HPP

#include "Tool.hpp"

#include <VistaBase/VistaColor.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaVertexArrayObject.h>

#include <glm/glm.hpp>
#include <memory>

namespace cs::scene {
class CelestialAnchorNode;
} // namespace cs::scene

class VistaBufferObject;
class VistaOpenGLNode;
class VistaVertexArrayObject;

namespace cs::core {
class TimeControl;
class SolarSystem;
class InputManager;
class GuiManager;
class GraphicsEngine;

namespace tools {

/// A mark is a single point on the surface. It is selectable and draggable.
class CS_CORE_EXPORT Mark : public IVistaOpenGLDraw, public Tool {
 public:
  /// Observable properties to get updates on state changes.
  cs::utils::Property<glm::dvec2> pLngLat   = glm::dvec2(0.0);
  cs::utils::Property<bool>       pHovered  = false;
  cs::utils::Property<bool>       pSelected = false;
  cs::utils::Property<bool>       pActive   = false;

  Mark(std::shared_ptr<InputManager> const&  pInputManager,
      std::shared_ptr<SolarSystem> const&    pSolarSystem,
      std::shared_ptr<GraphicsEngine> const& graphicsEngine,
      std::shared_ptr<GuiManager> const&     pGuiManager,
      std::shared_ptr<TimeControl> const& pTimeControl, std::string const& sCenter,
      std::string const& sFrame);

  Mark(Mark const& other);

  virtual ~Mark();

  std::shared_ptr<cs::scene::CelestialAnchorNode> const& getAnchor() const;
  std::shared_ptr<cs::scene::CelestialAnchorNode>&       getAnchor();

  /// Called from Tools class.
  void update() override;

  /// Inherited from IVistaOpenGLDraw.
  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

 protected:
  std::shared_ptr<InputManager>   mInputManager;
  std::shared_ptr<SolarSystem>    mSolarSystem;
  std::shared_ptr<GraphicsEngine> mGraphicsEngine;
  std::shared_ptr<GuiManager>     mGuiManager;
  std::shared_ptr<TimeControl>    mTimeControl;

  std::shared_ptr<cs::scene::CelestialAnchorNode> mAnchor = nullptr;
  VistaOpenGLNode*                                mParent = nullptr;

  double mOriginalDistance = -1.0;

 private:
  void initData(std::string const& sCenter, std::string const& sFrame);

  std::unique_ptr<VistaVertexArrayObject> mVAO;
  std::unique_ptr<VistaBufferObject>      mVBO;
  std::unique_ptr<VistaBufferObject>      mIBO;
  std::unique_ptr<VistaGLSLShader>        mShader;

  size_t mIndexCount;

  int mSelfLngLatConnection = -1, mHoveredNodeConnection = -1, mSelectedNodeConnection = -1,
      mButtonsConnection = -1, mHoveredPlanetConnection = -1, mHeightScaleConnection = -1;

  static const std::string SHADER_VERT;
  static const std::string SHADER_FRAG;
};

} // namespace tools
} // namespace cs::core

#endif // CS_CORE_TOOLS_MARK_HPP
