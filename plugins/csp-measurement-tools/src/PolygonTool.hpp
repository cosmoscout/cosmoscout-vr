////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_MEASUREMENT_TOOLS_POLYGONTOOL_HPP
#define CSP_MEASUREMENT_TOOLS_POLYGONTOOL_HPP

#include "../../../src/cs-core/tools/MultiPointTool.hpp"
#include "Plugin.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>

#include <glm/glm.hpp>
#include <vector>

#include "voronoi/VoronoiGenerator.hpp"

namespace cs::scene {
class CelestialAnchorNode;
}

namespace cs::gui {
class GuiItem;
class WorldSpaceGuiArea;
} // namespace cs::gui

class VistaTransformNode;

namespace csp::measurementtools {

/// Measures the area and volume of an arbitrary polygon on surface with a Delaunay-mesh. It
/// displays the bounding box of the selected polygon, which can be copied for cache generator.
class PolygonTool : public IVistaOpenGLDraw, public cs::core::tools::MultiPointTool {
 public:
  /// This text is shown on the ui and can be edited by the user.
  cs::utils::Property<std::string> pText = std::string("Polygon");

  /// Displays the generated mesh for area and volume computations.
  cs::utils::Property<bool> pShowMesh = false;

  PolygonTool(std::shared_ptr<cs::core::InputManager> const& pInputManager,
      std::shared_ptr<cs::core::SolarSystem> const&          pSolarSystem,
      std::shared_ptr<cs::core::Settings> const&             settings,
      std::shared_ptr<cs::core::TimeControl> const& pTimeControl, std::string const& sCenter,
      std::string const& sFrame);

  PolygonTool(PolygonTool const& other) = delete;
  PolygonTool(PolygonTool&& other)      = delete;

  PolygonTool& operator=(PolygonTool const& other) = delete;
  PolygonTool& operator=(PolygonTool&& other) = delete;

  ~PolygonTool() override;

  // Gets or sets the SPICE center name for all points.
  void setCenterName(std::string const& name) override;

  /// Gets or sets the SPICE frame name for all points.
  void setFrameName(std::string const& name) override;

  /// Called from Tools class
  void update() override;

  /// Inherited from IVistaOpenGLDraw
  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

  void setHeightDiff(float hDiff);
  void setMaxAttempt(uint32_t att);
  void setMaxPoints(uint32_t points);
  void setSleekness(uint32_t degree);

 private:
  void updateLineVertices();
  void updateCalculation();

  /// Returns the interpolated position in cartesian coordinates. The fourth component is
  /// height above the surface
  glm::dvec4 getInterpolatedPosBetweenTwoMarks(cs::core::tools::DeletableMark const& l0,
      cs::core::tools::DeletableMark const& l1, double value);

  /// Finds the intersection point between two sites
  static bool findIntersection(Site const& s1, Site const& s2, Site const& s3, Site const& s4,
      double& intersectionX, double& intersectionY);

  /// Creates a Delaunay-mesh and corrects it to match the original polygon
  /// (especially for concave polygons)
  void createMesh(std::vector<Triangle>& triangles);
  /// Checks sleekness of a triangle from the original Delaunay-mesh and its subtriangles
  /// If a triangle is too sleek, divides it
  /// Returns true if a lot of new points are added
  bool checkSleekness(int count);
  /// Draws the Delaunay-mesh on the planet's surface
  void displayMesh(Edge2 const& edge, double mdist, glm::dvec3 const& e, glm::dvec3 const& n,
      glm::dvec3 const& r, double scale, double& h1, double& h2);
  /// Refines mesh based on edge length and terrain
  void refineMesh(Edge2 const& edge, double mdist, glm::dvec3 const& e, glm::dvec3 const& n,
      glm::dvec3 const& r, int count, double h1, double h2, bool& fine);
  /// Calculates triangle areas and prism volumes
  void calculateAreaAndVolume(std::vector<Triangle> const& triangles, double mdist,
      glm::dvec3 const& e, glm::dvec3 const& n, glm::dvec3 const& r, double& area, double& pvol,
      double& nvol);
  // Checks if point is inside of the polygon or not
  bool checkPoint(glm::dvec2 const& point);

  // These are called by the base class MultiPointTool
  void onPointMoved() override;
  void onPointAdded() override;
  void onPointRemoved(int index) override;

  std::shared_ptr<cs::scene::CelestialAnchorNode> mGuiAnchor;
  std::unique_ptr<cs::gui::WorldSpaceGuiArea>     mGuiArea;
  std::unique_ptr<cs::gui::GuiItem>               mGuiItem;
  std::unique_ptr<VistaTransformNode>             mGuiTransform;
  std::unique_ptr<VistaOpenGLNode>                mGuiNode;
  std::unique_ptr<VistaOpenGLNode>                mParent;

  // For Lines
  VistaVertexArrayObject mVAO;
  VistaBufferObject      mVBO;

  // For Delaunay
  VistaVertexArrayObject mVAO2;
  VistaBufferObject      mVBO2;
  VistaGLSLShader        mShader;

  struct {
    uint32_t modelViewMatrix  = 0;
    uint32_t projectionMatrix = 0;
    uint32_t color            = 0;
    uint32_t farClip          = 0;
  } mUniforms;

  std::vector<glm::dvec3> mSampledPositions;
  size_t                  mIndexCount    = 0;
  bool                    mVerticesDirty = false;

  int mTextConnection  = -1;
  int mScaleConnection = -1;

  // minLng,maxLng,minLat,maxLat
  glm::dvec4 mBoundingBox = glm::dvec4(0.0);

  // For Delaunay-mesh
  std::vector<Site>              mCorners;
  std::vector<std::vector<Site>> mCornersFine;
  std::vector<glm::dvec3>        mTriangulation;
  glm::dvec3                     mNormal      = glm::dvec3(0.0);
  glm::dvec3                     mMiddlePoint = glm::dvec3(0.0);
  size_t                         mIndexCount2 = 0;

  // For triangle fineness
  float    mHeightDiff = 1.002F;
  uint32_t mMaxAttempt = 10;
  uint32_t mMaxPoints  = 1000;
  uint32_t mSleekness  = 15;

  // For volume calculation
  double     mOffset{};
  glm::dvec3 mNormal2      = glm::dvec3(0.0);
  glm::dvec3 mMiddlePoint2 = glm::dvec3(0.0);

  static const int   NUM_SAMPLES;
  static const char* SHADER_VERT;
  static const char* SHADER_FRAG;
};

} // namespace csp::measurementtools

#endif // CSP_MEASUREMENT_TOOLS_POLYGONTOOL_HPP
