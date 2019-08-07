////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_SCENE_TRAJECTORY_HPP
#define CS_SCENE_TRAJECTORY_HPP

#include "cs_scene_export.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <vector>

class VistaGLSLShader;
class VistaVertexArrayObject;
class VistaBufferObject;

namespace cs::scene {

/// This class is responsible for drawing trajectories. Trajectories are line segments which
/// typically follow an object in space. It is most often used to draw orbit paths.
/// A trajectories trail consists of a list of points in 3D space, where every 3D point is
/// extended by a fourth value, which indicates its age. This is used to fade older
/// points out. The lifetime of every point depends on the maxAge member.
/// The color of every point is also dependent on the lifetime. It is controlled with the members
/// startColor and endColor. A young point will have a color closer to the startColor and an old
/// point, which is close to the maxAge will have a color closer to the endColor.
class CS_SCENE_EXPORT Trajectory : public IVistaOpenGLDraw {
 public:
  Trajectory();
  ~Trajectory() override;

  /// Transforms all points by relativeTransform and uploads them to the GPU. Call this every frame
  /// in order to show the trajectory with observer centric coordinates. dTime determines the
  /// current age of all points.
  void upload(glm::dmat4 const& relativeTransform, double dTime,
      std::vector<glm::dvec4> const& vPoints, glm::dvec3 const& vTip, int startIndex);

  /// The method Do() gets the callback from scene graph during the rendering process.
  /// Renders the trajectory in its current state.
  bool Do() override;

  /// This method should return the bounding box of the openGL object you draw in the method Do().
  bool GetBoundingBox(VistaBoundingBox& bb) override;

  /// The age for which a point is part of the Trajectory.
  double getMaxAge() const;
  void   setMaxAge(double val);

  /// The color which young points are drawn with.
  glm::vec4 const& getStartColor() const;
  void             setStartColor(glm::vec4 const& val);

  /// The color which old points are drawn with.
  glm::vec4 const& getEndColor() const;
  void             setEndColor(glm::vec4 const& val);

  /// The width of the Trajectory line.
  float getWidth() const;
  void  setWidth(float val);

  /// If true, the depth buffer is assumed to contain linear depth values. This significantly
  /// reduces artifacts for large scenes.
  bool getUseLinearDepthBuffer() const;
  void setUseLinearDepthBuffer(bool bEnable);

 private:
  void createShader();

  VistaGLSLShader*        mShader;
  VistaVertexArrayObject* mVAO;
  VistaBufferObject*      mVBO;

  double    mMaxAge;
  glm::vec4 mStartColor;
  glm::vec4 mEndColor;
  float     mWidth;

  bool mShaderDirty          = true;
  bool mUseLinearDepthBuffer = false;

  int mPointCount;

  static const std::string SHADER_VERT;
  static const std::string SHADER_FRAG;
};
} // namespace cs::scene

#endif // CS_SCENE_TRAJECTORY_HPP
