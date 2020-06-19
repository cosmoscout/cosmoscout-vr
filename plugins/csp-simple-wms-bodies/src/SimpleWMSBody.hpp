////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_WMS_SIMPLE_BODIES_HPP
#define CSP_WMS_SIMPLE_BODIES_HPP

#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>
#include <VistaOGLExt/VistaBufferObject.h>
#include <VistaOGLExt/VistaGLSLShader.h>
#include <VistaOGLExt/VistaVertexArrayObject.h>

#include "../../../src/cs-scene/CelestialBody.hpp"
#include "Plugin.hpp"
#include "WebMapTextureLoader.hpp"
#include "utils.hpp"

namespace cs::core {
class SolarSystem;
class TimeControl;
class Settings;
} // namespace cs::core

class VistaTexture;

namespace csp::simplewmsbodies {

/// This is just a sphere with a background texture overlaid with WMS based textures, attached to
/// the given SPICE frame. All of the textures should be in equirectangular projection.
class SimpleWMSBody : public cs::scene::CelestialBody, public IVistaOpenGLDraw {
 public:
  SimpleWMSBody(std::shared_ptr<cs::core::Settings> const& settings,
      std::shared_ptr<cs::core::SolarSystem>               solarSystem,
      std::shared_ptr<Plugin::Settings> const&             pluginSettings,
      std::shared_ptr<cs::core::TimeControl> timeControl, std::string const& sCenterName,
      std::string const& sFrameName, double tStartExistence, double tEndExistence);

  SimpleWMSBody(SimpleWMSBody const& other) = delete;
  SimpleWMSBody(SimpleWMSBody&& other)      = delete;

  SimpleWMSBody& operator=(SimpleWMSBody const& other) = delete;
  SimpleWMSBody& operator=(SimpleWMSBody&& other) = delete;

  ~SimpleWMSBody() override;

  /// Configures the internal renderer according to the given values.
  void configure(Plugin::Settings::SimpleWMSBody const& settings);

  /// The sun object is used for lighting computation.
  void setSun(std::shared_ptr<const cs::scene::CelestialObject> const& sun);

  /// Interface implementation of the IntersectableObject, which is a base class of
  /// CelestialBody.
  bool getIntersection(
      glm::dvec3 const& rayOrigin, glm::dvec3 const& rayDir, glm::dvec3& pos) const override;

  /// Interface implementation of CelestialBody.
  double     getHeight(glm::dvec2 lngLat) const override;
  glm::dvec3 getRadii() const override;

  /// Interface implementation of IVistaOpenGLDraw.
  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& bb) override;

  /// Set the active WMS data set.
  void setActiveWMS(std::shared_ptr<Plugin::Settings::WMSConfig> wms);

  std::vector<TimeInterval> getTimeIntervals();

 private:
  std::shared_ptr<cs::core::Settings>               mSettings;
  std::shared_ptr<cs::core::SolarSystem>            mSolarSystem;
  std::shared_ptr<const cs::scene::CelestialObject> mSun;
  std::shared_ptr<cs::core::TimeControl>            mTimeControl;

  std::unique_ptr<VistaOpenGLNode> mGLNode;

  glm::dvec3 mRadii;
  std::mutex mWMSMutex;

  std::shared_ptr<Plugin::Settings> mPluginSettings;
  Plugin::Settings::SimpleWMSBody   mSimpleWMSBodySettings;
  std::shared_ptr<Plugin::Settings::WMSConfig>
      mActiveWMS; ///< WMS config of the active WMS data set.

  std::shared_ptr<VistaTexture> mBackgroundTexture; ///< The background texture of the body.
  std::shared_ptr<VistaTexture> mWMSTexture;        ///< The WMS texture.
  std::shared_ptr<VistaTexture> mSecondWMSTexture;  ///< Second WMS texture for time interpolation.
  bool                          mWMSTextureUsed;    ///< Whether to use the WMS texture.
  bool        mSecondWMSTextureUsed = false;        ///< Whether to use the second WMS texture.
  std::string mCurrentTexture;                      ///< Timestep of the current WMS texture.
  std::string mCurrentSecondTexture;                ///< Timestep of the second WMS texture.
  float       mFade;                                ///< Fading value between WMS textures.
  std::string mRequest;                             ///< WMS server request URL.
  std::string mFormat;                              ///< Time format style.
  int         mIntervalDuration;                    ///< Duration of the current time interval.
  std::vector<TimeInterval> mTimeIntervals;         ///< Time intervals of data set.

  std::map<std::string, std::future<std::string>>    mTextureFilesBuffer;
  std::map<std::string, std::future<unsigned char*>> mTexturesBuffer;
  std::map<std::string, unsigned char*>              mTextures;
  std::vector<std::string>                           mWrongTextures;

  VistaGLSLShader        mShader;
  VistaVertexArrayObject mSphereVAO;
  VistaBufferObject      mSphereVBO;
  VistaBufferObject      mSphereIBO;

  WebMapTextureLoader mTextureLoader;

  bool mShaderDirty              = true;
  int  mEnableLightingConnection = -1;
  int  mEnableHDRConnection      = -1;

  uint32_t mGridResolutionX = 200;
  uint32_t mGridResolutionY = 100;

  static const std::string SPHERE_VERT;
  static const std::string SPHERE_FRAG;

  boost::posix_time::ptime getStartTime(boost::posix_time::ptime time);
};

} // namespace csp::simplewmsbodies

#endif // CSP_WMS_SIMPLE_BODIES_HPP
