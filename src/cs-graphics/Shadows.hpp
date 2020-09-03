////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GRAPHICS_VISTASHADOWS_HPP
#define CS_GRAPHICS_VISTASHADOWS_HPP

#include "cs_graphics_export.hpp"

#include <VistaBase/VistaTransformMatrix.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>

#include <set>
#include <vector>

class VistaTexture;
class VistaFramebufferObj;
class VistaNode;

namespace cs::graphics {

class ShadowMap;

/// All objects which should cast shadows have to be derived from ShadowCaster. Typically they are
/// also derived from IVistaOpenGLDraw. See the Terrain class as an example.
class CS_GRAPHICS_EXPORT ShadowCaster {
 public:
  /// This will be called when the object has to be rendered into a shadow map the GL_MODELVIEW
  /// matrix will be loaded according to the world transform provided by getWorldTransform().
  virtual void doShadows() = 0;

  /// This will be called to retrieve the world transform of the caster.
  virtual bool getWorldTransform(VistaTransformMatrix& matTransform) const = 0;

  /// Called by registerCaster() from the shadow map.
  void       setShadowMap(ShadowMap* pShadowMap);
  ShadowMap* getShadowMap() const;

 protected:
  ShadowMap* mShadowMap = nullptr;
};

/// This shadow map implements parallel split cascaded shadow maps with percentage closer
/// filtering.
class CS_GRAPHICS_EXPORT ShadowMap : public IVistaOpenGLDraw {
 public:
  ShadowMap() = default;

  ShadowMap(ShadowMap const& other) = delete;
  ShadowMap(ShadowMap&& other)      = delete;

  ShadowMap& operator=(ShadowMap const& other) = delete;
  ShadowMap& operator=(ShadowMap&& other) = delete;

  ~ShadowMap() override;

  /// All objects which are able to cast shadows need to be registered.
  void registerCaster(ShadowCaster* caster);
  void deregisterCaster(ShadowCaster* caster);

  /// The light direction in world space.
  void                 setSunDirection(VistaVector3D const& direction);
  VistaVector3D const& getSunDirection() const;

  /// The resolution which is used for all cascades (default: 1024).
  void     setResolution(unsigned resolution);
  unsigned getResolution() const;

  /// The split distances in world space. Defaults to {0.1, 5, 20, 50, 100} but should definitely
  /// configured by the application.
  void                      setCascadeSplits(std::vector<float> const& splits);
  std::vector<float> const& getCascadeSplits() const;

  /// To make objects outside the view frustum cast or receive shadows, the sun frustum needs to be
  /// extended in light direction. Defaults to -500 and 500 respectively.
  void  setSunNearClipOffset(float offset);
  float getSunNearClipOffset() const;
  void  setSunFarClipOffset(float offset);
  float getSunFarClipOffset() const;

  /// The bias to prevent shadow acne (defaults to 0.0001).
  void  setBias(float bias);
  float getBias() const;

  /// For debugging, disables cascade updates.
  void setFreezeCascades(bool freeze);
  bool getFreezeCascades() const;

  /// Disables the generation of the shadow map.
  void setEnabled(bool enable);
  bool getEnabled() const;

  /// Returns all shadow maps which have been generated this frame and the matrices which should be
  /// used to perform lookups.
  std::vector<VistaTexture*> const&        getMaps() const;
  std::vector<VistaTransformMatrix> const& getShadowMatrices() const;

  bool Do() override;
  bool GetBoundingBox(VistaBoundingBox& oBoundingBox) override;

 private:
  void cleanUp();

  std::vector<VistaTexture*>        mShadowMaps;
  std::vector<VistaTransformMatrix> mShadowMatrices;
  std::vector<VistaFramebufferObj*> mShadowMapFBOs;
  std::set<ShadowCaster*>           mShadowCasters;
  VistaVector3D                     mSunDirection = VistaVector3D(0, 1, 0);
  unsigned                          mResolution   = 1024;
  std::vector<float>                mSplits;
  float                             mSunNearClipOffset = -500.0F;
  float                             mSunFarClipOffset  = 500.0F;
  float                             mBias              = 0.0001F;
  bool                              mFreezeCascades    = false;
  bool                              mEnabled           = true;

  VistaTransformMatrix matProjection, matView;

  bool mFBODirty = true;
};

} // namespace cs::graphics

#endif // CS_GRAPHICS_VISTASHADOWS_HPP
