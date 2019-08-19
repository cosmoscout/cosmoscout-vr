////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_GRAPHICS_TONEMAPPING_NODE_HPP
#define CS_GRAPHICS_TONEMAPPING_NODE_HPP

#include "cs_graphics_export.hpp"

#include "HDRBuffer.hpp"

#include <VistaKernel/EventManager/VistaEventHandler.h>
#include <VistaKernel/GraphicsManager/VistaOpenGLDraw.h>
#include <memory>

class VistaGLSLShader;
class IVistaClusterDataCollect;
class IVistaClusterDataSync;

namespace cs::graphics {

class CS_GRAPHICS_EXPORT ToneMappingNode : public IVistaOpenGLDraw, public VistaEventHandler {
 public:
  ToneMappingNode(std::shared_ptr<HDRBuffer> const& hdrBuffer, bool drawToBackBuffer);
  virtual ~ToneMappingNode();

  void  setExposure(float ev);
  float getExposure() const;

  void  setExposureCompensation(float ev);
  float getExposureCompensation() const;

  void  setMinAutoExposure(float ev);
  float getMinAutoExposure() const;

  void  setMaxAutoExposure(float ev);
  float getMaxAutoExposure() const;

  void  setExposureAdaptionSpeed(float speed);
  float getExposureAdaptionSpeed() const;

  void setEnableAutoExposure(bool value);
  bool getEnableAutoExposure() const;

  void                 setExposureMeteringMode(ExposureMeteringMode value);
  ExposureMeteringMode getExposureMeteringMode() const;

  void  setGlowIntensity(float intensity);
  float getGlowIntensity() const;

  float getLastAverageLuminance() const;

  virtual bool Do() override;
  virtual bool GetBoundingBox(VistaBoundingBox& bb) override;

  virtual void HandleEvent(VistaEvent* event) override;

 private:
  std::shared_ptr<HDRBuffer> mHDRBuffer;
  bool                       mDrawToBackBuffer;

  float mExposureCompensation  = 0.f;
  bool  mEnableAutoExposure    = false;
  float mExposure              = 0.f;
  float mAutoExposure          = 0.f;
  float mMinAutoExposure       = -15.f;
  float mMaxAutoExposure       = 15.f;
  float mExposureAdaptionSpeed = 1.f;
  float mGlowIntensity         = 0.f;

  ExposureMeteringMode mExposureMeteringMode = ExposureMeteringMode::AVERAGE;

  VistaGLSLShader* mShader;

  struct LuminanceData {
    int   mPixelCount     = 0;
    float mTotalLuminance = 0;
  };

  LuminanceData mLocalLuminaceData;
  LuminanceData mGlobalLuminaceData;

  IVistaClusterDataCollect* mLuminanceCollect = nullptr;
  IVistaClusterDataSync*    mLuminanceSync    = nullptr;

  static const std::string SHADER_VERT;
  static const std::string SHADER_FRAG;
};
} // namespace cs::graphics
#endif // CS_GRAPHICS_TONEMAPPING_NODE_HPP
