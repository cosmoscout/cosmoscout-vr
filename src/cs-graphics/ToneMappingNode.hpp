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

/// The ToneMappingNode is used to bring the content of the HDRBuffer to the LDR backbuffer. It
/// applies filmic tonemapping based on http://filmicworlds.com/blog/filmic-tonemapping-operators/.
/// Auto-exposure is implemented using the LuminanceMipMap of the HDRBuffer, an artificial glare can
/// be added using the GlowMipMap of the HDRBuffer.
/// In order to compute the exposure when auto-exposure is enabled, the total luminance values of
/// all connected cluster slaves are taken into account.
class CS_GRAPHICS_EXPORT ToneMappingNode : public IVistaOpenGLDraw, public VistaEventHandler {
 public:
  /// The node will draw to the backbuffer using the contents from the given HDRBuffer.
  ToneMappingNode(std::shared_ptr<HDRBuffer> const& hdrBuffer);
  virtual ~ToneMappingNode();

  /// Set the exposure in EV. getExposure() can be used to retreive the current exposure if
  /// auto-exposure is enabled.
  void  setExposure(float ev);
  float getExposure() const;

  /// Add an additional amount of exposure in EV. This can be used to tweak the exposure computed if
  /// auto-exposure is enabled.
  void  setExposureCompensation(float ev);
  float getExposureCompensation() const;

  /// Set auto-exposure will be clamped to this lower bound.
  void  setMinAutoExposure(float ev);
  float getMinAutoExposure() const;

  /// Set auto-exposure will be clamped to this upper bound.
  void  setMaxAutoExposure(float ev);
  float getMaxAutoExposure() const;

  /// An unit-less exponent controlling the speed of exposure adaption during auto-exposure. The
  /// calculation is based on "Time-dependent visual adaptation for fast realistic image display"
  /// (https://dl.acm.org/citation.cfm?id=344810).
  void  setExposureAdaptionSpeed(float speed);
  float getExposureAdaptionSpeed() const;

  /// Enables auto-exposure. getExposure() can be used to retreive the current exposure if this is
  /// enabled.
  void setEnableAutoExposure(bool value);
  bool getEnableAutoExposure() const;

  /// Controls the amount of artificial glare. Should be in the range [0-1]. If set to zero, the
  /// GlowMipMap will not be updated which will increase performance.
  void  setGlowIntensity(float intensity);
  float getGlowIntensity() const;

  /// Returns the average and maximum luminance across all connected cluster nodes.
  float getLastAverageLuminance() const;
  float getLastMaximumLuminance() const;

  virtual bool Do() override;
  virtual bool GetBoundingBox(VistaBoundingBox& bb) override;

  virtual void HandleEvent(VistaEvent* event) override;

 private:
  std::shared_ptr<HDRBuffer> mHDRBuffer;

  float mExposureCompensation  = 0.f;
  bool  mEnableAutoExposure    = false;
  float mExposure              = 0.f;
  float mAutoExposure          = 0.f;
  float mMinAutoExposure       = -15.f;
  float mMaxAutoExposure       = 15.f;
  float mExposureAdaptionSpeed = 1.f;
  float mGlowIntensity         = 0.f;

  VistaGLSLShader* mShader;

  struct LuminanceData {
    int   mPixelCount       = 0;
    float mTotalLuminance   = 0;
    float mMaximumLuminance = 0;
  };

  LuminanceData mLocalLuminanceData;
  LuminanceData mGlobalLuminanceData;

  IVistaClusterDataCollect* mLuminanceCollect = nullptr;
  IVistaClusterDataSync*    mLuminanceSync    = nullptr;

  static const std::string sVertexShader;
  static const std::string sFragmentShader;
};
} // namespace cs::graphics
#endif // CS_GRAPHICS_TONEMAPPING_NODE_HPP
