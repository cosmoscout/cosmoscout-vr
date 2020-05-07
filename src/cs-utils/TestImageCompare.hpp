////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_UTILS_TESTIMAGECOMPARE_HPP
#define CS_UTILS_TESTIMAGECOMPARE_HPP

#include "cs_utils_export.hpp"

#include <VistaKernel/EventManager/VistaEventHandler.h>

#include <VistaKernel/VistaSystem.h>
#include <sstream>
#include <string>

namespace cs::utils {

/// This class is a helper for comparing rendered frames to reference images. It is supposed to be
/// only used in unit tests. It requires imagemagick to be available in your PATH, so it might only
/// work on Linux.
/// The constructor will create a VistaSystem which can then be accessed with the global
/// GetVistaSystem(). Once your scene is setup, you can call doComparison(). See the individual
/// methods for a more detailed insight into the usage.
class CS_UTILS_EXPORT TestImageCompare {
 public:
  /// This creates a VistaSystem which can then be accessed with the global GetVistaSystem(). Once
  /// your scene is setup, you can call doComparison(). The VistaSystem will run the number of
  /// frames specified, the final frame will be stored in test/<imageName>.png.
  /// The VistaSystem is configured to use the vista_test.ini in config/base/vista.
  TestImageCompare(std::string const& imageName, int32_t frame);

  TestImageCompare(TestImageCompare const& other) = delete;
  TestImageCompare(TestImageCompare&& other)      = delete;

  TestImageCompare& operator=(TestImageCompare const& other) = delete;
  TestImageCompare& operator=(TestImageCompare&& other) = delete;

  /// This deletes the VistaSystem created by the constructor.
  virtual ~TestImageCompare() = default;

  /// This method assumes that a reference image is stored in test/reference/<imageName>.png. It
  /// will create a test/<imageName>-diff.png showing which pixels of the test image differ. The
  /// method will return the output of imagemagick's compare command - the maximum error percentage
  /// amongst all pixels. That means in case of full equality, 0 will be returned.
  float doComparison();

 private:
  class FrameCapture : public VistaEventHandler {
   public:
    FrameCapture(std::string fileName, int32_t frame);
    void HandleEvent(VistaEvent* pEvent) override;

   private:
    std::string mFileName;
    int32_t     mFrame;
  } mCapture;

  std::ostringstream mVistaOutput;
  std::string        mImageName;

  std::unique_ptr<VistaSystem> mVS;
};

} // namespace cs::utils

#endif // CS_UTILS_TESTIMAGECOMPARE_HPP
