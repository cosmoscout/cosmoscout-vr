////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "TestImageCompare.hpp"

#include "utils.hpp"

#include <VistaKernel/EventManager/VistaEventManager.h>
#include <VistaKernel/EventManager/VistaSystemEvent.h>
#include <VistaKernel/VistaFrameLoop.h>
#include <VistaKernel/VistaSystem.h>

namespace cs::utils {

////////////////////////////////////////////////////////////////////////////////////////////////////

TestImageCompare::TestImageCompare(std::string const& imageName, int32_t frame)
    : mCapture("test/" + imageName + ".png", frame)
    , mImageName(imageName) {

  auto vs = new VistaSystem();
  vs->SetIniSearchPaths({"../share/config/vista"});

  auto        killAfterFrame = std::to_string(frame);
  int         argc           = 5;
  const char* n_argv[]       = {
      "", "-vistaini", "vista_test.ini", "-kill_after_frame", killAfterFrame.c_str()};
  const char** argv = n_argv;

  vstr::SetOutStream(&mVistaOutput);
  vstr::SetWarnStream(&mVistaOutput);
  vstr::SetDebugStream(&mVistaOutput);

  if (!vs->Init(argc, const_cast<char**>(argv))) {
    throw std::runtime_error("Failed to initialize VistaSystem!");
  }

  vs->GetEventManager()->AddEventHandler(
      &mCapture, VistaSystemEvent::GetTypeId(), VistaSystemEvent::VSE_POSTGRAPHICS);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TestImageCompare::~TestImageCompare() {
  delete GetVistaSystem();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

float TestImageCompare::doComparison() {

  GetVistaSystem()->Run();

  std::string testImage       = "test/" + mImageName + ".png";
  std::string referenceImage  = "test/reference/" + mImageName + ".png";
  std::string differenceImage = "test/" + mImageName + "-diff.png";

  // See http://www.imagemagick.org/Usage/compare/ for more info on the available comparision
  // operations.
  std::string compareCommand =
      "compare -metric PAE " + testImage + " " + referenceImage + " " + differenceImage + " 2>&1";

  // The result is something like "  123 (0.345)". We are interested in the part between brackets.
  std::string result     = cs::utils::exec(compareCommand);
  float       errorValue = -1.f;
  try {
    errorValue = stof(result.substr(result.find('(') + 1));
  } catch (...) { throw std::runtime_error("Imagemagick's compare failed:" + result); }
  return errorValue * 100.f;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

TestImageCompare::FrameCapture::FrameCapture(std::string const& fileName, int32_t frame)
    : mFileName(fileName)
    , mFrame(frame) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TestImageCompare::FrameCapture::HandleEvent(VistaEvent* pEvent) {
  if (pEvent->GetId() == VistaSystemEvent::VSE_POSTGRAPHICS) {
    if (GetVistaSystem()->GetFrameLoop()->GetFrameCount() == mFrame) {
      // We assume that the window's title is 'CosmoScout VR' which is specified in the
      // vista_test.ini in config/base/vista.
      cs::utils::exec("import -window 'CosmoScout VR' " + mFileName);
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::utils
