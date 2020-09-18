////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Plugin.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-core/SolarSystem.hpp"
#include "../../../src/cs-core/TimeControl.hpp"
#include "../../../src/cs-utils/convert.hpp"
#include "../../../src/cs-utils/logger.hpp"
#include "logger.hpp"

#include <iostream>

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::recorder::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::recorder {

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "webAPIPort", o.mWebAPIPort);
  cs::core::Settings::deserialize(j, "recordObserver", o.mRecordObserver);
  cs::core::Settings::deserialize(j, "recordTime", o.mRecordTime);
  cs::core::Settings::deserialize(j, "recordExposure", o.mRecordExposure);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "webAPIPort", o.mWebAPIPort);
  cs::core::Settings::serialize(j, "recordObserver", o.mRecordObserver);
  cs::core::Settings::serialize(j, "recordTime", o.mRecordTime);
  cs::core::Settings::serialize(j, "recordExposure", o.mRecordExposure);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {

  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect(
      [this]() { mAllSettings->mPlugins["csp-recorder"] = mPluginSettings; });

  // Add the settings section to the side-bar.
  mGuiManager->addSettingsSectionToSideBarFromHTML(
      "Recorder", "fiber_manual_record", "../share/resources/gui/recorder_settings.html");

  // Add the record-button to the timeline. We remove / add the button every time it's pressed in
  // order to change its icon from a circle to a square and back again.
  mGuiManager->addTimelineButton(
      "Start Recording", "fiber_manual_record", "recorder.toggleRecording");

  mGuiManager->getGui()->registerCallback(
      "recorder.toggleRecording", "Enables or disables recording.", std::function([this]() {
        if (mRecording) {
          mGuiManager->removeTimelineButton("Stop Recording");
          mGuiManager->addTimelineButton(
              "Start Recording", "fiber_manual_record", "recorder.toggleRecording");
          mRecording = false;
        } else {
          mGuiManager->removeTimelineButton("Start Recording");
          mGuiManager->addTimelineButton("Stop Recording", "stop", "recorder.toggleRecording");
          mRecording = true;
        }
      }));

  // Add a callback to toggle recording of the observer transformation.
  mGuiManager->getGui()->registerCallback("recorder.setRecordObserver",
      "Enables or disables recording of the observer transformation.",
      std::function([this](bool value) { mPluginSettings.mRecordObserver = value; }));
  mPluginSettings.mRecordObserver.connectAndTouch(
      [this](bool enable) { mGuiManager->setCheckboxValue("recorder.setRecordObserver", enable); });

  // Add a callback to toggle recording of the simulation time.
  mGuiManager->getGui()->registerCallback("recorder.setRecordTime",
      "Enables or disables recording of the current simulation time.",
      std::function([this](bool value) { mPluginSettings.mRecordTime = value; }));
  mPluginSettings.mRecordTime.connectAndTouch(
      [this](bool enable) { mGuiManager->setCheckboxValue("recorder.setRecordTime", enable); });

  // Add a callback to toggle recording of the HDR exposure.
  mGuiManager->getGui()->registerCallback("recorder.setRecordExposure",
      "Enables or disables recording of the current camera exposure.",
      std::function([this](bool value) { mPluginSettings.mRecordExposure = value; }));
  mPluginSettings.mRecordExposure.connectAndTouch(
      [this](bool enable) { mGuiManager->setCheckboxValue("recorder.setRecordExposure", enable); });

  // Load initial settings.
  onLoad();

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {

  if (mRecording) {

    // We are recording but haven't opened the output file - that means it's the very first frame of
    // the current recording session. So open the file!
    if (!mOutFile.is_open()) {

      // Reset the frame counter.
      mFrameCounter = 0;

      // We use the current date as a filename.
      auto timeString =
          cs::utils::convert::time::toString(boost::posix_time::microsec_clock::local_time());
      cs::utils::replaceString(timeString, ":", "-");
      cs::utils::replaceString(timeString, ".", "-");
      cs::utils::replaceString(timeString, "T", "-");
      cs::utils::replaceString(timeString, "Z", "");

      mOutFile.open("recording-" + timeString + ".py");

      // Write the header of the file. This contains the functions which are then called for each
      // recorded frame.
      mOutFile << R"(#! python3

# This file has been automatically created by the csp-recorder plugin of CosmoScout VR.
# You can execute the script while CosmoScout VR is running. It will use the csp-web-api
# plugin to capture an image for each recorded frame.

import requests

# The CosmoScout server instance.
cosmoscout = "http://localhost:)" +
                      std::to_string(mPluginSettings.mWebAPIPort.get()) + R"("

# Simple one-liner to call JavaScript code on the CosmoScout VR side.
def runJS(code):
  requests.post(cosmoscout + "/run-js", code)

# Use this to capture a screenshot. The file can be a relative or absolute path to a png image,
# but the directory must exist.
def capture(file):
  r = requests.get(cosmoscout + "/capture?delay=1&format=png")
  with open(file, 'wb') as f:
    f.write(r.content)

)" << std::endl;
    }

    // Now that the output file is initialized, we can write the information for each frame. Here we
    // write a call navigation.setBodyFull() setting the current full observer transformation.
    if (mPluginSettings.mRecordTime.get()) {
      mOutFile << fmt::format("runJS(\"CosmoScout.callbacks.navigation.setBodyFull('{}', '{}', "
                              "{}, {}, {}, {}, {}, {}, {}, 0);\")",
                      mAllSettings->mObserver.pCenter.get(), mAllSettings->mObserver.pFrame.get(),
                      mAllSettings->mObserver.pPosition.get().x,
                      mAllSettings->mObserver.pPosition.get().y,
                      mAllSettings->mObserver.pPosition.get().z,
                      mAllSettings->mObserver.pRotation.get().x,
                      mAllSettings->mObserver.pRotation.get().y,
                      mAllSettings->mObserver.pRotation.get().z,
                      mAllSettings->mObserver.pRotation.get().w)
               << std::endl;
    }

    // Record the time if required.
    if (mPluginSettings.mRecordTime.get()) {
      mOutFile << "runJS(\"CosmoScout.callbacks.time.setDate('"
               << cs::utils::convert::time::toString(mTimeControl->pSimulationTime.get())
               << "');\")" << std::endl;
    }

    // Record the HDR exposure if required.
    if (mPluginSettings.mRecordExposure.get()) {
      mOutFile << "runJS(\"CosmoScout.callbacks.graphics.setExposure("
               << mAllSettings->mGraphics.pExposure.get() << ");\")" << std::endl;
    }

    // In any case, capture an image.
    mOutFile << "capture('frame_" << mFrameCounter++ << ".png')" << std::endl << std::endl;

  } else {

    // Recording has stopped last frame, so close the output file.
    if (mOutFile.is_open()) {
      mOutFile.close();
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  // Clean up the side-bar.
  mGuiManager->removeSettingsSection("Recorder");

  // Unregister all callbacks.
  mGuiManager->getGui()->unregisterCallback("recorder.toggleRecording");
  mGuiManager->getGui()->unregisterCallback("recorder.setRecordObserver");
  mGuiManager->getGui()->unregisterCallback("recorder.setRecordTime");
  mGuiManager->getGui()->unregisterCallback("recorder.setRecordExposure");

  // Remove the record button.
  if (mRecording) {
    mGuiManager->removeTimelineButton("Stop Recording");
  } else {
    mGuiManager->removeTimelineButton("Start Recording");
  }

  // Disconnect all signals.
  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {
  // Read settings from JSON.
  from_json(mAllSettings->mPlugins.at("csp-recorder"), mPluginSettings);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::recorder
