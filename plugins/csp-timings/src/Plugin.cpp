////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Plugin.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-utils/convert.hpp"
#include "logger.hpp"

#include <VistaKernel/VistaFrameLoop.h>
#include <VistaKernel/VistaSystem.h>
#include <iostream>

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::timings::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::timings {

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {

  logger().info("Loading plugin...");

  // The {mainUIZoom} will be ignored when loading the file from disc. This basically prevents all
  // other WebViews to be affected by the pMainUIScale factor. Why that is, is explained in the
  // documentation of cs::gui::WebView::setZoomLevel in great detail. This also means that all other
  // WebViews with an URL starting with "file://{mainUIZoom}../" will be automatically affected by
  // the pMainUIScale factor.
  mGuiItem = std::make_unique<cs::gui::GuiItem>(
      "file://{mainUIZoom}../share/resources/gui/timings.html", false);

  // Configure the positioning and attributes of the statistics GUI item.
  mGuiItem->setSizeX(600);
  mGuiItem->setSizeY(320);
  mGuiItem->setOffsetX(-300);
  mGuiItem->setOffsetY(500);
  mGuiItem->setRelPositionY(0.F);
  mGuiItem->setRelPositionX(1.F);
  mGuiItem->setIsInteractive(false);
  mGuiItem->setCanScroll(false);
  mGuiItem->setIsEnabled(false);

  // Add it to the local GUI area. This ensures that the statistics are drawn on each screen in a
  // clustered setup.
  mGuiManager->getLocalGuiArea().addItem(mGuiItem.get());

  // Add the sidebar user settings tab to the CosmoScout user interface.
  mGuiManager->addSettingsSectionToSideBarFromHTML(
      "Frame Timings", "timer", "../share/resources/gui/timings_settings.html");

  // This callback enables or disables the per-frame time measurements.
  mGuiManager->getGui()->registerCallback("timings.setEnableTimerQueries",
      "Enables or disables execution of timer queries for each frame.",
      std::function([this](bool enable) {
        mFrameTimings->pEnableMeasurements = enable;

        // If mFrameTimings->pEnableMeasurements are disabled, we make the two checkboxes of this
        // plugin unresponsive.
        if (enable) {
          mGuiManager->getGui()->executeJavascript(
              "document.querySelectorAll('.enable-if-timer-enabled').forEach((elem) => "
              "elem.classList.remove('unresponsive'));");
        } else {
          mGuiManager->getGui()->executeJavascript(
              "document.querySelectorAll('.enable-if-timer-enabled').forEach((elem) => "
              "elem.classList.add('unresponsive'));");
        }
      }));

  // Use the current state of mFrameTimings->pEnableMeasurements for our checkbox.
  mFrameTimingConnection = mFrameTimings->pEnableMeasurements.connectAndTouch([this](bool enable) {
    mGuiManager->setCheckboxValue("timings.setEnableTimerQueries", enable);
  });

  // Set the mEnableRecording value based on the corresponding checkbox.
  mGuiManager->getGui()->registerCallback("timings.setEnableRecording",
      "Enables or disables frame time recording.", std::function([this](bool enable) {
        mEnableRecording = enable;

        if (enable) {
          mGuiManager->getGui()->executeJavascript(
              "document.querySelector('.timings-record-button').innerHTML = "
              "'<i class=\"material-icons\">stop</i> Stop Recording';");
        } else {
          mGuiManager->getGui()->executeJavascript(
              "document.querySelector('.timings-record-button').innerHTML = "
              "'<i class=\"material-icons\">fiber_manual_record</i> Start New Recording';");
        }
      }));

  // Set the mEnableStatistics value based on the corresponding checkbox.
  mGuiManager->getGui()->registerCallback("timings.setEnableStatistics",
      "Shows or hides the on-screen timer statistics.",
      std::function([this](bool enable) { mEnableStatistics = enable; }));

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {

  // Enable or disable the statistics GUI item if necessary.
  mGuiItem->setIsEnabled(mEnableStatistics && mFrameTimings->pEnableMeasurements.get());

  // If frame timings are enabled, we may have to record them or update the on-screen statistics.
  if (mFrameTimings->pEnableMeasurements.get()) {

    // Get the frame timing information if either recording or the on-screen statistics are enabled.
    std::unordered_map<std::string, cs::utils::FrameTimings::QueryResult> timings;
    if (mEnableStatistics || mEnableRecording) {
      timings = mFrameTimings->getCalculatedQueryResults();
    }

    // Send the timing information to the statistics GUI item.
    if (mEnableStatistics) {
      std::string json("{");
      for (auto const& timing : timings) {
        uint64_t timeGPU(timing.second.mGPUTime);
        uint64_t timeCPU(timing.second.mCPUTime);

        uint64_t const thresholdNanos = 100000;
        if (timeGPU > thresholdNanos || timeCPU > thresholdNanos) {
          json += "\"" + timing.first + "\":[" + std::to_string(timeGPU) + "," +
                  std::to_string(timeCPU) + "],";
        }
      }
      json.back() = '}';

      if (json.length() <= 1) {
        json = "{}";
      }

      mGuiItem->callJavascript(
          "CosmoScout.timings.setData", json, GetVistaSystem()->GetFrameLoop()->GetFrameRate());
    }

    // Store the frame timing if we are in recording-mode.
    if (mEnableRecording) {
      mRecordedTimings.emplace_back(timings);
    }
  }

  // Recording seems to have stopped last frame, so write the output file!
  if (!mEnableRecording && !mRecordedTimings.empty()) {

    // We use the current date as a filename.
    auto timeString =
        cs::utils::convert::time::toString(boost::posix_time::microsec_clock::local_time());
    cs::utils::replaceString(timeString, ":", "-");
    cs::utils::replaceString(timeString, ".", "-");
    cs::utils::replaceString(timeString, "T", "-");
    cs::utils::replaceString(timeString, "Z", "");

    std::ofstream file("timing-" + timeString + ".csv");

    // First we collect all unique timer names. There may have been different timers active during
    // our recording session.
    std::set<std::string> timerNames;
    for (auto const& frameTiming : mRecordedTimings) {
      for (auto const& timing : frameTiming) {
        timerNames.insert(timing.first);
      }
    }

    // First we write the CSV heading. The first column contains the frame number.
    file << "frame";
    for (auto timerName : timerNames) {
      // Make sure that there are no ',' in the timer names.
      cs::utils::replaceString(timerName, ",", "_");
      file << ", " << timerName << " (CPU), " << timerName << " (GPU)";
    }
    file << std::endl;

    // Now we write the time series in the following lines.
    uint32_t frameCounter = 0;

    for (auto const& frameTiming : mRecordedTimings) {
      file << frameCounter++;
      for (auto const& timerName : timerNames) {
        auto it = frameTiming.find(timerName);

        if (it != frameTiming.end()) {
          file << fmt::format(
              ", {}, {}", it->second.mCPUTime * 0.000001, it->second.mGPUTime * 0.000001);
        } else {
          file << ", 0, 0";
        }
      }
      file << std::endl;
    }

    mRecordedTimings.clear();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  // Remove the settings tab of this plugin.
  mGuiManager->removeSettingsSection("Frame Timings");

  // Unregister all callbacks.
  mGuiManager->getGui()->unregisterCallback("timings.setEnableTimerQueries");
  mGuiManager->getGui()->unregisterCallback("timings.setEnableRecording");
  mGuiManager->getGui()->unregisterCallback("timings.setEnableStatistics");

  // Remove the statistic GUI item.
  mGuiManager->getLocalGuiArea().removeItem(mGuiItem.get());

  // Disconnect any signals.
  mFrameTimings->pEnableMeasurements.disconnect(mFrameTimingConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::timings
