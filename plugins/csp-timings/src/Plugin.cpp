////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Plugin.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "../../../src/cs-core/PluginBase.hpp"
#include "../../../src/cs-utils/convert.hpp"
#include "../../../src/cs-utils/filesystem.hpp"
#include "logger.hpp"

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
  mGuiItem->setSizeX(500);
  mGuiItem->setSizeY(300);
  mGuiItem->setOffsetX(-250);
  mGuiItem->setOffsetY(0);
  mGuiItem->setRelPositionY(0.5F);
  mGuiItem->setRelPositionX(1.F);
  mGuiItem->setIsInteractive(true);
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
        cs::utils::FrameTimings::get().pEnableMeasurements = enable;

        // If cs::utils::FrameTimings::get().pEnableMeasurements are disabled, we make the two
        // checkboxes of this plugin unresponsive.
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

  // Use the current state of cs::utils::FrameTimings::get().pEnableMeasurements for our checkbox.
  mFrameTimingConnection =
      cs::utils::FrameTimings::get().pEnableMeasurements.connectAndTouch([this](bool enable) {
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
  mGuiItem->setIsEnabled(
      mEnableStatistics && cs::utils::FrameTimings::get().pEnableMeasurements.get());

  // If frame timings are enabled, we may have to record them or update the on-screen statistics.
  if (cs::utils::FrameTimings::get().pEnableMeasurements.get() &&
      (mEnableRecording || mEnableStatistics)) {

    // Only record ranges longer than 10 µs.
    const uint32_t minTimeNanos    = 10000;
    auto const&    ranges          = cs::utils::FrameTimings::get().getRanges();
    uint32_t       maxNestingLevel = 0;

    // Compute the maximum nesting level amongst all recorded ranges.
    for (auto const& range : ranges) {
      maxNestingLevel = std::max(maxNestingLevel, range.mNestingLevel);
    }

    // The outer vector contains all ranges for a specific nesting level.
    std::vector<std::vector<Range>> cpuRanges(maxNestingLevel + 1);
    std::vector<std::vector<Range>> gpuRanges(maxNestingLevel + 1);

    // Compute frame-relative timestamps in microseconds.
    if (!ranges.empty()) {
      auto gpuFrameStart = ranges[0].mGPUStart;
      auto cpuFrameStart = ranges[0].mCPUStart;

      for (auto const& range : ranges) {
        if (range.mGPUEnd - range.mGPUStart >= minTimeNanos) {
          gpuRanges[range.mNestingLevel].emplace_back(range.mName,
              static_cast<uint32_t>(range.mGPUStart - gpuFrameStart) / 1000,
              static_cast<uint32_t>(range.mGPUEnd - gpuFrameStart) / 1000);
        }

        if (range.mCPUEnd - range.mCPUStart >= minTimeNanos) {
          cpuRanges[range.mNestingLevel].emplace_back(range.mName,
              static_cast<uint32_t>(range.mCPUStart - cpuFrameStart) / 1000,
              static_cast<uint32_t>(range.mCPUEnd - cpuFrameStart) / 1000);
        }
      }
    }

    // Send the timing information to the statistics GUI item.
    if (mEnableStatistics) {
      auto toJSON = [](std::vector<std::vector<Range>> const& ranges) {
        nlohmann::json json;

        for (auto const& level : ranges) {
          nlohmann::json levelJSON;
          for (auto const& range : level) {
            nlohmann::json rangeJSON;
            rangeJSON.push_back(range.mName);
            rangeJSON.push_back(range.mStart);
            rangeJSON.push_back(range.mEnd);
            levelJSON.push_back(rangeJSON);
          }
          json.push_back(levelJSON);
        }

        return json.dump();
      };

      mGuiItem->callJavascript("CosmoScout.timings.setData", toJSON(gpuRanges), toJSON(cpuRanges));
    }

    // Store the frame timing if we are in recording-mode.
    if (mEnableRecording) {
      mRecordedGPURanges.push_back(std::move(gpuRanges));
      mRecordedCPURanges.push_back(std::move(cpuRanges));
    }
  }

  // Recording seems to have stopped last frame, so write the output file!
  if (!mEnableRecording && !mRecordedGPURanges.empty()) {

    // We use the current date as a directory name.
    auto timeString =
        cs::utils::convert::time::toString(boost::posix_time::microsec_clock::local_time());
    cs::utils::replaceString(timeString, ":", "-");
    cs::utils::replaceString(timeString, ".", "-");
    cs::utils::replaceString(timeString, "T", "-");
    cs::utils::replaceString(timeString, "Z", "");

    std::string directory = "csp-timings/" + timeString;
    cs::utils::filesystem::createDirectoryRecursively(
        boost::filesystem::system_complete(directory));

    // This stores a CSV file for each nesting level in the directory created above. The prefix will
    // be prepended to the CSV file name.
    auto saveRecording = [&directory](std::string const&                         prefix,
                             std::vector<std::vector<std::vector<Range>>> const& recording) {
      // Retrieve the maximum nesting level amongst all recorded frames. We need this to decide how
      // many files to create. The nesting level may actually change during a recording if for
      // example new objects come into view.
      std::size_t maxNestingLevel = 0;
      for (auto const& record : recording) {
        maxNestingLevel = std::max(maxNestingLevel, record.size());
      }

      // Create a file for each nesting level.
      for (std::size_t level = 0; level < maxNestingLevel; ++level) {

        std::ofstream csv(directory + "/" + prefix + "-level-" + std::to_string(level) + ".csv");

        // Find unique range names in current level over all recorded frames.
        std::set<std::string> rangeNames;
        for (auto const& record : recording) {
          for (auto const& range : record[level]) {
            std::string name = range.mName;
            cs::utils::replaceString(name, ",", "_");
            rangeNames.insert(name);
          }
        }

        // Write CSV header.
        csv << "Frame";
        for (auto const& name : rangeNames) {
          csv << ", " << name;
        }
        csv << std::endl;

        // Write one CSV line for each frame. If a range has not been recorded for a specific frame,
        // a zero will be written to the corresponding field.
        for (std::size_t i = 0; i < recording.size(); ++i) {

          // Accumulate time per range as there may be multiple ranges with the same name.
          std::map<std::string, uint32_t> rangeTimes;
          for (auto const& name : rangeNames) {
            rangeTimes[name] = 0;
          }

          for (auto const& range : recording[i][level]) {
            std::string name = range.mName;
            cs::utils::replaceString(name, ",", "_");
            rangeTimes[name] += range.mEnd - range.mStart;
          }

          // Write the individual values to the CSV file.
          csv << i;
          for (auto const& time : rangeTimes) {
            csv << ", " << time.second;
          }
          csv << std::endl;
        }
      }
    };

    // Call the lambda above, once for the GPU timings, once for the CPU timings.
    saveRecording("gpu", mRecordedGPURanges);
    saveRecording("cpu", mRecordedCPURanges);

    mRecordedGPURanges.clear();
    mRecordedCPURanges.clear();
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
  cs::utils::FrameTimings::get().pEnableMeasurements.disconnect(mFrameTimingConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::timings
