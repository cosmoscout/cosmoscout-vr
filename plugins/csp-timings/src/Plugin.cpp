////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

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

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "useLocalGui", o.mUseLocalGui);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "useLocalGui", o.mUseLocalGui);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {

  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect([this]() { onSave(); });

  // The {mainUIZoom} will be ignored when loading the file from disc. This basically prevents all
  // other WebViews to be affected by the pMainUIScale factor. Why that is, is explained in the
  // documentation of cs::gui::WebView::setZoomLevel in great detail. This also means that all other
  // WebViews with an URL starting with "file://{mainUIZoom}../" will be automatically affected by
  // the pMainUIScale factor.
  mGuiItem = std::make_unique<cs::gui::GuiItem>(
      "file://{mainUIZoom}../share/resources/gui/timings.html", false);

  // Configure the positioning and attributes of the statistics GUI item.
  mGuiItem->setSizeX(500);
  mGuiItem->setSizeY(500);
  mGuiItem->setOffsetX(-250);
  mGuiItem->setOffsetY(0);
  mGuiItem->setRelPositionY(0.5F);
  mGuiItem->setRelPositionX(1.F);
  mGuiItem->setIsInteractive(true);
  mGuiItem->setCanScroll(false);
  mGuiItem->setIsEnabled(false);

  // If we add it to the local GUI area, the statistics are drawn on each screen in a clustered setup.
  mPluginSettings.mUseLocalGui.connectAndTouch([this](bool useLocal) {
    // Remove the statistic GUI item first in case it was added before. We don't exactly know whether
    // it was attached locally or globally, so we just attempt to remove it in both cases.
    if (mGuiManager->hasGlobalGuiArea()) {
      mGuiManager->getGlobalGuiArea().removeItem(mGuiItem.get());
    }

    mGuiManager->getLocalGuiArea().removeItem(mGuiItem.get());

    if (useLocal || !mGuiManager->hasGlobalGuiArea()) {
      mGuiManager->getLocalGuiArea().addItem(mGuiItem.get());
    } else {
      mGuiManager->getGlobalGuiArea().addItem(mGuiItem.get());
    }
  });

  // Add the sidebar user settings tab to the CosmoScout user interface.
  mGuiManager->addSettingsSectionToSideBarFromHTML(
      "Frame Timings", "timer", "../share/resources/gui/timings_settings.html");

  // This callback enables or disables the per-frame time measurements.
  mGuiManager->getGui()->registerCallback("timings.setEnableTimerQueries",
      "Enables or disables execution of timer queries for each frame.",
      std::function([this](bool enable) {
        cs::utils::FrameStats::get().pEnableMeasurements = enable;

        // If cs::utils::FrameStats::get().pEnableMeasurements are disabled, we make the two
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

  // Use the current state of cs::utils::FrameStats::get().pEnableMeasurements for our checkbox.
  mFrameTimingConnection =
      cs::utils::FrameStats::get().pEnableMeasurements.connectAndTouch([this](bool enable) {
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
          logger().debug("Timing started");
        } else {
          mGuiManager->getGui()->executeJavascript(
              "document.querySelector('.timings-record-button').innerHTML = "
              "'<i class=\"material-icons\">fiber_manual_record</i> Start New Recording';");
          logger().debug("Timing finished");
        }
      }));

  // Set the mEnableStatistics value based on the corresponding checkbox.
  mGuiManager->getGui()->registerCallback("timings.setEnableStatistics",
      "Shows or hides the on-screen timer statistics.",
      std::function([this](bool enable) { mEnableStatistics = enable; }));

  // Load settings.
  onLoad();

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {

  // Enable or disable the statistics GUI item if necessary.
  mGuiItem->setIsEnabled(
      mEnableStatistics && cs::utils::FrameStats::get().pEnableMeasurements.get());

  // If frame timings are enabled, we may have to record them or update the on-screen statistics.
  if (cs::utils::FrameStats::get().pEnableMeasurements.get() &&
      (mEnableRecording || mEnableStatistics)) {

    // Only record ranges longer than 10 µs.
    const uint32_t minTimeNanos      = 10000;
    auto const&    timerQueryResults = cs::utils::FrameStats::get().getTimerQueryResults();
    uint32_t       maxNestingLevel   = 0;

    // Compute the maximum nesting level amongst all recorded ranges.
    for (auto const& timerQueryResult : timerQueryResults) {
      maxNestingLevel = std::max(maxNestingLevel, timerQueryResult.mNestingLevel);
    }

    // The outer vector contains all timing ranges for a specific nesting level.
    std::vector<std::vector<TimerRange>> cpuRanges(maxNestingLevel + 1);
    std::vector<std::vector<TimerRange>> gpuRanges(maxNestingLevel + 1);

    // Compute frame-relative timestamps in microseconds.
    if (!timerQueryResults.empty()) {
      auto gpuFrameStart = timerQueryResults[0].mGPUStart;
      auto cpuFrameStart = timerQueryResults[0].mCPUStart;

      for (auto const& timerQueryResult : timerQueryResults) {
        if (timerQueryResult.mGPUEnd - timerQueryResult.mGPUStart >= minTimeNanos) {
          gpuRanges[timerQueryResult.mNestingLevel].emplace_back(timerQueryResult.mName,
              static_cast<uint32_t>(timerQueryResult.mGPUStart - gpuFrameStart) / 1000,
              static_cast<uint32_t>(timerQueryResult.mGPUEnd - gpuFrameStart) / 1000);
        }

        if (timerQueryResult.mCPUEnd - timerQueryResult.mCPUStart >= minTimeNanos) {
          cpuRanges[timerQueryResult.mNestingLevel].emplace_back(timerQueryResult.mName,
              static_cast<uint32_t>(timerQueryResult.mCPUStart - cpuFrameStart) / 1000,
              static_cast<uint32_t>(timerQueryResult.mCPUEnd - cpuFrameStart) / 1000);
        }
      }
    }

    auto const& samplesQueryResults    = cs::utils::FrameStats::get().getSamplesQueryResults();
    auto const& primitivesQueryResults = cs::utils::FrameStats::get().getPrimitivesQueryResults();

    // Send the timing information to the statistics GUI item.
    if (mEnableStatistics) {
      auto rangeToJSON = [](std::vector<std::vector<TimerRange>> const& ranges) {
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

      auto countToJSON = [](std::vector<cs::utils::FrameStats::CounterQueryResult> const& counts) {
        nlohmann::json json;

        for (auto const& count : counts) {
          json.push_back({count.mName, count.mCount});
        }

        return json.dump();
      };

      mGuiItem->callJavascript("CosmoScout.timings.setData", rangeToJSON(gpuRanges),
          rangeToJSON(cpuRanges), countToJSON(samplesQueryResults),
          countToJSON(primitivesQueryResults));
    }

    // Store the frame timing if we are in recording-mode.
    if (mEnableRecording) {
      mTimestamps.push_back(std::chrono::high_resolution_clock::now().time_since_epoch().count());
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
    auto saveRecording = [&directory, this](std::string const&                        prefix,
                             std::vector<std::vector<std::vector<TimerRange>>> const& recording) {
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
        csv << "Frame, Timestamp";
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
          csv << i << ", " << mTimestamps[i];
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

  // Remove the statistic GUI item. We don't exactly know whether it was attached locally or
  // globally, so we just attempt to remove it in both cases.
  if (mGuiManager->hasGlobalGuiArea()) {
    mGuiManager->getGlobalGuiArea().removeItem(mGuiItem.get());
  }
  mGuiManager->getLocalGuiArea().removeItem(mGuiItem.get());

  // Disconnect any signals.
  cs::utils::FrameStats::get().pEnableMeasurements.disconnect(mFrameTimingConnection);

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {

  // Read settings from JSON.
  from_json(mAllSettings->mPlugins.at("csp-timings"), mPluginSettings);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onSave() {
  mAllSettings->mPlugins["csp-timings"] = mPluginSettings;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::timings
