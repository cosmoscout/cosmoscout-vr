Param(
    [Parameter(Mandatory=$True)]
    [ValidateNotNullOrEmpty()]
    [String[]]
    $Name,

    [Boolean]
    $WithJSApi=$True,

    [Boolean]
    $WithSidebarTab=$True,

    [Boolean]
    $WithSettingsTab=$True
);

$LowerCaseName = $Name.ToLower();
$UpperCaseName = $Name.ToUpper();
$TitleCaseName = (Get-Culture).TextInfo.ToTitleCase($Name);

$DashedName = $LowerCaseName -replace ' ','-';
$LowerCaseJoinedName = $LowerCaseName -replace ' ';
$CamelCaseName = $TitleCaseName -replace ' ';
$LowerCamelCaseName = $CamelCaseName.Substring(0, 1).ToLower() + $CamelCaseName.Substring(1);
$SnakeCaseName = $LowerCaseName -replace ' ','_';
$ScreamingSnakeCaseName = $UpperCaseName -replace ' ','_';

function createREADME() {
    $ReadMeText = @"
# $TitleCaseName for CosmoScout VR

Put the plugin description here!

## Configuration

This plugin can be enabled with the following configuration in your ``settings.json``.
The given values present some good starting values for your customization:

``````javascript
{
    ...
    `"plugins`": {
        ...
        `"csp-$DashedName`": {
        }
    }
}
``````
"@;
    $ReadMeText | Out-File "README.md" -Encoding utf8;
}

function createCMakeLists() {
    $CMakeListsText = @"
# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

option(CSP_$ScreamingSnakeCaseName `"Enable compilation of this plugin`" ON)

if (NOT CSP_$ScreamingSnakeCaseName)
  return()
endif()

# build plugin -------------------------------------------------------------------------------------

file(GLOB SOURCE_FILES src/*.cpp)

# Resource files and header files are only added in order to make them available in your IDE.
file(GLOB HEADER_FILES src/*.hpp)
file(GLOB_RECURSE RESOURCE_FILES gui/*)

add_library(csp-$DashedName SHARED
  `${SOURCE_FILES}
  `${HEADER_FILES}
  `${RESOURCE_FILES}
)

target_link_libraries(csp-$DashedName
  PUBLIC
    cs-core
)

# Add this Plugin to a `"plugins`" folder in your IDE.
set_property(TARGET csp-$DashedName PROPERTY FOLDER `"plugins`")

# We mark all resource files as `"header`" in order to make sure that no one tries to compile them.
set_source_files_properties(`${RESOURCE_FILES} PROPERTIES HEADER_FILE_ONLY TRUE)

# Make directory structure available in your IDE.
source_group(TREE `"`${CMAKE_CURRENT_SOURCE_DIR}`" FILES
  `${SOURCE_FILES} `${HEADER_FILES} `${RESOURCE_FILES}
)

# install plugin -----------------------------------------------------------------------------------

install(TARGETS   csp-$DashedName DESTINATION `"share/plugins`")
$(if ($WithJSApi -or $WithSettingsTab -or $WithSidebarTab) {
  "install(DIRECTORY `"gui`"           DESTINATION `"share/resources`")"
})
"@;

    $CMakeListsText | Out-File "CMakeLists.txt" -Encoding utf8;
}

function createCppPluginClass() {
    $PluginHeaderText = @"
////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_${ScreamingSnakeCaseName}_PLUGIN_HPP
#define CSP_${ScreamingSnakeCaseName}_PLUGIN_HPP

#include `"../../../src/cs-core/PluginBase.hpp`"
#include `"../../../src/cs-core/Settings.hpp`"

#include <memory>

namespace csp::$LowerCaseJoinedName {

/// Your plugin description here!
class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {
  };

  void init() override;
  void deInit() override;
  void update() override;

 private:
  void onLoad();

  std::shared_ptr<Settings> mPluginSettings = std::make_shared<Settings>();

  int mOnLoadConnection = -1;
  int mOnSaveConnection = -1;
};
} // namespace csp::$LowerCaseJoinedName

#endif // CSP_${ScreamingSnakeCaseName}_PLUGIN_HPP
"@;

    $PluginSourceText = @"
////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include `"Plugin.hpp`"

#include `"../../../src/cs-core/GuiManager.hpp`"
#include `"logger.hpp`"

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::$LowerCaseJoinedName::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::$LowerCaseJoinedName {

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {
  logger().info(`"Loading plugin...`");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect(
      [this]() { mAllSettings->mPlugins[`"csp-$DashedName`"] = *mPluginSettings; });

  $(if ($WithJSApi) {
    "mGuiManager->addScriptToGuiFromJS(`"../share/resources/gui/js/csp-$DashedName.js`");"
  })
  $(if ($WithSidebarTab) {
    "mGuiManager->addPluginTabToSideBarFromHTML(`"$TitleCaseName`", `"label_off`", `"../share/resources/gui/$DashedName-tab.html`");"
  })
  $(if ($WithSettingsTab) {
    "mGuiManager->addSettingsSectionToSideBarFromHTML(`"$TitleCaseName`", `"label_off`", `"../share/resources/gui/$DashedName-settings.html`");"
  })

  onLoad();

  logger().info(`"Loading done.`");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info(`"Unloading plugin...`");

  $(if ($WithSidebarTab) {
    "mGuiManager->removePluginTab(`"$TitleCaseName`");"
  })
  $(if ($WithSettingsTab) {
    "mGuiManager->removeSettingsSection(`"$TitleCaseName`");"
  })

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info(`"Unloading done.`");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {
  // Read settings from JSON.
  from_json(mAllSettings->mPlugins.at(`"csp-$DashedName`"), *mPluginSettings);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::$LowerCaseJoinedName
"@;

    $PluginHeaderText | Out-File "Plugin.hpp" -Encoding utf8;
    $PluginSourceText | Out-File "Plugin.cpp" -Encoding utf8;
}

function createCppLogger() {
    $LoggerHeaderText = @"
////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_${ScreamingSnakeCaseName}_LOGGER_HPP
#define CSP_${ScreamingSnakeCaseName}_LOGGER_HPP

#include <spdlog/spdlog.h>

namespace csp::$LowerCaseJoinedName {

/// This creates the default singleton logger for "csp-$DashedName" when called for the first time
/// and returns it. See cs-utils/logger.hpp for more logging details.
spdlog::logger& logger();

} // namespace csp::$LowerCaseJoinedName

#endif // CSP_${ScreamingSnakeCaseName}_LOGGER_HPP
"@;

    $LoggerSourceText = @"
////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "logger.hpp"

#include "../../../src/cs-utils/logger.hpp"

namespace csp::$LowerCaseJoinedName {

////////////////////////////////////////////////////////////////////////////////////////////////////

spdlog::logger& logger() {
  static auto logger = cs::utils::createLogger("csp-$DashedName");
  return *logger;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::$LowerCaseJoinedName
"@;

    $LoggerHeaderText | Out-File "logger.hpp" -Encoding utf8;
    $LoggerSourceText | Out-File "logger.cpp" -Encoding utf8;
}

function createJSApi() {
    $JsApiText = @"
/* global IApi, CosmoScout */

(() => {
  /**
   * $TitleCaseName Api
   */
  class ${CamelCaseName}Api extends IApi {
    /**
     * @inheritDoc
     */
    name = '$LowerCamelCaseName';

    /**
     * @inheritDoc
     */
    init() {
    }

    /**
     * @inheritDoc
     */
    update() {
    }
  }

  CosmoScout.init(${CamelCaseName}Api);
})();
"@;

    $JsApiText | Out-File "csp-$DashedName.js" -Encoding utf8;
}

function createSidebarTabHtml() {
    $SidebarHtmlText = "<p>Your sidebar content here!</p>";
    $SidebarHtmlText | Out-File "$DashedName-tab.html" -Encoding utf8;
}

function createSettingsTabHtml() {
    $SettingsHtmlText = "<p>Your settings here!</p>";
    $SettingsHtmlText | Out-File "$DashedName-settings.html" -Encoding utf8;
}

$StartDirectory = Get-Location;

New-Item -Path "$PSScriptRoot/plugins" -Name "csp-$DashedName" -ItemType "directory";
Set-Location "$PSScriptRoot/plugins/csp-$DashedName";
$PluginRoot = Get-Location;

createREADME;
createCMakeLists;

New-Item -Path "." -Name "src" -ItemType "directory";
Set-Location "src";
createCppPluginClass;
createCppLogger;
Set-Location "..";

if ($WithJSApi -or $WithSidebarTab -or $WithSettingsTab) {
    New-Item -Path "." -Name "gui" -ItemType "directory";
    Set-Location "gui";

    if ($WithJSApi) {
        New-Item -Path "." -Name "js" -ItemType "directory";
        Set-Location "js";
        createJSApi;
        Set-Location "..";
    }

    if ($WithSidebarTab) {
        createSidebarTabHtml;
    }

    if ($WithSettingsTab) {
        createSettingsTabHtml;
    }
}

Set-Location $StartDirectory;