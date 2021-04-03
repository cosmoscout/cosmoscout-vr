# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

<#
    .SYNOPSIS
    This function creates a new CosmoScout VR plugin with the given name.
    
    .DESCRIPTION
    This function creates a new CosmoScout VR plugin with the given name. You can also choose a
    Material Icon that will be set for your plugin in the UI. It will generate all relevant files for
    your Plugin:
    - README.md
    - CMakeLists.txt
    - src/Plugin.hpp
    - src/Plugin.cpp
    - src/logger.hpp
    - src/logger.cpp
    
    It will also generate the following files by default, but you can disable them, if you don't
    want them:
    - gui/js/csp-<plugin-name>.js
    - gui/<plugin-name>-tab.html
    - gui/<plugin-name>-settings.html
    
    .PARAMETER Name
    The plugin name. Just write it out in natural language. Don't use CamelCase, KebabCase, 
    PascalCase or SnakeCase. The script will convert it to the correct case for each occurance.
    
    .PARAMETER Icon
    The name of the Material Icon that will be shown in the GUI for this plugin. You can find a
    list of icons to choose from here: https://fonts.google.com/icons?selected=Material+Icons

    .PARAMETER WithJSApi
    This will create a JavaScript Api class in the plugins gui folder and create the appropriate
    code in the Plugin.cpp file to load and register the class with the CosmoScout Api.

    .PARAMETER WithSidebarTab
    This will generate an HTML file, where you can write your sidebar tab content. The script will
    also create the appropriate code in the Plugin.cpp file to load and unload the sidebar tab.

    .PARAMETER WithSettingsTab
    This will generate an HTML file, where you can write your settings tab content. The script will
    also create the appropriate code in the Plugin.cpp file to load and unload the settings tab.

    .EXAMPLE
    PS> .\New-Plugin -Name "Hello World" -Icon "hail"

    .EXAMPLE
    PS> .\New-Plugin "Hello World" "hail"

    .EXAMPLE
    PS> .\New-Plugin "Hello World" -WithJSApi $false -WithSidebarTab $false -WithSettingsTab $false
#>

param (
    [Parameter(Mandatory=$true)]
    [ValidateNotNullOrEmpty()]
    [String[]]
    $Name,

    [ValidateNotNullOrEmpty()]
    [String[]]
    $Icon="label_off",

    [Boolean]
    $WithJSApi=$true,

    [Boolean]
    $WithSidebarTab=$true,

    [Boolean]
    $WithSettingsTab=$true
)

function New-Plugin {
    $lowerCaseName = $Name.ToLower()
    $upperCaseName = $Name.ToUpper()
    $titleCaseName = (Get-Culture).TextInfo.ToTitleCase($Name)

    $kebabCaseName = $lowerCaseName -replace ' ','-'
    $lowerCaseJoinedName = $lowerCaseName -replace ' '
    $pascalCaseName = $titleCaseName -replace ' '
    $camelCaseName = $pascalCaseName.Substring(0, 1).ToLower() + $pascalCaseName.Substring(1)
    $snakeCaseName = $lowerCaseName -replace ' ','_'
    $screamingSnakeCaseName = $upperCaseName -replace ' ','_'

    $pluginRootDir = "plugins/csp-$kebabCaseName"

    $startDirectory = Get-Location

    New-Item -Path "$PSScriptRoot/plugins" -Name "csp-$kebabCaseName" -ItemType "directory" | Out-Null
    Set-Location "$PSScriptRoot/plugins/csp-$kebabCaseName"

    New-README
    New-CMakeLists

    New-Item -Path "." -Name "src" -ItemType "directory" | Out-Null
    Set-Location "src"
    New-CppPluginClass
    New-CppLogger
    Set-Location ".."

    if ($WithJSApi -or $WithSidebarTab -or $WithSettingsTab) {
        New-Item -Path "." -Name "gui" -ItemType "directory" | Out-Null
        Set-Location "gui"

        if ($WithJSApi) {
            New-Item -Path "." -Name "js" -ItemType "directory" | Out-Null
            Set-Location "js"
            New-JSApi
            Set-Location ".."
        }

        if ($WithSidebarTab) {
            New-SidebarTabHtml
        }

        if ($WithSettingsTab) {
            New-SettingsTabHtml
        }
    }

    Set-Location $startDirectory
}

function New-README() {
    $readMeText = @"
# $titleCaseName for CosmoScout VR

Put the plugin description here!

## Configuration

This plugin can be enabled with the following configuration in your ``settings.json``.
The given values present some good starting values for your customization:

``````javascript
{
  ...
  `"plugins`": {
    ...
    `"csp-$kebabCaseName`": {
    }
  }
}
``````
"@
    Write-Host "Creating: '$pluginRootDir/README.md'"
    $readMeText | Out-File "README.md" -Encoding utf8
}

function New-CMakeLists() {
    $cMakeListsText = @"
# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
#       and may be used under the terms of the MIT license. See the LICENSE file for details.      #
#                         Copyright: (c) 2019 German Aerospace Center (DLR)                        #
# ------------------------------------------------------------------------------------------------ #

option(CSP_$screamingSnakeCaseName `"Enable compilation of this plugin`" ON)

if (NOT CSP_$screamingSnakeCaseName)
  return()
endif()

# build plugin -------------------------------------------------------------------------------------

file(GLOB SOURCE_FILES src/*.cpp)

# Resource files and header files are only added in order to make them available in your IDE.
file(GLOB HEADER_FILES src/*.hpp)
file(GLOB_RECURSE RESOURCE_FILES gui/*)

add_library(csp-$kebabCaseName SHARED
  `${SOURCE_FILES}
  `${HEADER_FILES}
  `${RESOURCE_FILES}
)

target_link_libraries(csp-$kebabCaseName
  PUBLIC
    cs-core
)

# Add this Plugin to a `"plugins`" folder in your IDE.
set_property(TARGET csp-$kebabCaseName PROPERTY FOLDER `"plugins`")

# We mark all resource files as `"header`" in order to make sure that no one tries to compile them.
set_source_files_properties(`${RESOURCE_FILES} PROPERTIES HEADER_FILE_ONLY true)

# Make directory structure available in your IDE.
source_group(TREE `"`${CMAKE_CURRENT_SOURCE_DIR}`" FILES
  `${SOURCE_FILES} `${HEADER_FILES} `${RESOURCE_FILES}
)

# install plugin -----------------------------------------------------------------------------------

install(TARGETS csp-$kebabCaseName DESTINATION `"share/plugins`")
$(if ($WithJSApi -or $WithSettingsTab -or $WithSidebarTab) {
    "install(DIRECTORY `"gui`" DESTINATION `"share/resources`")"
})
"@

    Write-Host "Creating: '$pluginRootDir/CMakeLists.txt'"
    $cMakeListsText | Out-File "CMakeLists.txt" -Encoding utf8
}

function New-CppPluginClass() {
    $pluginHeaderText = @"
////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_${screamingSnakeCaseName}_PLUGIN_HPP
#define CSP_${screamingSnakeCaseName}_PLUGIN_HPP

#include `"../../../src/cs-core/PluginBase.hpp`"
#include `"../../../src/cs-core/Settings.hpp`"

#include <memory>

namespace csp::$lowerCaseJoinedName {

/// Your plugin description here!
class Plugin : public cs::core::PluginBase {
 public:
  struct Settings {};

  void init() override;
  void deInit() override;
  void update() override;

 private:
  void onLoad();

  std::shared_ptr<Settings> mPluginSettings = std::make_shared<Settings>();

  int mOnLoadConnection = -1;
  int mOnSaveConnection = -1;
};

} // namespace csp::$lowerCaseJoinedName

#endif // CSP_${screamingSnakeCaseName}_PLUGIN_HPP
"@

    $pluginSourceText = @"
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
  return new csp::$lowerCaseJoinedName::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::$lowerCaseJoinedName {

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
      [this]() { mAllSettings->mPlugins[`"csp-$kebabCaseName`"] = *mPluginSettings; });

$(if ($WithJSApi) {
    "  mGuiManager->addScriptToGuiFromJS(`"../share/resources/gui/js/csp-$kebabCaseName.js`");"
})
$(if ($WithSidebarTab) {
    "  mGuiManager->addPluginTabToSideBarFromHTML(
      `"$titleCaseName`", `"$Icon`", `"../share/resources/gui/$kebabCaseName-tab.html`");"
})
$(if ($WithSettingsTab) {
    "  mGuiManager->addSettingsSectionToSideBarFromHTML(
      `"$titleCaseName`", `"$Icon`", `"../share/resources/gui/$kebabCaseName-settings.html`");"
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
    "  mGuiManager->removePluginTab(`"$titleCaseName`");"
})
$(if ($WithSettingsTab) {
    "  mGuiManager->removeSettingsSection(`"$titleCaseName`");"
})

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info(`"Unloading done.`");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::onLoad() {
  // Read settings from JSON.
  from_json(mAllSettings->mPlugins.at(`"csp-$kebabCaseName`"), *mPluginSettings);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::$lowerCaseJoinedName
"@

    Write-Host "Creating: '$pluginRootDir/src/Plugin.hpp'"
    $pluginHeaderText | Out-File "Plugin.hpp" -Encoding utf8

    Write-Host "Creating: '$pluginRootDir/src/Plugin.cpp'"
    $pluginSourceText | Out-File "Plugin.cpp" -Encoding utf8
}

function New-CppLogger() {
    $loggerHeaderText = @"
////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_${screamingSnakeCaseName}_LOGGER_HPP
#define CSP_${screamingSnakeCaseName}_LOGGER_HPP

#include <spdlog/spdlog.h>

namespace csp::$lowerCaseJoinedName {

/// This creates the default singleton logger for "csp-$kebabCaseName" when called for the first time
/// and returns it. See cs-utils/logger.hpp for more logging details.
spdlog::logger& logger();

} // namespace csp::$lowerCaseJoinedName

#endif // CSP_${screamingSnakeCaseName}_LOGGER_HPP
"@

    $loggerSourceText = @"
////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "logger.hpp"

#include "../../../src/cs-utils/logger.hpp"

namespace csp::$lowerCaseJoinedName {

////////////////////////////////////////////////////////////////////////////////////////////////////

spdlog::logger& logger() {
  static auto logger = cs::utils::createLogger("csp-$kebabCaseName");
  return *logger;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::$lowerCaseJoinedName
"@

    Write-Host "Creating: '$pluginRootDir/src/logger.hpp'"
    $loggerHeaderText | Out-File "logger.hpp" -Encoding utf8

    Write-Host "Creating: '$pluginRootDir/src/logger.cpp'"
    $loggerSourceText | Out-File "logger.cpp" -Encoding utf8
}

function New-JSApi() {
    $jsApiText = @"
/* global IApi, CosmoScout */

(() => {
  /**
  * $titleCaseName Api
  */
  class ${pascalCaseName}Api extends IApi {
    /**
     * @inheritDoc
     */
    name = '$camelCaseName'

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

  CosmoScout.init(${pascalCaseName}Api);
})();

//# sourceMappingURL=js/csp-$kebabCaseName.js
"@

    Write-Host "Creating: '$pluginRootDir/gui/js/csp-$kebabCaseName.js'"
    $jsApiText | Out-File "csp-$kebabCaseName.js" -Encoding utf8
}

function New-SidebarTabHtml() {
    $sidebarHtmlText = "<p>Your sidebar content here!</p>"

    Write-Host "Creating: '$pluginRootDir/gui/$kebabCaseName-tab.html'"
    $sidebarHtmlText | Out-File "$kebabCaseName-tab.html" -Encoding utf8
}

function New-SettingsTabHtml() {
    $settingsHtmlText = "<p>Your settings here!</p>"

    Write-Host "Creating: '$pluginRootDir/gui/$kebabCaseName-settings.html'"
    $settingsHtmlText | Out-File "$kebabCaseName-settings.html" -Encoding utf8
}

New-Plugin