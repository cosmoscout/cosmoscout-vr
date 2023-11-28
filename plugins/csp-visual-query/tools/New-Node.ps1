# ------------------------------------------------------------------------------------------------ #
#                                This file is part of CosmoScout VR                                #
# ------------------------------------------------------------------------------------------------ #

# SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
# SPDX-License-Identifier: MIT

<#
    .SYNOPSIS
    This function creates a new node with the given name in the given folder.

    .DESCRIPTION
    This function creates a new node with the given name in the given folder. It will generate all
    relevant files for you:
    - src/<Folder>/<Name>/<Name>.cpp
    - src/<Folder>/<Name>/<Name>.hpp
    - src/<Folder>/<Name>/<Name>.js

    .PARAMETER Name
    The node name. Just write it out in natural language. Don't use CamelCase, KebabCase,
    PascalCase or SnakeCase. The script will convert it to the correct case for each occurance.

    .PARAMETER Folder
    The name of the folder inside the of the src folder, your node belongs in.

    .EXAMPLE
    PS> .\New-Node -Name "Create Constant" -Folder "common-nodes"

    .EXAMPLE
    PS> .\New-Node -Name "Add" -Folder "operation-nodes"
#>

param (
    [Parameter(Mandatory=$true)]
    [ValidateNotNullOrEmpty()]
    [String]
    $Name,

    [Parameter(Mandatory=$true)]
    [ValidateNotNullOrEmpty()]
    [String]
    $Folder
)

$lowerCaseName = $Name.ToLower()
$upperCaseName = $Name.ToUpper()
$titleCaseName = (Get-Culture).TextInfo.ToTitleCase($Name)

$kebabCaseName = $lowerCaseName -replace ' ','-'
$lowerCaseJoinedName = $lowerCaseName -replace ' '
$pascalCaseName = $titleCaseName -replace ' '
$camelCaseName = $pascalCaseName.Substring(0, 1).ToLower() + $pascalCaseName.Substring(1)
$screamingSnakeCaseName = $upperCaseName -replace ' ','_'

$nodeRootDir = "src/$Folder/$pascalCaseName"

function New-Node {
    $nodeDir = New-Item -Path "$PSScriptRoot/../src/$Folder" -Name "$pascalCaseName" -ItemType "directory"

    New-CppNodeClass $nodeDir
    New-JSApi $nodeDir
}

function New-CppNodeClass($directory) {
    $nodeHeaderText = @"
////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_VISUAL_QUERY_${screamingSnakeCaseName}_HPP
#define CSP_VISUAL_QUERY_${screamingSnakeCaseName}_HPP

#include "../../../../csl-node-editor/src/Node.hpp"

#include <memory>

namespace csp::visualquery {

/// Your node description here!
class $pascalCaseName final : public csl::nodeeditor::Node {
 public:
  // static interface ------------------------------------------------------------------------------

  static const std::string               sName;
  static std::string                     sSource();
  static std::unique_ptr<$pascalCaseName> sCreate();

  // instance interface ----------------------------------------------------------------------------

  /// Each node must override this. It simply returns the static sName.
  std::string const& getName() const override;

  /// Whenever the value of the node is required, the ${pascalCaseName}Node will send a message to
  /// the C++ instance of the node via onMessageFromJS, which in turn will call the process()
  /// method.
  /// This method may also get called occasionally by the node editor,
  /// for example if a new web client was connected hence needs updated values for all nodes.
  void process() override;

  /// This will be called whenever the CosmoScout.sendMessageToCPP() is called by the JavaScript
  /// client part of this node.
  /// @param message  A JSON object as sent by the JavaScript node.
  void onMessageFromJS(nlohmann::json const& message) override;

  /// This is called whenever the node needs to be serialized.
  nlohmann::json getData() const override;

  /// This is called whenever the node needs to be deserialized.
  void setData(nlohmann::json const& json) override;
};

} // namespace csp::visualquery

#endif // CSP_VISUAL_QUERY_${screamingSnakeCaseName}_HPP
"@

    $nodeSourceText = @"
////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "$pascalCaseName.hpp"
#include "../../../../src/cs-utils/filesystem.hpp"

namespace csp::visualquery {

////////////////////////////////////////////////////////////////////////////////////////////////////

const std::string $pascalCaseName::sName = "$pascalCaseName";

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string $pascalCaseName::sSource() {
  return cs::utils::filesystem::loadToString(
      "../share/resources/nodes/csp-visual-query/$pascalCaseName.js");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::unique_ptr<$pascalCaseName> $pascalCaseName::sCreate() {
  return std::make_unique<$pascalCaseName>();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& $pascalCaseName::getName() const {
  return sName;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void $pascalCaseName::process() {
  // TODO
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void $pascalCaseName::onMessageFromJS(nlohmann::json const& message) {
  // TODO: Handle the message, maybe call process
  // process();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

nlohmann::json $pascalCaseName::getData() const {
  return {}; // TODO
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void $pascalCaseName::setData(nlohmann::json const& json) {
  // TODO
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::visualquery
"@

    Write-Host "Creating: '$nodeRootDir/$pascalCaseName.hpp'"
    $nodeHeaderText | Out-File "$directory/$pascalCaseName.hpp" -Encoding utf8

    Write-Host "Creating: '$nodeRootDir/$pascalCaseName.cpp'"
    $nodeSourceText | Out-File "$directory/$pascalCaseName.cpp" -Encoding utf8
}

function New-JSApi($directory) {
    $jsApiText = @"
////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

class ${pascalCaseName}Component extends Rete.Component {
  constructor() {
    super("${pascalCaseName}");

    // This specifies the submenu from which this node can be created in the node editor.
    this.category = "TODO"; // TODO
  }

  // Called whenever a new node of this type needs to be constructed.
  builder(node) {
    node.onInit = (nodeDiv) => {};
    return node;
  }
}

//# sourceMappingURL=$nodeRootDir/$pascalCaseName.js
"@

    Write-Host "Creating: '$nodeRootDir/$pascalCaseName.js'"
    $jsApiText | Out-File "$directory/$pascalCaseName.js" -Encoding utf8
}

New-Node
