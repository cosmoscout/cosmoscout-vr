////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "Plugin.hpp"

#include "../../../src/cs-core/GuiManager.hpp"
#include "APITypes.hpp"
#include "RenderTypes.hpp"
#include "RestRequestManager.hpp"
#include "logger.hpp"

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN cs::core::PluginBase* create() {
  return new csp::virtualsatellite::Plugin;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

EXPORT_FN void destroy(cs::core::PluginBase* pluginBase) {
  delete pluginBase; // NOLINT(cppcoreguidelines-owning-memory)
}

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace csp::virtualsatellite {

////////////////////////////////////////////////////////////////////////////////////////////////////

void from_json(nlohmann::json const& j, Plugin::Settings& o) {
  cs::core::Settings::deserialize(j, "url", o.mUrl);
}

void to_json(nlohmann::json& j, Plugin::Settings const& o) {
  cs::core::Settings::serialize(j, "url", o.mUrl);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::init() {
  logger().info("Loading plugin...");

  mOnLoadConnection = mAllSettings->onLoad().connect([this]() { onLoad(); });
  mOnSaveConnection = mAllSettings->onSave().connect(
      [this]() { mAllSettings->mPlugins["csp-virtual-satellite"] = *mPluginSettings; });

  mGuiManager->addPluginTabToSideBarFromHTML(
      "Virtual Satellite", "satellite", "../share/resources/gui/virtual-satellite-tab.html");
  mGuiManager->addSettingsSectionToSideBarFromHTML(
      "Virtual Satellite", "satellite", "../share/resources/gui/virtual-satellite-settings.html");
  mGuiManager->executeJavascriptFile("../share/resources/gui/js/csp-virtual-satellite.js");

  onLoad();

  logger().info("Loading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::update() {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::deInit() {
  logger().info("Unloading plugin...");

  mGuiManager->removePluginTab("Virtual Satellite");
  mGuiManager->removeSettingsSection("Virtual Satellite");

  mAllSettings->onLoad().disconnect(mOnLoadConnection);
  mAllSettings->onSave().disconnect(mOnSaveConnection);

  logger().info("Unloading done.");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::vec3 getPositionFromCA(CategoryAssignment const& categoryAssignment) {
  return {
      std::get<float>(categoryAssignment.positionXBean.value),
      std::get<float>(categoryAssignment.positionYBean.value),
      std::get<float>(categoryAssignment.positionZBean.value),
  };
}

glm::quat getRotationFromCA(CategoryAssignment const& categoryAssignment) {
  return glm::vec3{
      std::get<float>(categoryAssignment.rotationXBean.value) * glm::pi<float>() / 180.0,
      std::get<float>(categoryAssignment.rotationYBean.value) * glm::pi<float>() / 180.0,
      std::get<float>(categoryAssignment.rotationZBean.value) * glm::pi<float>() / 180.0,
  };
}

glm::vec3 getSizeFromCA(CategoryAssignment const& categoryAssignment) {
  return {
      std::get<float>(categoryAssignment.sizeXBean.value),
      std::get<float>(categoryAssignment.sizeYBean.value),
      std::get<float>(categoryAssignment.sizeZBean.value),
  };
}

glm::vec4 getColorFromCA(CategoryAssignment const& categoryAssignment) {
  auto value = std::get<int64_t>(categoryAssignment.colorBean.value);
  auto color = static_cast<uint32_t>(value);

  std::array<uint8_t, 4> _rgb{};
  std::memcpy(_rgb.data(), &color, sizeof(uint32_t));

  return {_rgb[2] / 255.0, _rgb[1] / 255.0, _rgb[0] / 255.0,
      1 - std::get<float>(categoryAssignment.transparencyBean.value)};
}

void Plugin::onLoad() {
  // Read settings from JSON.
  from_json(mAllSettings->mPlugins.at("csp-virtual-satellite"), *mPluginSettings);

  mManagementAPI = RestRequestManager(
      mPluginSettings->mUrl.get() + "rest/management/v0.0.1/project/", "user_a", "1234");
  mModelAPI = RestRequestManager(
      mPluginSettings->mUrl.get() + "rest/model/v0.0.1/repository/", "user_a", "1234");

  mGuiManager->getGui()->registerCallback("virtualSatellite.setRepository",
      "Set the current repository to the one with the given name.",
      std::function([this](std::string&& name) { setRepository(name); }));

  mGuiManager->getGui()->registerCallback("virtualSatellite.setRootSEI",
      "Set the current root SEI to the one with the given name.",
      std::function([this](std::string&& uuid) {
        if (!mRepoName) {
          return;
        }

        setRootSEI(uuid);
      }));

  // get repo name
  if (auto result = mManagementAPI.getRequest(""); result.is_array() && !result.empty()) {
    mRepositories = result.get<std::vector<std::string>>();

    mGuiManager->getGui()->callJavascript(
        "CosmoScout.gui.clearDropdown", "virtualSatellite.setRepository");
    mGuiManager->getGui()->callJavascript(
        "CosmoScout.gui.addDropdownValue", "virtualSatellite.setRepository", "None", "None", true);

    for (auto const& repoName : mRepositories) {
      mGuiManager->getGui()->callJavascript("CosmoScout.gui.addDropdownValue",
          "virtualSatellite.setRepository", repoName, repoName, false);
    }
  }

  // get root SEIs
  if (mRepoName) {
    auto result =
        mModelAPI.getRequest(*mRepoName + "/seis", {{"sync", "true"}, {"build", "false"}});
    auto rootSEIs = result.get<std::vector<BeanStructuralElementInstance>>();

    std::function<void(BeanStructuralElementInstance const&, uint32_t)> printTree;
    printTree = [this, &printTree](BeanStructuralElementInstance const& sei, uint32_t depth) {
      std::string indent = std::string(2 * depth, ' ');
      std::string categories;
      for (size_t i = 0; i < sei.categoryAssignments.size(); i++) {
        if (i > 0) {
          categories += ", ";
        }
        categories += sei.categoryAssignments[i].name;
      }

      logger().info("{}{}", indent, std::string(90 - 2 * depth, '-'));

      logger().info("{}|       UUID: {}", indent, sei.uuid);
      logger().info("{}|       Name: {}", indent, sei.name);
      logger().info("{}|       Type: {}", indent, sei.type.value_or(""));
      logger().info("{}| Categories: {}", indent, categories);

      if (!sei.categoryAssignments.empty() && sei.categoryAssignments[0].name == "Visualisation") {
        auto ca = getCA(sei.categoryAssignments[0].uuid);
        logger().info("{}|   CA Shape: {}", indent, std::get<std::string>(ca.shapeBean.value));
        logger().info("{}|   CA   Pos: ({}, {}, {})", indent,
            std::get<float>(ca.positionXBean.value), std::get<float>(ca.positionYBean.value),
            std::get<float>(ca.positionZBean.value));

        logger().info("{}|   CA   Rot: ({}, {}, {})", indent,
            std::get<float>(ca.rotationXBean.value), std::get<float>(ca.rotationYBean.value),
            std::get<float>(ca.rotationZBean.value));

        if (std::get<std::string>(ca.shapeBean.value) == "BOX") {
          logger().info("{}|   CA  Size: ({}, {}, {})", indent, std::get<float>(ca.sizeXBean.value),
              std::get<float>(ca.sizeYBean.value), std::get<float>(ca.sizeZBean.value));
        } else if (std::get<std::string>(ca.shapeBean.value) == "SPHERE") {
          logger().info("{}| CA  Radius: {}", indent, std::get<float>(ca.radiusBean.value));
        }

        if (ca.colorBean.value.index() != 0) {
          auto value = std::get<int64_t>(ca.colorBean.value);
          auto color = static_cast<uint32_t>(value);

          std::stringstream stream;
          stream << std::hex << color;
          std::string colorString(stream.str());

          std::array<uint8_t, 4> _rgb;
          std::memcpy(_rgb.data(), &color, sizeof(uint32_t));

          logger().info("{}|   CA Color: {}", indent, colorString);
          logger().info("{}|   CA Color: ({}, {}, {})", indent, _rgb[0] / 255.0, _rgb[1] / 255.0,
              _rgb[2] / 255.0);
        }

        if (ca.transparencyBean.value.index() != 0) {
          auto value = std::get<float>(ca.transparencyBean.value);
          logger().info("{}|   CA Alpha: {}", indent, value);
        }
      }

      logger().info("{}{}", indent, std::string(90 - 2 * depth, '-'));

      for (auto const& child : sei.children) {
        printTree(getSEI(child.uuid), depth + 1);
      }
    };

    for (auto& rootSEI : rootSEIs) {
      if (auto sei = getSEI(rootSEI.uuid);
          sei.type == "de.dlr.sc.virsat.model.extension.ps.ConfigurationTree") {
        printTree(sei, 0);
      }
    }

    std::vector<Box>    boxes;
    std::vector<Sphere> spheres;

    std::function<void(BeanStructuralElementInstance)> buildVisualisationTree;
    buildVisualisationTree = [this, &buildVisualisationTree, &boxes](
                                 BeanStructuralElementInstance const& sei) {
      if (sei.children.empty() && !sei.categoryAssignments.empty() &&
          sei.categoryAssignments[0].name == "Visualisation") {
        auto ca = getCA(sei.categoryAssignments[0].uuid);

        if (std::get<std::string>(ca.shapeBean.value) == "BOX") {
          Box box{
              getPositionFromCA(ca),
              getRotationFromCA(ca),
              getSizeFromCA(ca),
              getColorFromCA(ca),
          };

          auto currSei = sei;
          while (currSei.parent) {
            currSei = getSEI(currSei.parent.value());
            if (!currSei.categoryAssignments.empty() &&
                currSei.categoryAssignments[0].name == "Visualisation") {
              auto currCA = getCA(currSei.categoryAssignments[0].uuid);
              box.pos += getPositionFromCA(currCA);
              box.rot *= getRotationFromCA(currCA);
            }
          }
          boxes.push_back(box);
        }
      } else if (!sei.children.empty()) {
        for (auto const& child : sei.children) {
          buildVisualisationTree(getSEI(child.uuid));
        }
      }
    };

    for (auto& rootSEI : rootSEIs) {
      if (auto sei = getSEI(rootSEI.uuid);
          sei.type == "de.dlr.sc.virsat.model.extension.ps.ConfigurationTree") {
        buildVisualisationTree(sei);
      }
    }

    for (const auto& [pos, rot, size, color] : boxes) {
      logger().info("Box ---------------------------");
      logger().info("pos:   ({:.2f}, {:.2f}, {:.2f})", pos.x, pos.y, pos.z);
      logger().info("rot:   ({:.2f}, {:.2f}, {:.2f}, {:.2f})", rot.x, rot.y, rot.z, rot.w);
      logger().info("size:  ({:.2f}, {:.2f}, {:.2f})", size.x, size.y, size.z);
      logger().info("color: ({:.2f}, {:.2f}, {:.2f}, {:.2f})", color.r, color.g, color.b, color.a);
    }

    mBoxRenderer = std::make_unique<BoxRenderer>(mPluginSettings, mSolarSystem);
    mBoxRenderer->setObjectName("ISS");
    mBoxRenderer->setBoxes(boxes);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::setRepository(std::string const& repoName) {
  mGuiManager->getGui()->callJavascript("CosmoScout.virtualSatellite.resetSEISelect");
  mRootSEIs.clear();

  if (repoName == "None") {
    mRepoName.reset();
    return;
  }

  mRepoName = repoName;

  auto result = mModelAPI.getRequest(*mRepoName + "/seis", {{"sync", "true"}, {"build", "false"}});
  auto rootSEIs = result.get<std::vector<BeanStructuralElementInstance>>();

  for (auto const& rootSEI : rootSEIs) {
    auto sei = getSEI(rootSEI.uuid);
    mRootSEIs.emplace(std::pair{rootSEI.uuid, sei});

    mGuiManager->getGui()->callJavascript(
        "CosmoScout.virtualSatellite.addSEI", sei.uuid, sei.name, false);
  }

  mGuiManager->getGui()->callJavascript("CosmoScout.virtualSatellite.refreshSEISelect");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Plugin::setRootSEI(std::string const& uuid) {
  mBoxRenderer = std::make_unique<BoxRenderer>(mPluginSettings, mSolarSystem);
  mBoxRenderer->setObjectName("ISS");
  if (uuid == "None") {
    mRootSEI.reset();
    return;
  }

  mRootSEI = mRootSEIs.at(uuid);

  std::vector<Box> boxes;
  std::function<void(BeanStructuralElementInstance)> buildVisualisationTree;
    buildVisualisationTree = [this, &buildVisualisationTree, &boxes](
                                 BeanStructuralElementInstance const& sei) {
      if (sei.children.empty() && !sei.categoryAssignments.empty() &&
          sei.categoryAssignments[0].name == "Visualisation") {
        auto ca = getCA(sei.categoryAssignments[0].uuid);

        if (std::get<std::string>(ca.shapeBean.value) == "BOX") {
          Box box{
              getPositionFromCA(ca),
              getRotationFromCA(ca),
              getSizeFromCA(ca),
              getColorFromCA(ca),
          };

          auto currSei = sei;
          while (currSei.parent) {
            currSei = getSEI(currSei.parent.value());
            if (!currSei.categoryAssignments.empty() &&
                currSei.categoryAssignments[0].name == "Visualisation") {
              auto currCA = getCA(currSei.categoryAssignments[0].uuid);
              box.pos += getPositionFromCA(currCA);
              box.rot *= getRotationFromCA(currCA);
            }
          }
          boxes.push_back(box);
        }
      } else if (!sei.children.empty()) {
        for (auto const& child : sei.children) {
          buildVisualisationTree(getSEI(child.uuid));
        }
      }
    };

    buildVisualisationTree(*mRootSEI);

    mBoxRenderer->setBoxes(boxes);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

BeanStructuralElementInstance Plugin::getSEI(std::string const& uuid) {
  if (mSEICache.find(uuid) != mSEICache.end()) {
    return mSEICache.at(uuid);
  }

  auto result =
      mModelAPI.getRequest(*mRepoName + "/sei/" + uuid, {{"sync", "false"}, {"build", "false"}});
  auto sei = result.get<BeanStructuralElementInstance>();
  mSEICache.insert({uuid, sei});
  return sei;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

CategoryAssignment Plugin::getCA(std::string const& uuid) {
  if (mCACache.find(uuid) != mCACache.end()) {
    return mCACache.at(uuid);
  }

  auto result =
      mModelAPI.getRequest(*mRepoName + "/ca/" + uuid, {{"sync", "false"}, {"build", "false"}});
  auto ca = result.get<CategoryAssignment>();
  mCACache.insert({uuid, ca});
  return ca;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::virtualsatellite
