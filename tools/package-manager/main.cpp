#include "../../src/cs-utils/CommandLine.hpp"
#include <boost/filesystem.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <regex>
#include <spdlog/spdlog.h>

struct CSPMSettings {
  std::string configPath;
  std::string packagePath;
};

void installPackageFile(const CSPMSettings& settings, std::string_view actualPath);
void installPackageLink(const CSPMSettings& settings);
void mergeJSONFiles(std::string_view configFile, std::string_view packageFile);

int main(int argc, char** argv) {
  spdlog::set_pattern("%^[%L]%$ %v");

  bool printHelp = argc == 1;
  bool revert    = false;

  CSPMSettings settings{"../share/config/simple_desktop.json", ""};

  cs::utils::CommandLine args(
      "Welcome to the CosmoScout VR Package Manager (cspm)! Here are the available options:");

  args.addArgument({"-c", "--config"}, &settings.configPath,
      "The path to the config file you want to add your package to.");

  args.addArgument({"-p", "--package", "-i", "--install"}, &settings.packagePath,
      "The path or link to the package you want to install.");

  args.addArgument({"-r", "--revert"}, &revert, "Reverts the last change to the given config.");

  args.addArgument({"-h", "--help"}, &printHelp, "Show this help message.");

  try {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    std::vector<std::string> arguments(argv + 1, argv + argc);
    args.parse(arguments);
  } catch (std::runtime_error const& e) {
    spdlog::error("Failed to parse command line arguments:{}", e.what());
    return 1;
  }

  if (printHelp) {
    args.printHelp();
    return 0;
  }

  if (revert) {
    const auto        revertPath = boost::filesystem::path(settings.configPath);
    const std::string backupPath = "./backups/" + revertPath.filename().string();

    if (boost::filesystem::exists(backupPath)) {
      boost::filesystem::remove(revertPath);
      boost::filesystem::copy_file(backupPath, revertPath);
      spdlog::info("Reverted successfully!");
      exit(0);
    } else {
      spdlog::error("No backup for '{}' found!", settings.configPath);
      exit(1);
    }
  }

  if (settings.packagePath.empty()) {
    spdlog::error("Not given a package to install.");
    return 1;
  }

  // TODO if config doesn't exist, create a new one.
  if (!boost::filesystem::exists(settings.configPath)) {
    spdlog::error("The config '{}' doesn't exist.", settings.configPath);
    return 1;
  }

  const std::regex urlRegex =
      std::regex(R"(^(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w \.-]*)*\/?$)");

  if (boost::filesystem::exists(settings.packagePath)) {
    installPackageFile(settings, settings.packagePath);
  } else if (std::regex_match(settings.packagePath, urlRegex)) {
    installPackageLink(settings);
  } else {
    spdlog::error(
        "Could not load the package: '{}'! It is not a valid file or link.", settings.packagePath);
    return 1;
  }

  return 0;
}

void installPackageFile(const CSPMSettings& settings, std::string_view actualPath) {
  spdlog::info("Installing package: '{}'", settings.packagePath);

  std::string extension = boost::filesystem::extension(actualPath.data());
  if (extension == ".json") {
    mergeJSONFiles(settings.configPath, actualPath);
  } else if (extension == ".zip") {
    spdlog::warn("This feature isn't supported yet! Coming soon. (TM)");
    exit(0);
  } else {
    spdlog::error(
        "Unknown package format: '{}'! Please provide a '.json' or '.zip' file.", extension);
    exit(1);
  }
}

void installPackageLink(const CSPMSettings& settings) {
  spdlog::info("Installing package: '{}'", settings.packagePath);

  spdlog::warn("This feature isn't supported yet! Coming soon. (TM)");
  exit(0);
}

void mergeJSONFiles(std::string_view configFile, std::string_view packageFile) {
  std::ifstream  configStream(configFile.data());
  nlohmann::json config;
  configStream >> config;

  std::ifstream  packageStream(packageFile.data());
  nlohmann::json package;
  packageStream >> package;

  const auto oldConfig = config;
  config.merge_patch(package);

  const auto diff = nlohmann::json::diff(oldConfig, config);

  std::vector<nlohmann::json> deletions{};
  std::vector<nlohmann::json> replacements{};
  std::vector<nlohmann::json> additions{};

  for (const auto& entry : diff) {
    const auto operation = entry.at("op").get<std::string>();
    if (operation == "remove") {
      deletions.push_back(entry);
    } else if (operation == "replace") {
      replacements.push_back(entry);
    } else {
      additions.push_back(entry);
    }
  }

  if (!deletions.empty() || !replacements.empty()) {
    spdlog::warn("The installation of this package might do unwanted changes!");

    if (!deletions.empty()) {
      spdlog::warn("  DELETIONS");
      for (const auto& deletion : deletions) {
        spdlog::warn("   - {}", deletion.at("path").get<std::string>());
      }
    }

    if (!replacements.empty()) {
      spdlog::warn("  REPLACEMENTS");
      for (const auto& replacement : replacements) {
        spdlog::warn("   - {}: {}", replacement.at("path").get<std::string>(),
            replacement.at("value").dump());
      }
    }

    if (!additions.empty()) {
      spdlog::warn("  ADDITIONS");
      for (const auto& addition : additions) {
        spdlog::warn(
            "   - {}: {}", addition.at("path").get<std::string>(), addition.at("value").dump());
      }
    }

    spdlog::warn("Do you want to continue? (y/n)");
    std::string answer{};
    std::cin >> answer;
    if (answer != "y" && answer != "Y" && answer != "yes") {
      spdlog::info("Aborting.");
      exit(0);
    }
  }

  if (!boost::filesystem::exists("./backups")) {
    boost::filesystem::create_directory("./backups");
  }

  const auto backupPath = boost::filesystem::path(configFile.data());
  const auto backupFile = "./backups/" + backupPath.filename().string();

  if (boost::filesystem::exists(backupFile)) {
    boost::filesystem::remove(backupFile);
  }

  boost::filesystem::copy_file(configFile.data(), backupFile);

  const auto epoch = std::chrono::system_clock::now().time_since_epoch() / std::chrono::seconds(1);
  boost::filesystem::copy_file(configFile.data(), backupFile + "_" + std::to_string(epoch));

  std::ofstream newConfigStream(configFile.data());
  newConfigStream << config.dump(2);
}