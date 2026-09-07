////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "filesystem.hpp"

#include "utils.hpp"

#include <curlcpp/curl_easy.h>
#include <curlcpp/curl_option.h>

#include <fstream>

namespace cs::utils::filesystem {

////////////////////////////////////////////////////////////////////////////////////////////////////

void createDirectoryRecursively(
    std::filesystem::path const& path, std::filesystem::perms permissions) {

  if (!std::filesystem::exists(path.parent_path())) {
    createDirectoryRecursively(path.parent_path(), permissions);
  }

  std::filesystem::create_directory(path);
  std::filesystem::permissions(path, permissions);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::set<std::string> listFiles(std::string const& directory, std::regex const& regex) {
  std::set<std::string> result;

  for (auto& p : std::filesystem::directory_iterator(directory)) {

    if (auto const& path = p.path();
        std::regex_match(path.string(), regex) && std::filesystem::is_regular_file(path)) {
      result.insert(path.generic_string());
    }
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::set<std::string> listDirs(std::string const& directory, std::regex const& regex) {
  std::set<std::string> result;

  for (auto& p : std::filesystem::directory_iterator(directory)) {
    if (auto const& path = p.path();
        std::regex_match(path.string(), regex) && std::filesystem::is_directory(path)) {
      result.insert(path.generic_string());
    }
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string loadToString(std::string const& file) {
  std::ifstream f(file);
  std::string   content;

  f.seekg(0, std::ios::end);
  content.reserve(f.tellg());
  f.seekg(0, std::ios::beg);

  content.assign((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  replaceString(content, "\r\n", "\n");

  return content;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void writeStringToFile(std::string const& filePath, std::string const& content) {
  std::ofstream file(filePath, std::ofstream::out);
  file << content;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static int progressWrapper(
    void* data, curl_off_t dlTotal, curl_off_t dlNow, curl_off_t, curl_off_t) {
  auto* callback = static_cast<std::function<void(double, double)> const*>(data);
  (*callback)(static_cast<double>(dlNow), static_cast<double>(dlTotal));
  return 0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void downloadFile(std::string const& url, std::string const& destination,
    std::function<void(double, double)> const& progressCallback) {
  createDirectoryRecursively(std::filesystem::path(destination).parent_path());
  std::ofstream stream(destination, std::ofstream::out | std::ofstream::binary);

  if (!stream) {
    throw std::runtime_error("Failed to open " + destination + " for downloading " + url + "!");
  }

  curl::curl_ios<std::ostream> streamIos(stream);

  curl::curl_easy request(streamIos);
  request.add<CURLOPT_URL>(url.c_str());
  request.add<CURLOPT_NOSIGNAL>(true);
  request.add<CURLOPT_NOPROGRESS>(false);
  request.add<CURLOPT_SSL_VERIFYPEER>(false);
  request.add<CURLOPT_FOLLOWLOCATION>(true);
  request.add<CURLOPT_XFERINFOFUNCTION>(&progressWrapper);
  request.add<CURLOPT_PROGRESSDATA>(
      curl::detail::Option_type<CURLOPT_XFERINFODATA>(&progressCallback));

  request.perform();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::utils::filesystem
