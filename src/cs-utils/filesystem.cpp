////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "filesystem.hpp"

#include "utils.hpp"

#include <curlpp/Easy.hpp>
#include <curlpp/Info.hpp>
#include <curlpp/Infos.hpp>
#include <curlpp/Options.hpp>

#include <cmath>
#include <iostream>

namespace cs::utils::filesystem {

////////////////////////////////////////////////////////////////////////////////////////////////////

void createDirectoryRecursively(
    boost::filesystem::path const& path, boost::filesystem::perms permissions) {

  if (!boost::filesystem::exists(path.parent_path())) {
    createDirectoryRecursively(path.parent_path(), permissions);
  }

  boost::filesystem::create_directory(path);
  boost::filesystem::permissions(path, permissions);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::set<std::string> listFiles(std::string const& directory, std::regex const& regex) {
  std::set<std::string> result;

  for (auto& p : boost::filesystem::directory_iterator(directory)) {
    auto path = p.path().generic_path();

    if (std::regex_match(path.string(), regex) && boost::filesystem::is_regular_file(path)) {
      result.insert(path.string());
    }
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::set<std::string> listDirs(std::string const& directory, std::regex const& regex) {
  std::set<std::string> result;

  for (auto& p : boost::filesystem::directory_iterator(directory)) {
    auto path = p.path().generic_path();
    if (std::regex_match(path.string(), regex) && boost::filesystem::is_directory(path)) {
      result.insert(path.string());
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

void downloadFile(std::string const& url, std::string const& destination,
    std::function<void(double, double)> const& progressCallback) {
  createDirectoryRecursively(boost::filesystem::path(destination).parent_path());
  std::ofstream stream(destination, std::ofstream::out | std::ofstream::binary);

  if (!stream) {
    throw std::runtime_error("Failed to open " + destination + " for downloading " + url + "!");
  }

  curlpp::Easy request;
  request.setOpt(curlpp::options::Url(url));
  request.setOpt(curlpp::options::WriteStream(&stream));
  request.setOpt(curlpp::options::NoSignal(true));
  request.setOpt(curlpp::options::NoProgress(false));
  request.setOpt(curlpp::options::SslVerifyPeer(false));
  request.setOpt(curlpp::options::ProgressFunction(
      [&](double a, double b, double /*unused*/, double /*unused*/) {
        progressCallback(b, a);
        return 0;
      }));

  request.perform();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::utils::filesystem
