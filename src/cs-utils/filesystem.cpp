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

std::set<std::string> listFiles(std::string const& directory) {
  boost::filesystem::path               dir(directory);
  boost::filesystem::directory_iterator end_iter;

  std::set<std::string> result;

  if (boost::filesystem::exists(dir) && boost::filesystem::is_directory(dir)) {
    for (boost::filesystem::directory_iterator dir_iter(dir); dir_iter != end_iter; ++dir_iter) {
      if (boost::filesystem::is_regular_file(dir_iter->status())) {
        result.insert(boost::filesystem::path(*dir_iter).normalize().string());
      }
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
  request.setOpt(curlpp::options::ProgressFunction([&](double a, double b, double c, double d) {
    progressCallback(b, a);
    return 0;
  }));

  request.perform();

  std::cout << " ... Done" << std::endl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::utils::filesystem
