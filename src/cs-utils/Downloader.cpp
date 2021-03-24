////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Downloader.hpp"

#include "filesystem.hpp"
#include "logger.hpp"

namespace cs::utils {

////////////////////////////////////////////////////////////////////////////////////////////////////

Downloader::Downloader(size_t threadCount)
    : mThreadPool(threadCount) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Downloader::download(std::string const& url, std::string const& file) {
  if (boost::filesystem::exists(file)) {
    return;
  }

  std::unique_lock<std::mutex> lock(mProgressMutex);
  size_t                       progressIndex = mProgress.size();
  mProgress.emplace_back(0.0, 0.0);

  // We download to a file with a .part suffix. Once the download is done, we will remove the
  // suffix.
  mThreadPool.enqueue([this, file, url, progressIndex]() {
    logger().info("Downloading file '{}'...", file);

    filesystem::downloadFile(
        url, file + ".part", [this, progressIndex](double progress, double total) {
          std::unique_lock<std::mutex> lock(mProgressMutex);
          mProgress[progressIndex] = {progress, total};
        });

    std::rename((file + ".part").c_str(), file.c_str());
    logger().info("Finished downloading file '{}'.", file);
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double Downloader::getProgress() const {
  std::unique_lock<std::mutex> lock(mProgressMutex);

  double progress = 0.0;
  double total    = 0.0;

  for (auto const& p : mProgress) {
    progress += p.first;
    total += p.second;
  }

  if (total <= 0.0) {
    return 100.0;
  }

  return progress / total * 100.0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Downloader::hasFinished() const {
  return mThreadPool.hasFinished();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::utils
