////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Downloader.hpp"

#include "../cs-utils/filesystem.hpp"

namespace cs::utils {

////////////////////////////////////////////////////////////////////////////////////////////////////

Downloader::Downloader(size_t threadCount)
    : mThreadPool(threadCount) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void Downloader::download(std::string const& url, std::string const& file) {
  std::unique_lock<std::mutex> lock(mProgressMutex);
  size_t                       progressIndex = mProgress.size();
  mProgress.push_back({0.0, 0.0});

  mThreadPool.enqueue([this, file, url, progressIndex]() {
    if (!boost::filesystem::exists(file)) {
      filesystem::downloadFile(url, file, [this, progressIndex](double progress, double total) {
        std::unique_lock<std::mutex> lock(mProgressMutex);
        mProgress[progressIndex] = {progress, total};
      });
    }
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

double Downloader::getProgress() const {
  std::unique_lock<std::mutex> lock(mProgressMutex);

  double progress = 0.0, total = 0.0;

  for (auto const& p : mProgress) {
    progress += p.first;
    total += p.second;
  }

  if (total <= 0.0) {
    return 0.0;
  }

  return progress / total * 100.0;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool Downloader::hasFinished() const {
  return mThreadPool.hasFinished();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace cs::utils
