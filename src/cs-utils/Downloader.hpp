////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_UTILS_DOWNLOADER_HPP
#define CS_UTILS_DOWNLOADER_HPP

#include "cs_utils_export.hpp"

#include "ThreadPool.hpp"

namespace cs::utils {

class CS_UTILS_EXPORT Downloader {
 public:
  Downloader(size_t threadCount);

  void download(std::string const& url, std::string const& file);

  double getProgress() const;

  bool hasFinished() const;

 private:
  ThreadPool mThreadPool;

  mutable std::mutex                     mProgressMutex;
  std::vector<std::pair<double, double>> mProgress;
};

} // namespace cs::utils

#endif // CS_UTILS_DOWNLOADER_HPP
