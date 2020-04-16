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

/// This class can be used to download a set of files in parallel.
class CS_UTILS_EXPORT Downloader {
 public:
  /// This initializes the internal thread pool with the given number of threads.
  explicit Downloader(size_t threadCount);

  /// Queue a file to be downloaded. If a file with the given name already exists, nothing will be
  /// done. This method will return quickly, as the actual download is done in a separate thread.
  ///  If the path to the destination file does not exist, it will be created.
  void download(std::string const& url, std::string const& file);

  /// Returns the total download progress in percent. If no file was downloaded, it will return 100.
  double getProgress() const;

  /// Returns true when the internal thread pool has no running or pending tasks.
  bool hasFinished() const;

 private:
  ThreadPool mThreadPool;

  mutable std::mutex                     mProgressMutex;
  std::vector<std::pair<double, double>> mProgress;
};

} // namespace cs::utils

#endif // CS_UTILS_DOWNLOADER_HPP
