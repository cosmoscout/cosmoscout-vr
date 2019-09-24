////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_UTILS_FILESYSTEM_HPP
#define CS_UTILS_FILESYSTEM_HPP

#include "cs_utils_export.hpp"

#include <boost/filesystem.hpp>
#include <set>
#include <string>

/// Utility functions for all sorts of stuff.
namespace cs::utils::filesystem {

/// Creates all required directories for @param path if they do not exist
CS_UTILS_EXPORT void createDirectoryRecursively(boost::filesystem::path const& path,
    boost::filesystem::perms permissions = boost::filesystem::perms::owner_all |
                                           boost::filesystem::perms::group_read |
                                           boost::filesystem::perms::others_read);

/// Lists all files in the given directory.
CS_UTILS_EXPORT std::set<std::string> listFiles(std::string const& directory);

/// Returns the contents of the file as a string. Any occurrences of \r\n will be replaced by \n.
CS_UTILS_EXPORT std::string loadToString(std::string const& file);

/// Downloads a file from te internet. This call will block until the file is downloaded
/// successfully or an error occurred. If the path to the destination file does not exist, it will
/// be created. This will throw a std::runtime_error if something bad happend.
CS_UTILS_EXPORT void downloadFile(
    std::string const& url, std::string const& destination, bool printProgress);

} // namespace cs::utils::filesystem

#endif // CS_UTILS_FILESYSTEM_HPP
