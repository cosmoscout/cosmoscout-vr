////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CS_UTILS_PARALLLEL_HPP
#define CS_UTILS_PARALLLEL_HPP

#include "cs_utils_export.hpp"
#include <functional>
#include <thread>

namespace cs::utils {

/// Executes function f count times in parallel and gives it the current execution index. It is
/// slightly faster than a thread pool, but also less flexible.
void CS_UTILS_EXPORT parallelFor(size_t count, std::function<void(size_t)> const& f);
void CS_UTILS_EXPORT parallelFor(size_t count, size_t numThreads, std::function<void(size_t)> const& f);

}

#endif // CS_UTILS_PARALLLEL_HPP
