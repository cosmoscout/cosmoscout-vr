////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_LOD_BODIES_TILEDATATYPE_HPP
#define CSP_LOD_BODIES_TILEDATATYPE_HPP

#include <array>

namespace csp::lodbodies {

/// Contains an enumeration of data types that can be stored in a tile.
enum class TileDataType { eElevation = 0, eColor = 1 };

/// Update this to reflect the number of items in the enum above.
const std::size_t TileDataTypeCount = 2;

/// This is used in various places where a thing needs to be stored once for each supported data
/// channel. For instance, the TileNode stores TileData per data type with such a struct.
template <typename T>
struct PerDataType {
  T& get(TileDataType type) {
    return mChannels.at(static_cast<int>(type));
  }

  T const& get(TileDataType type) const {
    return mChannels.at(static_cast<int>(type));
  }

  void set(TileDataType type, T const& value) {
    mChannels.at(static_cast<int>(type)) = value;
  }

  void set(TileDataType type, T&& value) {
    mChannels.at(static_cast<int>(type)) = std::move(value);
  }

  std::array<T, TileDataTypeCount> mChannels{};
};

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_TILEDATATYPE_HPP
