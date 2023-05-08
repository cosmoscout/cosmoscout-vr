////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_LOD_BODIES_TILE_DATA_HPP
#define CSP_LOD_BODIES_TILE_DATA_HPP

#include "TileDataBase.hpp"

namespace csp::lodbodies {

/// Concrete class storing data samples of the template argument type T.
template <typename T>
class TileData : public TileDataBase {
 public:
  using value_type = T;

  explicit TileData(uint32_t resolution);

  TileData(TileData const& other) = delete;
  TileData(TileData&& other)      = delete;

  TileData& operator=(TileData const& other) = delete;
  TileData& operator=(TileData&& other) = delete;

  ~TileData() override;

  static TileDataType getStaticDataType();

  TileDataType getDataType() const override;

  void const* getDataPtr() const override;
  void*       getDataPtr() override;

  std::vector<T> const& data() const;
  std::vector<T>&       data();

 private:
  std::vector<T> mData;
};

namespace detail {

/// DataTypeTrait<T> is used to map from a type T to the corresponding TileDataType enum value.
/// To support additional data types stored in a TileData add a specialization. Do not forget to add
/// a definition of the static member in TileData.cpp! Only declare base template, define explicit
/// specializations for supported types below - this causes a convenient compile error if an attempt
/// is made to instantiate TileData<T> with an unsupported type T
template <typename T>
struct DataTypeTrait;

template <>
struct DataTypeTrait<float> {
  static TileDataType const value = TileDataType::eElevation;
};

template <>
struct DataTypeTrait<glm::u8vec4> {
  static TileDataType const value = TileDataType::eColor;
};
} // namespace detail

template <typename T>
TileData<T>::TileData(uint32_t resolution)
    : TileDataBase(resolution)
    , mData(resolution * resolution) {
}

template <typename T>
TileData<T>::~TileData() = default;

template <typename T>
TileDataType TileData<T>::getStaticDataType() {
  return detail::DataTypeTrait<T>::value;
}

template <typename T>
TileDataType TileData<T>::getDataType() const {
  return getStaticDataType();
}

template <typename T>
void const* TileData<T>::getDataPtr() const {
  return static_cast<void const*>(mData.data());
}

template <typename T>
void* TileData<T>::getDataPtr() {
  return static_cast<void*>(mData.data());
}

template <typename T>
std::vector<T> const& TileData<T>::data() const {
  return mData;
}

template <typename T>
std::vector<T>& TileData<T>::data() {
  return mData;
}

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_TILE_DATA_HPP
