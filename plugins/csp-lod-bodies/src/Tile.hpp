////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_LOD_BODIES_TILE_HPP
#define CSP_LOD_BODIES_TILE_HPP

#include "TileBase.hpp"

namespace csp::lodbodies {

/// Concrete class storing data samples of the template argument type T.
template <typename T>
class Tile : public TileBase {
 public:
  typedef std::array<T, TileBase::SizeX * TileBase::SizeY> Storage;
  using value_type = T;

  explicit Tile(int level, glm::int64 patchIdx);

  Tile(Tile const& other) = delete;
  Tile(Tile&& other)      = delete;

  Tile& operator=(Tile const& other) = delete;
  Tile& operator=(Tile&& other) = delete;

  ~Tile() override;

  static std::type_info const& getStaticTypeId();
  static TileDataType          getStaticDataType();

  std::type_info const& getTypeId() const override;
  TileDataType          getDataType() const override;

  void const* getDataPtr() const override;

  Storage const& data() const;
  Storage&       data();

 private:
  Storage mData;
};

namespace detail {

/// DataTypeTrait<T> is used to map from a type T to the corresponding TileDataType enum value.
/// To support additional data types stored in a Tile add a specialization. Do not forget to add a
/// definition of the static member in Tile.cpp! Only declare base template, define explicit
/// specializations for supported types below - this causes a convenient compile error if an attempt
/// is made to instantiate Tile<T> with an unsupported type T
template <typename T>
struct DataTypeTrait;

template <>
struct DataTypeTrait<float> {
  static TileDataType const value = TileDataType::eFloat32;
};

template <>
struct DataTypeTrait<glm::uint8> {
  static TileDataType const value = TileDataType::eUInt8;
};

template <>
struct DataTypeTrait<glm::u8vec3> {
  static TileDataType const value = TileDataType::eU8Vec3;
};
} // namespace detail

template <typename T>
Tile<T>::Tile(int level, glm::int64 patchIdx)
    : TileBase(level, patchIdx)
    , mData() {
}

template <typename T>
Tile<T>::~Tile() = default;

template <typename T>
std::type_info const& Tile<T>::getStaticTypeId() {
  return typeid(T);
}

template <typename T>
TileDataType Tile<T>::getStaticDataType() {
  return detail::DataTypeTrait<T>::value;
}

template <typename T>
std::type_info const& Tile<T>::getTypeId() const {
  return getStaticTypeId();
}

template <typename T>
TileDataType Tile<T>::getDataType() const {
  return getStaticDataType();
}

template <typename T>
void const* Tile<T>::getDataPtr() const {
  return static_cast<void const*>(mData.data());
}

template <typename T>
typename Tile<T>::Storage const& Tile<T>::data() const {
  return mData;
}

template <typename T>
typename Tile<T>::Storage& Tile<T>::data() {
  return mData;
}

} // namespace csp::lodbodies

#endif // CSP_LOD_BODIES_TILE_HPP
