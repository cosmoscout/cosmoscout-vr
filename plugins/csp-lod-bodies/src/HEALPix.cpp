////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "HEALPix.hpp"

#include "../../../src/cs-utils/convert.hpp"

namespace csp::lodbodies {

////////////////////////////////////////////////////////////////////////////////////////////////////

std::ostream& operator<<(std::ostream& os, EdgeDirection ed) {
  switch (ed) {
  case EdgeDirection::eNorthEast:
    os << "NorthEast";
    break;
  case EdgeDirection::eNorthWest:
    os << "NorthWest";
    break;
  case EdgeDirection::eSouthWest:
    os << "SouthWest";
    break;
  case EdgeDirection::eSouthEast:
    os << "SouthEast";
    break;

    // no default - to get compiler warning when the set of enum values is
    // extended.
  }

  return os;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* static */ std::array<glm::uint16, 256> const HEALPixLevel::sCompressLUT = {{
#define GEN_LUT_0(v) v, (v) + 1, 256 + (v), 256 + (v) + 1
#define GEN_LUT_1(v)                                                                               \
  GEN_LUT_0(v)                                                                                     \
  , GEN_LUT_0((v) + 2), GEN_LUT_0((v) + 512), GEN_LUT_0((v) + 514)
#define GEN_LUT_2(v)                                                                               \
  GEN_LUT_1(v)                                                                                     \
  , GEN_LUT_1((v) + 4), GEN_LUT_1((v) + 1024), GEN_LUT_1((v) + 1028)
    GEN_LUT_2(0), GEN_LUT_2(8), GEN_LUT_2(2048), GEN_LUT_2(2056)
#undef GEN_LUT_0
#undef GEN_LUT_1
#undef GEN_LUT_2
}};

/* static */ std::array<glm::uint64, 256> const HEALPixLevel::sExpandLUT = {{
#define GEN_LUT_0(v) 0x##v##0, 0x##v##1, 0x##v##4, 0x##v##5
#define GEN_LUT_1(v)                                                                               \
  GEN_LUT_0(v##0)                                                                                  \
  , GEN_LUT_0(v##1), GEN_LUT_0(v##4), GEN_LUT_0(v##5)
#define GEN_LUT_2(v)                                                                               \
  GEN_LUT_1(v##0)                                                                                  \
  , GEN_LUT_1(v##1), GEN_LUT_1(v##4), GEN_LUT_1(v##5)
    GEN_LUT_2(0), GEN_LUT_2(1), GEN_LUT_2(4), GEN_LUT_2(5)
#undef GEN_LUT_0
#undef GEN_LUT_1
#undef GEN_LUT_2
}};

////////////////////////////////////////////////////////////////////////////////////////////////////

/* static */ std::array<glm::i64vec2, 4> const HEALPixLevel::sNeighbourOffsetLUT = {{
    glm::i64vec2(1, 0),  // NE
    glm::i64vec2(0, 1),  // NW
    glm::i64vec2(-1, 0), // SW
    glm::i64vec2(0, -1)  // SE
}};

////////////////////////////////////////////////////////////////////////////////////////////////////

/* static */ std::array<glm::ivec4, 12> const HEALPixLevel::sBaseNeighbourLUT = {{// base patch 0
    //         NE  NW  SW  SE
    glm::ivec4(1, 3, 4, 5),
    // base patch 1
    //         NE  NW  SW  SE
    glm::ivec4(2, 0, 5, 6),
    // base patch 2
    //         NE  NW  SW  SE
    glm::ivec4(3, 1, 6, 7),
    // base patch 3
    //         NE  NW  SW  SE
    glm::ivec4(0, 2, 7, 4),

    // base patch 4
    //         NE  NW  SW  SE
    glm::ivec4(0, 3, 11, 8),
    // base patch 5
    //         NE  NW  SW  SE
    glm::ivec4(1, 0, 8, 9),
    // base patch 6
    //         NE  NW  SW  SE
    glm::ivec4(2, 1, 9, 10),
    // base patch 7
    //         NE  NW  SW  SE
    glm::ivec4(3, 2, 10, 11),

    // base patch 8
    //         NE  NW  SW  SE
    glm::ivec4(5, 4, 11, 9),
    // base patch 9
    //         NE  NW  SW  SE
    glm::ivec4(6, 5, 8, 10),
    // base patch 10
    //         NE  NW  SW  SE
    glm::ivec4(7, 6, 9, 11),
    // base patch 11
    //         NE  NW  SW  SE
    glm::ivec4(4, 7, 10, 8)}};

////////////////////////////////////////////////////////////////////////////////////////////////////

/* static */ std::array<glm::uint16, 12> const HEALPixLevel::sF1LUT = {
    {2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4}};

////////////////////////////////////////////////////////////////////////////////////////////////////

/* static */ std::array<glm::uint16, 12> const HEALPixLevel::sF2LUT = {
    {1, 3, 5, 7, 0, 2, 4, 6, 1, 3, 5, 7}};

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::i64vec3 HEALPixLevel::getBaseXY(glm::int64 patchIdx) const {
  // find index of patch relative to base patch (patchIdx mod Nside^2)
  // Nside is a power of 2 so we can just apply a mask
  glm::int64 patchCount  = getPatchCount();
  glm::int64 relPatchIdx = patchIdx & (patchCount - 1);

  glm::int64 x = extractEvenBits(relPatchIdx);
  glm::int64 y = extractOddBits(relPatchIdx);

  return glm::i64vec3(getBasePatch(patchIdx), x, y);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::int64 HEALPixLevel::getPatchIdx(glm::i64vec3 const& bxy) const {
  // shift base patch index to left by 2*level, since each level adds
  // two bits to the patch index.
  // Then add index within base patch which is constructed by
  // taking bxy[1] (aka x) as the even bits and bxy[2] (aka y) as odd bits.

  return (bxy[0] << (2 * mLevel)) + replaceBits(bxy[1], bxy[2]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec2 HEALPixLevel::getCenterLngLat(glm::int64 patchIdx) const {
  glm::i64vec3 bxy = getBaseXY(patchIdx);
  return bxy2geo(bxy);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 HEALPixLevel::getCenterCartesian(
    glm::int64 patchIdx, double radiusE, double radiusP) const {
  return cs::utils::convert::toCartesian(getCenterLngLat(patchIdx), radiusE, radiusP);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::array<glm::dvec2, 4> HEALPixLevel::getCornersLngLat(glm::int64 patchIdx) const {
  std::array<glm::dvec2, 4> result{};

  glm::i64vec3 bxy = getBaseXY(patchIdx);
  double       cx  = (bxy[1] + 0.5) / mNSide;
  double       cy  = (bxy[2] + 0.5) / mNSide;
  double       d   = 0.5 / mNSide;

  result[0] = HEALPix::convertBaseXY2LngLat(static_cast<int>(bxy[0]), cx + d, cy + d);
  result[1] = HEALPix::convertBaseXY2LngLat(static_cast<int>(bxy[0]), cx - d, cy + d);
  result[2] = HEALPix::convertBaseXY2LngLat(static_cast<int>(bxy[0]), cx - d, cy - d);
  result[3] = HEALPix::convertBaseXY2LngLat(static_cast<int>(bxy[0]), cx + d, cy - d);

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::array<glm::dvec3, 4> HEALPixLevel::getCornersCartesian(
    glm::int64 patchIdx, double radiusE, double radiusP) const {
  std::array<glm::dvec3, 4> result{};
  std::array<glm::dvec2, 4> corners = getCornersLngLat(patchIdx);

  result[0] = cs::utils::convert::toCartesian(corners[0], radiusE, radiusP);
  result[1] = cs::utils::convert::toCartesian(corners[1], radiusE, radiusP);
  result[2] = cs::utils::convert::toCartesian(corners[2], radiusE, radiusP);
  result[3] = cs::utils::convert::toCartesian(corners[3], radiusE, radiusP);

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::array<glm::dvec2, 4> HEALPixLevel::getEdgeCentersLngLat(glm::int64 patchIdx) const {
  std::array<glm::dvec2, 4> result{};

  glm::i64vec3 bxy = getBaseXY(patchIdx);
  double       cx  = (bxy[1] + 0.5) / mNSide;
  double       cy  = (bxy[2] + 0.5) / mNSide;
  double       d   = 0.5 / mNSide;

  result[0] = HEALPix::convertBaseXY2LngLat(static_cast<int>(bxy[0]), cx + d, cy);
  result[1] = HEALPix::convertBaseXY2LngLat(static_cast<int>(bxy[0]), cx, cy + d);
  result[2] = HEALPix::convertBaseXY2LngLat(static_cast<int>(bxy[0]), cx - d, cy);
  result[3] = HEALPix::convertBaseXY2LngLat(static_cast<int>(bxy[0]), cx, cy - d);

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::array<glm::dvec3, 4> HEALPixLevel::getEdgeCentersCartesian(
    glm::int64 patchIdx, double radiusE, double radiusP) const {
  std::array<glm::dvec3, 4> result{};
  std::array<glm::dvec2, 4> edges = getEdgeCentersLngLat(patchIdx);

  result[0] = cs::utils::convert::toCartesian(edges[0], radiusE, radiusP);
  result[1] = cs::utils::convert::toCartesian(edges[1], radiusE, radiusP);
  result[2] = cs::utils::convert::toCartesian(edges[2], radiusE, radiusP);
  result[3] = cs::utils::convert::toCartesian(edges[3], radiusE, radiusP);

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::array<glm::int64, 4> HEALPixLevel::getNeighbours(glm::int64 patchIdx) const {
  std::array<glm::int64, 4> result{};

  glm::i64vec3 bxy = getBaseXY(patchIdx);

  for (std::size_t i = 0; i < result.size(); ++i) {
    // neighbour bxy
    glm::i64vec3 bxyN = bxy;

    bxyN[1] += sNeighbourOffsetLUT.at(i)[0];
    bxyN[2] += sNeighbourOffsetLUT.at(i)[1];

    if (bxyN[1] < 0) {
      // crossed to SW neighbour base patch
      bxyN[0] = sBaseNeighbourLUT.at(bxy[0])[static_cast<int>(i)];

      if (bxy[0] >= 8) {
        bxyN[1] = bxyN[2];
        bxyN[2] = 0;
      } else {
        bxyN[1] = mNSide - 1;
      }
    } else if (bxyN[1] >= mNSide) {
      // crosed to NE neighbour base patch
      bxyN[0] = sBaseNeighbourLUT.at(bxy[0])[static_cast<int>(i)];

      if (bxy[0] < 4) {
        bxyN[1] = bxyN[2];
        bxyN[2] = mNSide - 1;
      } else {
        bxyN[1] = 0;
      }
    } else if (bxyN[2] < 0) {
      // crossed to SE neighbour base patch
      bxyN[0] = sBaseNeighbourLUT.at(bxy[0])[static_cast<int>(i)];

      if (bxy[0] >= 8) {
        bxyN[2] = bxyN[1];
        bxyN[1] = 0;
      } else {
        bxyN[2] = mNSide - 1;
      }
    } else if (bxyN[2] >= mNSide) {
      // crossed to NW neighbour base patch
      bxyN[0] = sBaseNeighbourLUT.at(bxy[0])[static_cast<int>(i)];

      if (bxy[0] < 4) {
        bxyN[2] = bxyN[1];
        bxyN[1] = mNSide - 1;
      } else {
        bxyN[2] = 0;
      }
    }

    result.at(i) = getPatchIdx(bxyN);
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int HEALPixLevel::getF1(glm::int64 patchIdx) const {
  glm::int64 patchCount = getPatchCount();
  glm::int64 bpatchIdx  = patchIdx / patchCount;

  return sF1LUT.at(bpatchIdx);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int HEALPixLevel::getF2(glm::int64 patchIdx) const {
  glm::int64 patchCount = getPatchCount();
  glm::int64 bpatchIdx  = patchIdx / patchCount;

  return sF2LUT.at(bpatchIdx);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec3 HEALPixLevel::getPatchOffsetScale(glm::int64 patchIdx) const {
  glm::dvec3   result;
  glm::i64vec3 bxy = getBaseXY(patchIdx);

  result[0] = static_cast<double>(bxy[1]) / mNSide;
  result[1] = static_cast<double>(bxy[2]) / mNSide;
  result[2] = 1.0 / mNSide;

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* explicit */
HEALPixLevel::HEALPixLevel(int level)
    : mLevel(level)
    , mNSide(int64_t(1) << level) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::dvec2 HEALPixLevel::bxy2geo(glm::i64vec3 const& bxy) const {
  return HEALPix::convertBaseXY2LngLat(
      static_cast<int>(bxy[0]), (bxy[1] + 0.5) / mNSide, (bxy[2] + 0.5) / mNSide);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* static */ std::array<HEALPixLevel, 20> const HEALPix::sLevels = {
    {HEALPixLevel(0), HEALPixLevel(1), HEALPixLevel(2), HEALPixLevel(3), HEALPixLevel(4),
        HEALPixLevel(5), HEALPixLevel(6), HEALPixLevel(7), HEALPixLevel(8), HEALPixLevel(9),
        HEALPixLevel(10), HEALPixLevel(11), HEALPixLevel(12), HEALPixLevel(13), HEALPixLevel(14),
        HEALPixLevel(15), HEALPixLevel(16), HEALPixLevel(17), HEALPixLevel(18), HEALPixLevel(19)}};

////////////////////////////////////////////////////////////////////////////////////////////////////

int HEALPixLevel::getLevel() const {
  return mLevel;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::int64 HEALPixLevel::getNSide() const {
  return mNSide;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::int64 HEALPixLevel::getPatchCount() const {
  return mNSide * mNSide;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::int64 HEALPixLevel::getTotalPatchCount() const {
  int64_t const maxPatches = 12;
  return maxPatches * getPatchCount();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int HEALPixLevel::getBasePatch(glm::int64 patchIdx) const {
  return static_cast<int>(patchIdx / getPatchCount());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::int64 HEALPixLevel::extractEvenBits(glm::int64 value) {
  // Value has binary representation: xwvu tsrq ponm lkji hgfe dcba
  // where each letter is a binary digit (i.e. either 0 or 1).
  // We want the even bits:           .... .... .... wusq omki geca

  // 1) Mask all odd bits:            0w0u 0s0q 0o0m 0k0i 0g0e 0c0a
  uint64_t const oddBitMask = 0x5555555555555555ULL;
  glm::int64     tmp        = value & oddBitMask;

  // 2) Shift high bytes down to replace the zeros.
  //                                  .w.u .s.q .o.m .k.i wgue scqa
  //                                                      ^ ^  ^ ^
  tmp |= tmp >> 15;

  // Below we use a 8 bit look-up table to convert "wgue scqa" to
  // "0000 wusq 0000 geca" which places the bits contained in the lowest
  // byte at their correct positions (compare with desired output above).
  // So the lowest 8 bits produce bits 9-12 and 0-4 in the output, similarly
  // the next 8 bit produce output bits 13-16 and 5-8 and so on.
  // Therefore the desired output is found with 4 table lookups and OR'ing
  // the results together.
  return sCompressLUT.at(tmp & 0xFF) | (sCompressLUT.at((tmp >> 8) & 0xFF) << 4) |
         (sCompressLUT.at((tmp >> 32) & 0xFF) << 16) | (sCompressLUT.at((tmp >> 40) & 0xFF) << 20);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::int64 HEALPixLevel::extractOddBits(glm::int64 value) {
  return extractEvenBits(value >> 1);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void HEALPixLevel::extractBits(glm::int64 value, glm::int64& evenBits, glm::int64& oddBits) {
  evenBits = extractEvenBits(value);
  oddBits  = extractOddBits(value);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::int64 HEALPixLevel::replaceEvenBits(glm::int64 value) {
  return sExpandLUT.at(value & 0xFF) | sExpandLUT.at((value >> 8) & 0xFF) << 16 |
         sExpandLUT.at((value >> 16) & 0xFF) << 32 | sExpandLUT.at((value >> 24) & 0xFF) << 48;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::int64 HEALPixLevel::replaceOddBits(glm::int64 value) {
  return replaceEvenBits(value) << 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

glm::int64 HEALPixLevel::replaceBits(glm::int64 evenBits, glm::int64 oddBits) {
  return replaceEvenBits(evenBits) | replaceOddBits(oddBits);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* static */ HEALPixLevel const& HEALPix::getLevel(int level) {
  return sLevels.at(level);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* static */ glm::int64 HEALPix::getNSide(TileId const& tileId) {
  return getLevel(tileId.level()).getNSide();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* static */ int HEALPix::getBasePatch(TileId const& tileId) {
  return getLevel(tileId.level()).getBasePatch(tileId.patchIdx());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* static */ glm::i64vec3 HEALPix::getBaseXY(TileId const& tileId) {
  return getLevel(tileId.level()).getBaseXY(tileId.patchIdx());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* static */ glm::dvec2 HEALPix::getCenterLngLat(TileId const& tileId) {
  return getLevel(tileId.level()).getCenterLngLat(tileId.patchIdx());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* static */ glm::dvec3 HEALPix::getCenterCartesian(
    TileId const& tileId, double radiusE, double radiusP) {
  return getLevel(tileId.level()).getCenterCartesian(tileId.patchIdx(), radiusE, radiusP);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* static */ std::array<glm::dvec2, 4> HEALPix::getCornersLngLat(TileId const& tileId) {
  return getLevel(tileId.level()).getCornersLngLat(tileId.patchIdx());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* static */ std::array<glm::dvec3, 4> HEALPix::getCornersCartesian(
    TileId const& tileId, double radiusE, double radiusP) {
  return getLevel(tileId.level()).getCornersCartesian(tileId.patchIdx(), radiusE, radiusP);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* static */ std::array<glm::dvec2, 4> HEALPix::getEdgeCentersLngLat(TileId const& tileId) {
  return getLevel(tileId.level()).getEdgeCentersLngLat(tileId.patchIdx());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* static */ std::array<glm::dvec3, 4> HEALPix::getEdgeCentersCartesian(
    TileId const& tileId, double radiusE, double radiusP) {
  return getLevel(tileId.level()).getEdgeCentersCartesian(tileId.patchIdx(), radiusE, radiusP);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* static */ std::array<glm::int64, 4> HEALPix::getNeighbours(TileId const& tileId) {
  return getLevel(tileId.level()).getNeighbours(tileId.patchIdx());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* static */ std::array<TileId, 4> HEALPix::getNeighbourIds(TileId const& tileId) {
  std::array<TileId, 4>     result;
  std::array<glm::int64, 4> neighbours = getNeighbours(tileId);

  for (std::size_t i = 0; i < neighbours.size(); ++i) {
    result.at(i) = TileId(tileId.level(), neighbours.at(i));
  }

  return result;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* static */ int HEALPix::getF1(TileId const& tileId) {
  return getLevel(tileId.level()).getF1(tileId.patchIdx());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* static */ int HEALPix::getF2(TileId const& tileId) {
  return getLevel(tileId.level()).getF2(tileId.patchIdx());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* static */ glm::dvec3 HEALPix::getPatchOffsetScale(TileId const& tileId) {
  return getLevel(tileId.level()).getPatchOffsetScale(tileId.patchIdx());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* static */ glm::dvec2 HEALPix::convertBaseXY2LngLat(int basePatchIdx, double x, double y) {
  glm::dvec2 lngLat;
  auto const pi = glm::pi<double>();
  double     jr = HEALPixLevel::sF1LUT.at(basePatchIdx) - x - y;
  double     nr{};

  if (jr < 1.0) {
    nr        = jr;
    lngLat[1] = 1.0 - (nr * nr / 3.0);
  } else if (jr > 3.0) {
    nr        = 4.0 - jr;
    lngLat[1] = (nr * nr / 3.0) - 1.0;
  } else {
    nr        = 1.0;
    lngLat[1] = (2.0 - jr) * 2.0 / 3.0;
  }

  double tmp = HEALPixLevel::sF2LUT.at(basePatchIdx) * nr + x - y;
  tmp        = (tmp < 0.0) ? tmp + 8.0 : tmp;
  tmp        = (tmp >= 8.0) ? tmp - 8.0 : tmp;

  lngLat[0] = ((nr < 1e-15) ? 0.0 : (0.25 * pi * tmp) / nr) - pi;
  lngLat[1] = 0.5 * pi - std::acos(lngLat[1]);

  return lngLat;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* static */ glm::dvec2 HEALPix::convertBaseLngLat2XY(int basePatchIdx, glm::dvec2 const& lngLat) {
  double _f1 = HEALPixLevel::sF1LUT.at(basePatchIdx);
  double _f2 = HEALPixLevel::sF2LUT.at(basePatchIdx);

  double x = std::fmod(lngLat.x / glm::pi<double>() + 1.0, 2.0); //!< Ranges from  0 to 2
  double y = std::sin(lngLat.y);                                 //!< Ranges from -1 to 1

  double ySep = 2.0 / 3.0;

  double h{};
  double v{};

  // north polar cap (row 0)
  if (y >= ySep) {
    // Only 0 - 3
    assert(basePatchIdx < 4);

    double i = sqrt((1.0 - y) * 3.0);
    double j = 2 * x * i + 0.5;

    h = 2 * j - _f2 * i - 1;

    v = _f1 - i - 1;
  }
  // south polar cap (row 2)
  else if (y < -ySep) {
    // Only 8 - 11
    assert(basePatchIdx > 7);

    double i2 = sqrt(3.0 * (1.0 + y));

    h = 4 * x * i2 - _f2 * i2;
    v = _f1 + i2 - 5.0;
  }
  // equatorial belt (row 1)
  else {
    double i = (2.0 / 3.0 - y / 2.0) * 3;

    v = _f1 - i - 1;

    // Wrap Arround for Meridian Patch
    if (basePatchIdx == 4 && x > 0.25) {
      x -= 2.0;
    }

    h = 4 * x - _f2;
  }

  glm::dvec2 relative;
  relative.y = (v - h) / 2.0;
  relative.x = v - relative.y;

  relative.x += 0.5;
  relative.y += 0.5;

  return relative;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int HEALPix::convertLngLat2Base(glm::dvec2 const& lngLat) {
  double x = std::fmod(lngLat.x / glm::pi<double>() + 1, 2.0); //!< Ranges from  0 to 2
  double y = std::sin(lngLat.y);                               //!< Ranges from -1 to 1

  const double ySep  = 2.0 / 3.0;   //!< separation height (tip of diamonds)
  const double slope = ySep / 0.25; //!< slope of diamond edges

  // decide which longitude-octant the point is in
  int octant = 0;

  while (x > (octant + 1) * 0.25) {
    octant++;
  }

  // north polar cap (row 0)
  if (y >= ySep) {
    return octant / 2;
  }

  // south polar cap (row 2)
  if (y < -ySep) {
    return 8 + octant / 2;
  }

  // equatorial belt (row 1)
  x -= 0.25 * octant; //!< normalize X

  if (octant % 2 == 0) {
    if (y > ySep - x * slope) //!< north patch
    {
      return octant / 2;
    }
    if (y < -ySep + x * slope) //!< south patch
    {
      return 8 + octant / 2;
    }
  } else {
    if (y > x * slope) //!< north patch
    {
      return static_cast<int32_t>(octant * 0.5);
    }
    if (y < -x * slope) //!< south patch
    {
      return static_cast<int32_t>(8 + octant * 0.5);
    }
  }

  // diamond at equator
  return static_cast<int32_t>(4 + (octant % 7 + 1) * 0.5);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* static */ int HEALPix::getRootIdx(TileId const& tileId) {
  int const        lvl = tileId.level();
  glm::int64 const idx = tileId.patchIdx();

  return static_cast<int>((idx >> (lvl * 2)) & 0x0F);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* static */ int HEALPix::getChildIdxAtLevel(TileId const& tileId, int level) {
  if (level == 0) {
    return getRootIdx(tileId);
  }

  int const        lvl = tileId.level();
  glm::int64 const idx = tileId.patchIdx();
  assert(level <= lvl); // check that user asked for a parent

  return static_cast<int>((idx >> ((lvl - level) * 2)) & 0x03);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* static */ int HEALPix::getChildIdx(TileId const& tileId) {
  return getChildIdxAtLevel(tileId, tileId.level());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* static */ int HEALPix::getChildLevel(TileId const& parentId) {
  return parentId.level() + 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* static */ glm::int64 HEALPix::getChildPatchIdx(TileId const& parentId, int childIdx) {
  return parentId.patchIdx() * 4 + childIdx;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* static */ TileId HEALPix::getChildTileId(TileId const& parentId, int childIdx) {
  return TileId(getChildLevel(parentId), getChildPatchIdx(parentId, childIdx));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* static */ int HEALPix::getParentLevel(TileId const& childId) {
  return childId.level() - 1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* static */ glm::int64 HEALPix::getParentPatchIdx(TileId const& childId) {
  return childId.patchIdx() / 4;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* static */ TileId HEALPix::getParentTileId(TileId const& childId) {
  return TileId(getParentLevel(childId), getParentPatchIdx(childId));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::lodbodies
