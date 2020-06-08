////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "TileSourceWebMapService.hpp"

#include "HEALPix.hpp"
#include "TileNode.hpp"
#include "logger.hpp"

#include "../../../src/cs-utils/filesystem.hpp"

#include <boost/filesystem.hpp>
#include <curlpp/Easy.hpp>
#include <curlpp/Info.hpp>
#include <curlpp/Infos.hpp>
#include <curlpp/Options.hpp>
#include <curlpp/cURLpp.hpp>
#include <fstream>
#include <sstream>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image.h>

#include <tiffio.h>

namespace csp::lodbodies {

////////////////////////////////////////////////////////////////////////////////////////////////////

namespace {

////////////////////////////////////////////////////////////////////////////////////////////////////

enum class CopyPixels { eAll, eAboveDiagonal, eBelowDiagonal };

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
bool loadImpl(
    TileSourceWebMapService* source, TileNode* node, int level, int x, int y, CopyPixels which) {
  auto        tile = static_cast<Tile<T>*>(node->getTile());
  std::string cacheFile;

  try {
    cacheFile = source->loadData(level, x, y);
  } catch (std::exception const& e) {
    logger().error("Tile loading failed: {}", e.what());
    return false;
  }

  if (tile->getDataType() == TileDataType::eFloat32) {
    TIFFSetWarningHandler(nullptr);
    auto* data = TIFFOpen(cacheFile.c_str(), "r");
    if (!data) {
      logger().error("Tile loading failed: Cannot open '{}' with libtiff!", cacheFile);
      return false;
    }

    int imagelength{};
    TIFFGetField(data, TIFFTAG_IMAGELENGTH, &imagelength);
    for (int y = 0; y < imagelength; y++) {
      if (which == CopyPixels::eAll) {
        TIFFReadScanline(data, &tile->data()[257 * y], y);
      } else if (which == CopyPixels::eAboveDiagonal) {
        std::array<float, 257> tmp{};
        TIFFReadScanline(data, tmp.data(), y);
        int offset = 257 * y;
        int count  = 257 - y - 1;

        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        std::memcpy(tile->data().data() + offset, tmp.data(), count * sizeof(float));
      } else if (which == CopyPixels::eBelowDiagonal) {
        std::array<float, 257> tmp{};
        TIFFReadScanline(data, tmp.data(), y);
        int offset = 257 * y + (257 - y);
        int count  = y;

        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        std::memcpy(tile->data().data() + offset, tmp.data() + 257 - y, count * sizeof(float));
      }
    }
    TIFFClose(data);
  } else {
    int width{};
    int height{};
    int bpp{};
    int channels = tile->getDataType() == TileDataType::eU8Vec3 ? 3 : 1;

    auto* data =
        reinterpret_cast<T*>(stbi_load(cacheFile.c_str(), &width, &height, &bpp, channels));

    if (!data) {
      logger().error("Tile loading failed: Cannot open '{}' with stbi!", cacheFile);
      return false;
    }

    if (which == CopyPixels::eAll) {
      std::memcpy(tile->data().data(), data, channels * width * height);
    } else if (which == CopyPixels::eAboveDiagonal) {
      for (int y = 0; y < height; ++y) {
        int offset = width * y;
        int count  = channels * (width - y - 1);

        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        std::memcpy(tile->data().data() + offset, data + offset, count);
      }
    } else if (which == CopyPixels::eBelowDiagonal) {
      for (int y = 0; y < height; ++y) {
        int offset = width * y + (width - y);
        int count  = channels * y;

        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        std::memcpy(tile->data().data() + offset, data + offset, count);
      }
    }

    stbi_image_free(data);
  }

  return true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
void fillDiagonal(TileNode* node) {
  auto tile = static_cast<Tile<T>*>(node->getTile());
  for (int y = 1; y <= 257; y++) {
    int pixelPos           = y * (257 - 1);
    tile->data()[pixelPos] = (y < 257) ? tile->data()[pixelPos - 1] : tile->data()[pixelPos + 1];
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
TileNode* loadImpl(TileSourceWebMapService* source, uint32_t level, glm::int64 patchIdx) {
  auto* node = new TileNode(); // NOLINT(cppcoreguidelines-owning-memory): TODO this is bad!

  node->setTile(std::make_unique<Tile<T>>(level, patchIdx));
  node->setChildMaxLevel(std::min(level + 1, source->getMaxLevel()));

  int  x{};
  int  y{};
  bool onDiag = csp::lodbodies::TileSourceWebMapService::getXY(level, patchIdx, x, y);
  if (onDiag) {
    if (!loadImpl<T>(source, node, level, x, y, CopyPixels::eBelowDiagonal)) {
      delete node; // NOLINT(cppcoreguidelines-owning-memory): TODO this is bad!
      return nullptr;
    }

    x += 4 * (1 << level);
    y -= 4 * (1 << level);

    if (!loadImpl<T>(source, node, level, x, y, CopyPixels::eAboveDiagonal)) {
      delete node; // NOLINT(cppcoreguidelines-owning-memory): TODO this is bad!
      return nullptr;
    }

    fillDiagonal<T>(node);
  } else {
    if (!loadImpl<T>(source, node, level, x, y, CopyPixels::eAll)) {
      delete node; // NOLINT(cppcoreguidelines-owning-memory): TODO this is bad!
      return nullptr;
    }
  }

  // TODO: the NE and NW edges of all tiles should contain the values of the
  // respective neighbours (for tile stiching). This is done by increasing the
  // bounding box of the request by one pixel - this works more or less in the
  // general case, but it doesn't when we are at a base patch border of the
  // northern hemisphere. In this case we will get empty pixels! Therefore we
  // fill the last column and row by copying.
  // The proper solution would load the real neighbouring tiles and copy the
  // pixel values!

  auto         tile   = static_cast<Tile<T>*>(node->getTile());
  glm::i64vec3 baseXY = HEALPix::getBaseXY(TileId(level, patchIdx));
  glm::int64   nSide  = HEALPix::getNSide(TileId(level, patchIdx));

  // northern hemisphere
  if (baseXY.x < 4) {
    // at north west boundary of base patch
    if (baseXY.z == nSide - 1) {
      // copy second pixel row to first
      for (int i = 0; i < 257; i++) {
        tile->data()[i + 257] = tile->data()[i + 257 * 2];
        tile->data()[i]       = tile->data()[i + 257 * 2];
      }
    }

    // at north east boundary of base patch
    if (baseXY.y == nSide - 1) {
      // copy last pixel column to last but one
      for (int i = 0; i < 257; i++) {
        tile->data()[i * 257 + 255] = tile->data()[i * 257 + 254];
        tile->data()[i * 257 + 256] = tile->data()[i * 257 + 254];
      }
    }
  }

  // flip y --- that shouldn't be requiered, but somehow is how it was
  // implemented in the original databases
  for (int i = 0; i < 257 / 2; i++) {
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    std::swap_ranges(tile->data().data() + i * 257, tile->data().data() + (i + 1) * 257,
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        tile->data().data() + (256 - i) * 257);
  }

  if (tile->getDataType() == TileDataType::eFloat32) {
    // Creating a MinMaxPyramid alongside the sampling beginning with a resolution of
    // 128x128
    // The MinMaxPyramid is later needed to deduce height information from this
    // coarser level DEM tile to deeper level IMG tiles
    auto* demTile = reinterpret_cast<Tile<float>*>(tile);
    demTile->setMinMaxPyramid(std::make_unique<MinMaxPyramid>(demTile));
  }

  return node;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace

////////////////////////////////////////////////////////////////////////////////////////////////////

std::mutex TileSourceWebMapService::mTileSystemMutex;

////////////////////////////////////////////////////////////////////////////////////////////////////

TileSourceWebMapService::TileSourceWebMapService()
    : mThreadPool(32) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* virtual */ TileNode* TileSourceWebMapService::loadTile(int level, glm::int64 patchIdx) {
  if (mFormat == TileDataType::eFloat32) {
    return loadImpl<float>(this, level, patchIdx);
  }
  if (mFormat == TileDataType::eUInt8) {
    return loadImpl<glm::uint8>(this, level, patchIdx);
  }
  if (mFormat == TileDataType::eU8Vec3) {
    return loadImpl<glm::u8vec3>(this, level, patchIdx);
  }

  throw std::domain_error(fmt::format("Unsupported format: {}!", mFormat));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool TileSourceWebMapService::getXY(int level, glm::int64 patchIdx, int& x, int& y) {
  std::array<glm::ivec2, 12> basePatchExtends = {glm::ivec2(1, 4), glm::ivec2(2, 3),
      glm::ivec2(3, 2), glm::ivec2(4, 1), glm::ivec2(0, 4), glm::ivec2(1, 3), glm::ivec2(2, 2),
      glm::ivec2(3, 1), glm::ivec2(0, 3), glm::ivec2(1, 2), glm::ivec2(2, 1), glm::ivec2(3, 0)};

  glm::i64vec3 baseXY = HEALPix::getBaseXY(TileId(level, patchIdx));

  x = static_cast<int32_t>(basePatchExtends.at(baseXY[0])[0] * (1 << level) + baseXY[1]);
  y = static_cast<int32_t>(basePatchExtends.at(baseXY[0])[1] * (1 << level) + baseXY[2]);

  if (basePatchExtends.at(baseXY[0])[0] == 0 && basePatchExtends.at(baseXY[0])[1] == 4) {
    // check if tile is located above the diagonal
    if (y > x + 4 * (1 << level)) {
      x += 4 * (1 << level);
      y -= 4 * (1 << level);
    }
    // check if tile is crossed by diagonal
    else if (y == x + 4 * (1 << level)) {
      return true;
    }
  }

  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string TileSourceWebMapService::loadData(int level, int x, int y) {

  std::string format;
  std::string type;

  if (mFormat == TileDataType::eFloat32) {
    format = "tiffGray";
    type   = "tiff";
  } else if (mFormat == TileDataType::eU8Vec3) {
    format = "pngRGB";
    type   = "png";
  } else {
    format = "pngGray";
    type   = "png";
  }

  std::stringstream cacheDir;
  cacheDir << mCache << "/" << mLayers << "/" << level << "/" << x;

  std::stringstream cacheFile(cacheDir.str());
  cacheFile << cacheDir.str() << "/" << y << "." << type;
  std::stringstream url;

  double size = 1.0 / (1 << level);

  url.precision(std::numeric_limits<double>::max_digits10);
  url << mUrl << "&version=1.1.0&request=GetMap&tiled=true&layers=" << mLayers
      << "&bbox=" << x * size << "," << y * size << "," << x * size + size << "," << y * size + size
      << "&width=257&height=257&srs=EPSG:900914&format=" << format;

  auto cacheFilePath(boost::filesystem::path(cacheFile.str()));

  // the file is already there, we can return it
  if (boost::filesystem::exists(cacheFilePath) &&
      boost::filesystem::file_size(cacheFile.str()) > 0) {
    return cacheFile.str();
  }

  // the file is corrupt not available
  {
    std::unique_lock<std::mutex> lock(mTileSystemMutex);

    if (boost::filesystem::exists(cacheFilePath) &&
        boost::filesystem::file_size(cacheFile.str()) == 0) {
      boost::filesystem::remove(cacheFilePath);
    }

    auto cacheDirPath(boost::filesystem::absolute(boost::filesystem::path(cacheDir.str())));
    if (!(boost::filesystem::exists(cacheDirPath))) {
      try {
        cs::utils::filesystem::createDirectoryRecursively(
            cacheDirPath, boost::filesystem::perms::all_all);
      } catch (std::exception& e) {
        logger().error("Failed to create cache directory: {}", e.what());
      }
    }
  }

  bool fail = false;
  {
    std::ofstream out;
    out.open(cacheFile.str(), std::ofstream::out | std::ofstream::binary);

    if (!out) {
      logger().error(
          "Failed to download tile data: Cannot open '{}' for writing!", cacheFile.str());
    }

    curlpp::Easy request;
    request.setOpt(curlpp::options::Url(url.str()));
    request.setOpt(curlpp::options::WriteStream(&out));
    request.setOpt(curlpp::options::NoSignal(true));

    request.perform();

    fail = curlpp::Info<CURLINFO_CONTENT_TYPE, std::string>::get(request).substr(0, 11) ==
           "application";
  }

  if (fail) {
    std::ifstream     in(cacheFile.str());
    std::stringstream sstr;
    sstr << in.rdbuf();

    std::remove(cacheFile.str().c_str());
    throw std::runtime_error(sstr.str());
  }

  boost::filesystem::perms filePerms =
      boost::filesystem::perms::owner_read | boost::filesystem::perms::owner_write |
      boost::filesystem::perms::group_read | boost::filesystem::perms::group_write |
      boost::filesystem::perms::others_read | boost::filesystem::perms::others_write;
  boost::filesystem::permissions(cacheFilePath, filePerms);

  return cacheFile.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/* virtual */ void TileSourceWebMapService::loadTileAsync(
    int level, glm::int64 patchIdx, OnLoadCallback cb) {
  mThreadPool.enqueue([=]() {
    auto* n = loadTile(level, patchIdx);
    cb(this, level, patchIdx, n);
  });
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int TileSourceWebMapService::getPendingRequests() {
  return static_cast<int>(mThreadPool.getPendingTaskCount() + mThreadPool.getRunningTaskCount());
}
////////////////////////////////////////////////////////////////////////////////////////////////////

void TileSourceWebMapService::setMaxLevel(uint32_t maxLevel) {
  mMaxLevel = maxLevel;
}

uint32_t TileSourceWebMapService::getMaxLevel() const {
  return mMaxLevel;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileSourceWebMapService::setCacheDirectory(std::string const& cacheDirectory) {
  mCache = cacheDirectory;
}

std::string const& TileSourceWebMapService::getCacheDirectory() const {
  return mCache;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileSourceWebMapService::setLayers(std::string const& layers) {
  mLayers = layers;
}

std::string const& TileSourceWebMapService::getLayers() const {
  return mLayers;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileSourceWebMapService::setUrl(std::string const& url) {
  mUrl = url;
}

std::string const& TileSourceWebMapService::getUrl() const {
  return mUrl;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void TileSourceWebMapService::setDataType(TileDataType type) {
  mFormat = type;
}

TileDataType TileSourceWebMapService::getDataType() const {
  return mFormat;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool TileSourceWebMapService::isSame(TileSource const* other) const {
  auto const* casted = dynamic_cast<TileSourceWebMapService const*>(other);

  return casted != nullptr && mUrl == casted->mUrl && mCache == casted->mCache &&
         mLayers == casted->mLayers && mFormat == casted->mFormat && mMaxLevel == casted->mMaxLevel;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::lodbodies
