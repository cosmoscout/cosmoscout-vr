#include "GDALReader.hpp"

// GDAL c++ includes
#include "cpl_conv.h" // for CPLMalloc()
#include "cpl_string.h"
#include "cpl_vsi.h"
#include "gdalwarper.h"
#include "ogr_spatialref.h"

#include <cstring>
#include <iostream>
#include <limits>

#include "../utils.hpp"

std::map<std::string, GDALReader::GreyScaleTexture> GDALReader::mTextureCache;
std::map<std::string, int>                          GDALReader::mBandsCache;
std::mutex                                          GDALReader::mMutex;
bool                                                GDALReader::mIsInitialized = false;

////////////////////////////////////////////////////////////////////////////////////////////////////

void GDALReader::InitGDAL() {
  GDALAllRegister();
  GDALReader::mIsInitialized = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GDALReader::AddTextureToCache(const std::string& path, GreyScaleTexture& texture) {
  GDALReader::mMutex.lock();
  // Cache the texture
  mTextureCache.insert(std::make_pair(path, texture));
  GDALReader::mMutex.unlock();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

int GDALReader::ReadNumberOfLayers(std::string filename) {
  if (!GDALReader::mIsInitialized) {
    csp::wcsoverlays::logger().error(
        "[GDALReader] GDAL not initialized! Call GDALReader::InitGDAL() first");
    return -1;
  }

  // Check for texture in cache
  GDALReader::mMutex.lock();
  auto it = mBandsCache.find(filename);
  if (it != mBandsCache.end()) {
    auto bands = it->second;

    GDALReader::mMutex.unlock();
    csp::wcsoverlays::logger().debug("Found {} in gdal bands cache.", filename);

    return bands;
  }
  GDALReader::mMutex.unlock();

  auto* poDatasetSrc = static_cast<GDALDataset*>(GDALOpen(filename.data(), GA_ReadOnly));

  if (poDatasetSrc == nullptr) {
    csp::wcsoverlays::logger().error(
        "[GDALReader::ReadNumberOfLayers] Failed to load {}", filename);
    return -1;
  }
  int bands = poDatasetSrc->GetRasterCount();
  GDALClose(poDatasetSrc);

  csp::wcsoverlays::logger().info("Reading number of layers from {} : {}", filename, bands);
  return bands;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GDALReader::ReadGrayScaleTexture(GreyScaleTexture& texture, std::string filename, int layer) {
  if (!GDALReader::mIsInitialized) {
    csp::wcsoverlays::logger().error(
        "[GDALReader] GDAL not initialized! Call GDALReader::InitGDAL() first");
    return;
  }

  csp::wcsoverlays::logger().info("Reading filename {} and layer {}", filename, layer);
  std::stringstream str;
  str << filename << layer;

  // Check for texture in cache
  GDALReader::mMutex.lock();
  auto it = mTextureCache.find(str.str());
  if (it != mTextureCache.end()) {
    texture = it->second;

    GDALReader::mMutex.unlock();
    csp::wcsoverlays::logger().debug("Found {} in gdal cache.", str.str());

    return;
  }
  GDALReader::mMutex.unlock();

  // Open the file. Needs to be supported by GDAL
  // TODO: There seems a multithreading issue in netCDF so we need to lock data reading
  GDALReader::mMutex.lock();
  auto* dataset = static_cast<GDALDataset*>(GDALOpen(filename.data(), GA_ReadOnly));
  GDALReader::mMutex.unlock();

  GDALReader::BuildTexture(dataset, texture, filename, layer);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GDALReader::ReadGrayScaleTexture(GDALReader::GreyScaleTexture& texture,
    std::stringstream const& data, const std::string& filename, int layer) {
  if (!GDALReader::mIsInitialized) {
    csp::wcsoverlays::logger().error(
        "[GDALReader] GDAL not initialized! Call GDALReader::InitGDAL() first");
    return;
  }

  csp::wcsoverlays::logger().info("Reading streamdata and layer {}", layer);
  std::stringstream str;
  str << filename << layer;

  // Check for texture in cache
  GDALReader::mMutex.lock();
  auto it = mTextureCache.find(str.str());
  if (it != mTextureCache.end()) {
    texture = it->second;

    GDALReader::mMutex.unlock();
    csp::wcsoverlays::logger().debug("Found {} in gdal cache.", str.str());

    return;
  }
  GDALReader::mMutex.unlock();

  // TODO: This is not optimal (?)
  std::streambuf*                            buf    = data.rdbuf();
  const typename std::stringstream::pos_type offset = buf->pubseekoff(0, std::stringstream::end);
  buf->pubseekpos(0);

  std::ostringstream dataInStream;
  dataInStream << buf;
  std::string wcsData = dataInStream.str();

  /// See https://gdal.org/user/virtual_file_systems.html#vsimem-in-memory-files for more info
  /// on in memory files
  std::stringstream memPath;
  memPath << "/vsimem/";
  memPath << csp::wcsoverlays::utils::split(filename, '/').back();

  VSILFILE* fpMem = VSIFileFromMemBuffer(
      memPath.str().c_str(), (GByte*)wcsData.c_str(), static_cast<vsi_l_offset>(offset), FALSE);
  VSIFCloseL(fpMem);

  GDALDataset* poDatasetSrc =
      static_cast<GDALDataset*>(GDALOpen(memPath.str().c_str(), GA_ReadOnly));

  GDALReader::BuildTexture(poDatasetSrc, texture, filename, layer);

  VSIUnlink(memPath.str().c_str());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GDALReader::ClearCache() {
  std::map<std::string, GreyScaleTexture>::iterator it;

  GDALReader::mMutex.lock();
  // Loop over textures and delete buffer
  for (it = mTextureCache.begin(); it != mTextureCache.end(); it++) {
    GreyScaleTexture texture = it->second;
    free(texture.buffer);
  }
  mTextureCache.clear();
  GDALReader::mMutex.unlock();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GDALReader::BuildTexture(GDALDataset* poDatasetSrc, GDALReader::GreyScaleTexture& texture,
    std::string const& filename, int layer) {
  // Meta data storage
  double                adfSrcGeoTransform[6];
  double                adfDstGeoTransform[6];
  std::array<double, 4> bounds{};
  std::array<double, 2> d_dataRange{};

  int resX = 0;
  int resY = 0;

  if (poDatasetSrc == nullptr) {
    csp::wcsoverlays::logger().error(
        "[GDALReader::ReadGrayScaleTexture] Failed to load {}", filename);
    return;
  }

  if (poDatasetSrc->GetProjectionRef() == nullptr) {
    csp::wcsoverlays::logger().error(
        "[GDALReader::ReadGrayScaleTexture] No projection defined for {}", filename);
    return;
  }

  // Read geotransform from src image
  poDatasetSrc->GetGeoTransform(adfSrcGeoTransform);

  int   bGotMin = 0;
  int   bGotMax = 0; // like bool if it was successful
  auto* poBand  = poDatasetSrc->GetRasterBand(layer);

  d_dataRange[0] = poBand->GetMinimum(&bGotMin);
  d_dataRange[1] = poBand->GetMaximum(&bGotMax);
  if (!(bGotMin && bGotMax)) {
    GDALComputeRasterMinMax(static_cast<GDALRasterBandH>(poBand), TRUE, d_dataRange.data());
  }

  /////////////////////// Reprojection /////////////////////
  char* pszDstWKT = nullptr;

  // Setup output coordinate system to WGS84 (latitude/longitude).
  OGRSpatialReference oSRS;
  oSRS.SetWellKnownGeogCS("WGS84");
  oSRS.exportToWkt(&pszDstWKT);

  // Create the transformation object handle
  auto* hTransformArg = GDALCreateGenImgProjTransformer(
      poDatasetSrc, poDatasetSrc->GetProjectionRef(), nullptr, pszDstWKT, FALSE, 0.0, 1);

  // Create output coordinate system and store transformation
  GDALSuggestedWarpOutput(
      poDatasetSrc, GDALGenImgProjTransform, hTransformArg, adfDstGeoTransform, &resX, &resY);

  // Calculate extents of the image
  bounds[0] =
      (adfDstGeoTransform[0] + 0 * adfDstGeoTransform[1] + 0 * adfDstGeoTransform[2]) * M_PI / 180;
  bounds[1] =
      (adfDstGeoTransform[3] + 0 * adfDstGeoTransform[4] + 0 * adfDstGeoTransform[5]) * M_PI / 180;
  bounds[2] =
      (adfDstGeoTransform[0] + resX * adfDstGeoTransform[1] + resY * adfDstGeoTransform[2]) * M_PI /
      180;
  bounds[3] =
      (adfDstGeoTransform[3] + resX * adfDstGeoTransform[4] + resY * adfDstGeoTransform[5]) * M_PI /
      180;

  // Store the data type of the raster band
  auto eDT = GDALGetRasterDataType(GDALGetRasterBand(poDatasetSrc, layer));

  // Setup the warping parameters
  GDALWarpOptions* psWarpOptions = GDALCreateWarpOptions();
  psWarpOptions->hSrcDS          = poDatasetSrc;
  psWarpOptions->hDstDS          = nullptr;
  psWarpOptions->nBandCount      = 1;
  psWarpOptions->panSrcBands =
      static_cast<int*>(CPLMalloc(sizeof(int) * psWarpOptions->nBandCount));
  psWarpOptions->panSrcBands[0] = layer;
  psWarpOptions->panDstBands =
      static_cast<int*>(CPLMalloc(sizeof(int) * psWarpOptions->nBandCount));
  psWarpOptions->panDstBands[0] = 1;
  psWarpOptions->pfnProgress    = GDALTermProgress;

  psWarpOptions->pTransformerArg = GDALCreateGenImgProjTransformer3(
      GDALGetProjectionRef(poDatasetSrc), adfSrcGeoTransform, pszDstWKT, adfDstGeoTransform);
  psWarpOptions->pfnTransformer = GDALGenImgProjTransform;

  // Allocate memory for the image pixels

  int bufferSize = resX * resY * GDALGetDataTypeSizeBytes(eDT);

  void* bufferData = CPLMalloc(bufferSize);

  // execute warping from src to dst
  GDALWarpOperation oOperation;
  oOperation.Initialize(psWarpOptions);
  oOperation.WarpRegionToBuffer(0, 0, resX, resY, bufferData, eDT);
  GDALDestroyGenImgProjTransformer(psWarpOptions->pTransformerArg);
  GDALDestroyWarpOptions(psWarpOptions);

  mBandsCache.insert(std::make_pair(filename, poDatasetSrc->GetRasterCount()));

  GDALClose(poDatasetSrc);

  /////////////////////// Reprojection End /////////////////
  texture.buffersize   = bufferSize;
  texture.buffer       = static_cast<void*>(CPLMalloc(bufferSize));
  texture.x            = resX;
  texture.y            = resY;
  texture.dataRange    = d_dataRange;
  texture.lnglatBounds = bounds;
  texture.type         = eDT;
  texture.layers       = poDatasetSrc->GetRasterCount();
  std::memcpy(texture.buffer, bufferData, bufferSize);

  switch (eDT) {
  case 1: // UInt8
    texture.typeSize = std::numeric_limits<uint8_t>::max();
    break;
  case 2: // UInt16
    texture.typeSize = std::numeric_limits<uint16_t>::max();
    break;
  case 3: // Int16
    texture.typeSize = std::numeric_limits<int16_t>::max();
    break;
  case 4: // UInt32
    texture.typeSize = std::numeric_limits<uint32_t>::max();
    break;
  case 5: // Int32
    texture.typeSize = std::numeric_limits<int32_t>::max();
    break;

  default: // Float
    texture.typeSize = 1;
  }

  GDALReader::AddTextureToCache(filename, texture);
}
