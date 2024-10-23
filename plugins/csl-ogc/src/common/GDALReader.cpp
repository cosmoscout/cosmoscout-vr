////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "GDALReader.hpp"

// GDAL c++ includes
#include <cpl_conv.h> // for CPLMalloc()
#include <cpl_string.h>
#include <cpl_vsi.h>
#include <gdalwarper.h>
#include <ogr_spatialref.h>
#include <ogrsf_frmts.h>

#include <cstring>
#include <iostream>
#include <limits>

#include "utils.hpp"

namespace csl::ogc {

std::map<std::string, GDALReader::Texture> GDALReader::mTextureCache;
std::mutex                                 GDALReader::mMutex;
bool                                       GDALReader::mIsInitialized = false;

////////////////////////////////////////////////////////////////////////////////////////////////////

void GDALReader::InitGDAL() {
  GDALAllRegister();
  GDALReader::mIsInitialized = true;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GDALReader::AddTextureToCache(std::string const& path, Texture const& texture) {
  GDALReader::mMutex.lock();
  // Cache the texture
  mTextureCache.insert(std::make_pair(path, texture));
  GDALReader::mMutex.unlock();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GDALReader::ReadTexture(Texture& texture, std::string filename) {
  if (!GDALReader::mIsInitialized) {
    csl::ogc::logger().error(
        "[GDALReader] GDAL not initialized! Call GDALReader::InitGDAL() first");
    return;
  }

  // Check for texture in cache
  GDALReader::mMutex.lock();
  auto it = mTextureCache.find(filename);
  if (it != mTextureCache.end()) {
    texture = it->second;

    GDALReader::mMutex.unlock();
    csl::ogc::logger().debug("Found {} in gdal cache.", filename);

    return;
  }
  GDALReader::mMutex.unlock();

  // Open the file. Needs to be supported by GDAL
  // TODO: There seems a multithreading issue in netCDF so we need to lock data reading
  GDALReader::mMutex.lock();
  auto* dataset = static_cast<GDALDataset*>(GDALOpen(filename.data(), GA_ReadOnly));
  GDALReader::mMutex.unlock();

  GDALReader::BuildTexture(dataset, texture, filename);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GDALReader::ReadTexture(
    GDALReader::Texture& texture, std::stringstream const& data, const std::string& filename) {
  if (!GDALReader::mIsInitialized) {
    csl::ogc::logger().error(
        "[GDALReader] GDAL not initialized! Call GDALReader::InitGDAL() first");
    return;
  }

  // Check for texture in cache
  GDALReader::mMutex.lock();
  auto it = mTextureCache.find(filename);
  if (it != mTextureCache.end()) {
    texture = it->second;

    GDALReader::mMutex.unlock();
    csl::ogc::logger().debug("Found {} in gdal cache.", filename);

    return;
  }
  GDALReader::mMutex.unlock();

  std::string dataStr  = data.str();
  std::size_t dataSize = dataStr.size();

  /// See https://gdal.org/user/virtual_file_systems.html#vsimem-in-memory-files for more info
  /// on in memory files
  VSILFILE* fpMem = VSIFileFromMemBuffer(
      "/vsimem/tmp.tiff", reinterpret_cast<GByte*>(&dataStr[0]), dataSize, FALSE);
  VSIFCloseL(fpMem);

  GDALDataset* dataset = static_cast<GDALDataset*>(GDALOpen("/vsimem/tmp.tiff", GA_ReadOnly));

  GDALReader::BuildTexture(dataset, texture, filename);

  VSIUnlink("/vsimem/tmp.tiff");
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GDALReader::ClearCache() {
  std::map<std::string, Texture>::iterator it;

  GDALReader::mMutex.lock();
  // Loop over textures and delete buffer
  for (it = mTextureCache.begin(); it != mTextureCache.end(); it++) {
    Texture texture = it->second;
    free(texture.mBuffer);
  }
  mTextureCache.clear();
  GDALReader::mMutex.unlock();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void GDALReader::BuildTexture(
    GDALDataset* dataset, GDALReader::Texture& texture, std::string const& filename) {

  if (dataset == nullptr) {
    csl::ogc::logger().error("[GDALReader::ReadTexture] Failed to load {}", filename);
    return;
  }

  if (dataset->GetProjectionRef() == nullptr) {
    csl::ogc::logger().error("[GDALReader::ReadTexture] No projection defined for {}", filename);
    return;
  }

  // Get band ranges -------------------------------------------------------------------------------

  texture.mBandDataRanges.clear();

  // Get the global min and max values of all bands.
  texture.mDataRange[0] = std::numeric_limits<double>::max();
  texture.mDataRange[1] = std::numeric_limits<double>::lowest();

  uint32_t bandCount = dataset->GetRasterCount();

  std::array<double, 2> bandRange{
      std::numeric_limits<double>::max(), std::numeric_limits<double>::lowest()};

  for (uint32_t b = 1; b <= bandCount; b++) {
    int   bGotMin = 0;
    int   bGotMax = 0; // like bool if it was successful
    auto* poBand  = dataset->GetRasterBand(b);

    bandRange[0] = poBand->GetMinimum(&bGotMin);
    bandRange[1] = poBand->GetMaximum(&bGotMax);
    if (!bGotMin || !bGotMax) {
      GDALComputeRasterMinMax(static_cast<GDALRasterBandH>(poBand), TRUE, bandRange.data());
    }

    texture.mBandDataRanges.push_back(bandRange);

    texture.mDataRange[0] = std::min(texture.mDataRange[0], bandRange[0]);
    texture.mDataRange[1] = std::max(texture.mDataRange[1], bandRange[1]);
  }

  csl::ogc::logger().info("[GDALReader::ReadTexture] Band count {} ", bandCount);

  // Reprojection ----------------------------------------------------------------------------------

  // Read geotransform from src image'.
  double adfSrcGeoTransform[6];
  dataset->GetGeoTransform(adfSrcGeoTransform);

  char* pszDstWKT = nullptr;

  // Setup output coordinate system to WGS84 (latitude/longitude).
  OGRSpatialReference oSRS;
  oSRS.SetWellKnownGeogCS("WGS84");
  oSRS.exportToWkt(&pszDstWKT);

  // Create the transformation object handle
  auto* hTransformArg = GDALCreateGenImgProjTransformer(
      dataset, dataset->GetProjectionRef(), nullptr, pszDstWKT, FALSE, 0.0, 1);

  // Create output coordinate system and store transformation
  double transform[6];
  int    resX = 0;
  int    resY = 0;
  GDALSuggestedWarpOutput(dataset, GDALGenImgProjTransform, hTransformArg, transform, &resX, &resY);

  // Calculate extents of the image
  std::array<double, 4> bounds{};
  bounds[0] = (transform[0] + 0 * transform[1] + 0 * transform[2]) * M_PI / 180;
  bounds[1] = (transform[3] + 0 * transform[4] + 0 * transform[5]) * M_PI / 180;
  bounds[2] = (transform[0] + resX * transform[1] + resY * transform[2]) * M_PI / 180;
  bounds[3] = (transform[3] + resX * transform[4] + resY * transform[5]) * M_PI / 180;

  // Store the data type of the raster band
  auto eDT = GDALGetRasterDataType(GDALGetRasterBand(dataset, 1));

  // Setup the warping parameters
  GDALWarpOptions* psWarpOptions = GDALCreateWarpOptions();
  psWarpOptions->hSrcDS          = dataset;
  psWarpOptions->hDstDS          = nullptr;

  psWarpOptions->nBandCount  = bandCount;
  psWarpOptions->panSrcBands = static_cast<int*>(CPLMalloc(sizeof(int) * bandCount));
  psWarpOptions->panDstBands = static_cast<int*>(CPLMalloc(sizeof(int) * bandCount));

  for (uint32_t i = 0; i < bandCount; i++) {
    psWarpOptions->panSrcBands[i] = i + 1;
    psWarpOptions->panDstBands[i] = i + 1;
  }

  psWarpOptions->pTransformerArg = GDALCreateGenImgProjTransformer3(
      GDALGetProjectionRef(dataset), adfSrcGeoTransform, pszDstWKT, transform);
  psWarpOptions->pfnTransformer = GDALGenImgProjTransform;

  // Allocate memory for the image pixels
  int elementCount = resX * resY * bandCount;
  int bufferSize   = elementCount * GDALGetDataTypeSizeBytes(eDT);

  void* bufferData = CPLMalloc(bufferSize);

  // execute warping from src to dst
  GDALWarpOperation oOperation;
  oOperation.Initialize(psWarpOptions);
  oOperation.WarpRegionToBuffer(0, 0, resX, resY, bufferData, eDT);
  GDALDestroyGenImgProjTransformer(psWarpOptions->pTransformerArg);
  GDALDestroyWarpOptions(psWarpOptions);

  GDALClose(dataset);

  /////////////////////// Reprojection End /////////////////
  texture.mBufferSize   = bufferSize;
  texture.mBuffer       = static_cast<void*>(CPLMalloc(bufferSize));
  texture.mWidth        = resX;
  texture.mHeight       = resY;
  texture.mLnglatBounds = bounds;
  texture.mDataType     = eDT;
  texture.mBands        = bandCount;
  std::memcpy(texture.mBuffer, bufferData, bufferSize);

  if (eDT == 7) {
    // Double support. we need to convert to float do to opengl
    std::vector<double> dData(static_cast<double*>(texture.mBuffer),
        static_cast<double*>(texture.mBuffer) + elementCount);

    std::vector<float> fData{};
    fData.resize(dData.size());

    std::transform(
        dData.begin(), dData.end(), fData.begin(), [](double d) { return static_cast<float>(d); });

    CPLFree(texture.mBuffer);

    texture.mBuffer = static_cast<void*>(CPLMalloc(elementCount * sizeof(float)));
    std::memcpy(texture.mBuffer, &fData[0], elementCount * sizeof(float));
  }

  GDALReader::AddTextureToCache(filename, texture);
}

} // namespace csl::ogc