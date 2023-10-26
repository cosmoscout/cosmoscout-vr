////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "utils.hpp"

#include "HEALPix.hpp"

#include "BaseTileData.hpp"
#include "VistaPlanet.hpp"

#include "../../../src/cs-utils/convert.hpp"

#include <VistaKernel/GraphicsManager/VistaOpenGLNode.h>

#include <glm/gtx/quaternion.hpp>

namespace csp::lodbodies::utils {

////////////////////////////////////////////////////////////////////////////////////////////////////

double getHeight(
    VistaPlanet const* planet, HeightSamplePrecision precision, glm::dvec2 const& lngLat) {

  // Check if TreeManagerDEM is ok
  if (planet->getTileRenderer().getTreeManager() == nullptr) {
    return 0.0F;
  }

  // Check if TreeManagerDEM -> GetTree is ok
  if (planet->getTileRenderer().getTreeManager()->getTree() == nullptr) {
    return 0.0F;
  }

  int rootIndex = HEALPix::convertLngLat2Base(lngLat); //!< Patch Index of Root

  glm::dvec2 relative1; //!< Relative Coordinates in a Patch
  glm::dvec2 relative2; //!< Relative Coordinates in a Patch before change

  // Convert to HEALPix
  relative1 = HEALPix::convertBaseLngLat2XY(rootIndex, lngLat);

  // Get the right root
  TileNode* parent = planet->getTileRenderer().getTreeManager()->getTree()->getRoot(rootIndex);

  // Check if parent is valid
  if (parent == nullptr) {
    return 0.0F;
  }

  // Start with the Root
  TileNode* child = parent;

  // Clamp Values
  glm::clamp(relative1, glm::dvec2(0), glm::dvec2(1));

  // Index of Child
  int childIndex = -1;

  // Go down tree only if not coarse precision
  while (child && int(precision) > 1) {
    parent = child;

    relative2 = relative1;

    if (relative1.x < 0.5 && relative1.y < 0.5) {
      childIndex  = 0;
      relative1.x = relative1.x * 2.0;
      relative1.y = relative1.y * 2.0;
    } else if (relative1.x >= 0.5 && relative1.y < 0.5) {
      childIndex  = 1;
      relative1.x = (relative1.x - 0.5) * 2.0;
      relative1.y = relative1.y * 2.0;
    } else if (relative1.x < 0.5 && relative1.y >= 0.5) {
      childIndex  = 2;
      relative1.x = relative1.x * 2.0;
      relative1.y = (relative1.y - 0.5) * 2.0;
    } else {
      childIndex  = 3;
      relative1.x = (relative1.x - 0.5) * 2.0;
      relative1.y = (relative1.y - 0.5) * 2.0;
    }

    // Get the new Child
    child = parent->getChild(childIndex);

    // Child is unavailable and precision is "HeightSamplePrecision::eActual"
    if (child == nullptr && precision == HeightSamplePrecision::eActual) {
      child = parent;

      // Reset coordinates
      relative1 = relative2;
      break;
    }

    // Child is unavailable and precision is "Fine"
    if (child == nullptr && precision == HeightSamplePrecision::eFine) {
      std::vector<TileId> requested;

      requested.push_back(HEALPix::getChildTileId(parent->getTileId(), childIndex));

      planet->getTileRenderer().getTreeManager()->request(requested);

      // planet->getTileRenderer().getTreeManager()->merge();
      planet->getTileRenderer().getTreeManager()->update();

      child = parent->getChild(childIndex);
    }
  }

  // Check if Child Exists
  if (child == nullptr) {
    return 0.0;
  }

  uint32_t size = child->getTileData().get(TileDataType::eElevation)->getResolution();

  // Figure out flip
  std::swap(relative1.x, relative1.y);

  double u = relative1.x * (size - 1);
  double v = relative1.y * (size - 1);

  int uB = static_cast<int>(u);
  int vB = static_cast<int>(v);

  double uP = u - uB;
  double vP = v - vB;

  double h{};
  double hP1{};
  double hP2{};
  double hPP{};

  const auto* ptr = child->getTileData().get(TileDataType::eElevation)->getTypedPtr<float>();
  h               = ptr[vB + size * uB]; // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  hP1 = ptr[vB + size * (uB + 1)];       // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  hP2 = ptr[vB + 1 + size * uB];         // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)
  hPP = ptr[vB + 1 + size * (uB + 1)];   // NOLINT(cppcoreguidelines-pro-bounds-pointer-arithmetic)

  double interpol1 = (1.0 - uP) * h + uP * hP1;
  double interpol2 = (1.0 - uP) * hP2 + uP * hPP;
  double height    = (1.0 - vP) * interpol1 + vP * interpol2;

  return height;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool intersectTileBounds(TileNode const* tileNode, VistaPlanet const* planet,
    glm::dvec4 const& origin, glm::dvec4 const& direction, double& minDist, double& maxDist) {
  BoundingBox<double> tile_bounds = tileNode->getBounds();
  std::array dMin{tile_bounds.getMin()[0], tile_bounds.getMin()[1], tile_bounds.getMin()[2]};
  std::array dMax{tile_bounds.getMax()[0], tile_bounds.getMax()[1], tile_bounds.getMax()[2]};

  // TODO: double precision?
  glm::dvec3 aabb_min(dMin.at(0), dMin.at(1), dMin.at(2));
  glm::dvec3 aabb_max(dMax.at(0), dMax.at(1), dMax.at(2));
  glm::dvec3 o(origin);
  glm::dvec3 d(direction);

  return BoundingBox<double>(aabb_min, aabb_max)
      .GetIntersectionDistance(o, d, true, minDist, maxDist);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

bool intersectPlanet(
    VistaPlanet const* planet, glm::dvec3 rayOrigin, glm::dvec3 rayDir, glm::dvec3& pos) {
  // Check if TreeManagerDEM is ok
  if (planet->getTileRenderer().getTreeManager() == nullptr) {
    return false;
  }

  // Check if TreeManagerDEM -> GetTree is ok
  if (planet->getTileRenderer().getTreeManager()->getTree() == nullptr) {
    return false;
  }

  // Initialize Result to Zero
  pos = glm::dvec3(0);

  // Planet transform -> Inverse -> so we are in planet space
  glm::dmat4 planet_transform = planet->getWorldTransform();
  glm::dmat4 planet_transformnv;
  planet_transformnv = glm::inverse(planet_transform);

  // Transform ray into planet coordinate system
  glm::dvec4 origin(rayOrigin[0], rayOrigin[1], rayOrigin[2], 1.0);
  origin = planet_transformnv * origin;

  glm::dvec4 direction(rayDir[0], rayDir[1], rayDir[2], 0.0);
  direction = planet_transformnv * direction;
  direction = glm::normalize(direction);

  // Determine intersected root patches
  std::multimap<double, TileNode*> intersected_tiles;
  for (int rootIndex = 0; rootIndex < 12; ++rootIndex) {

    TileNode* root_node = planet->getTileRenderer().getTreeManager()->getTree()->getRoot(rootIndex);

    if (root_node == nullptr) {
      return false;
    }

    double min_dist{};
    double max_dist{};
    bool intersects = intersectTileBounds(root_node, planet, origin, direction, min_dist, max_dist);
    if (intersects) {
      intersected_tiles.insert(std::pair<double, TileNode*>(min_dist, root_node));
    }
  }

  TileNode* parent = nullptr;
  TileNode* child  = nullptr;

  // Process intersected tile patch priority queue:
  while (!intersected_tiles.empty()) {
    parent = intersected_tiles.begin()->second;
    intersected_tiles.erase(intersected_tiles.begin());

    if (parent == nullptr) {
      return false;
    }

    // Sample height field of cut leaf node
    if (!parent->childrenAvailable()) {
      // Get entry and exit point again:
      double min_dist{};
      double max_dist{};
      intersectTileBounds(parent, planet, origin, direction, min_dist, max_dist);

      // Prevent entry being behind the camera:
      min_dist             = std::max(0.0, min_dist);
      glm::dvec4 entry     = origin + direction * min_dist;
      glm::dvec4 exit      = origin + direction * max_dist;
      auto       sampleDir = (exit - entry).xyz();
      glm::dvec3 lastSampleCartesian{};
      glm::dvec3 sampleCartesian{};
      glm::dvec3 lastSampleLngLatHeight{};
      glm::dvec3 sampleLngLatHeight{};
      double     height(0.0);
      double     lastHeight(0.0);
      bool       first_sample(true);

      // Estimate number of sampling steps
      //        -------------------- BboxMax
      //        |        /\        |
      //        |   255 /  \       |
      //        |     /      \     |
      //        |   /          \   |
      //        | /     Tile     \ |
      //        | \               /|
      //        |   \            / |
      //        |     \        /   |
      //        |  255 \      /    |
      //        |        \   /     |
      //        |         \/       |
      // BboxMin--------------------
      auto const& tile        = parent->getTileData().get(TileDataType::eElevation);
      auto        tile_bounds = parent->getBounds();

      // Tile sizes
      int size = tile->getResolution();

      auto max_tile_samplings = std::sqrt((size * size) + (size * size));
      auto max_bbox_samplings = std::sqrt(
          (max_tile_samplings * max_tile_samplings) + (max_tile_samplings * max_tile_samplings));

      auto step_factor =
          glm::length(sampleDir) / glm::length(tile_bounds.getMax() - tile_bounds.getMin());
      auto step_nr = step_factor * max_bbox_samplings;

      // Sample between entry and exit:
      for (int step(0); step <= static_cast<int>(step_nr); ++step) {
        lastSampleCartesian    = sampleCartesian;
        lastSampleLngLatHeight = sampleLngLatHeight;
        lastHeight             = height;

        // Sample along ray and then convert to polar:
        sampleCartesian = glm::dvec3(entry[0], entry[1], entry[2]) +
                          glm::dvec3((step / step_nr) * sampleDir[0],
                              (step / step_nr) * sampleDir[1], (step / step_nr) * sampleDir[2]);

        sampleLngLatHeight =
            cs::utils::convert::cartesianToLngLatHeight(sampleCartesian, planet->getRadii());

        // Calc correct HPix coordinate for child patch in relation to base batch
        int        base   = HEALPix::getBasePatch(parent->getTileId());
        auto       scale  = HEALPix::getPatchOffsetScale(parent->getTileId());
        glm::dvec2 HPixPt = HEALPix::convertBaseLngLat2XY(base, sampleLngLatHeight.xy());
        HPixPt            = (HPixPt - glm::dvec2(scale[0], scale[1])) / scale[2];

        /* Figure out flip */
        std::swap(HPixPt.x, HPixPt.y);

        // Calc coords in texture space
        double u = HPixPt.x * (size - 1);
        double v = HPixPt.y * (size - 1);

        int uB = static_cast<int>(u);
        int vB = static_cast<int>(v);

        double uP = u - uB;
        double vP = v - vB;

        double hP1{};
        double hP2{};
        double hPP{};
        if (uB >= size - 1 || uB < 0) {
          continue;
        }
        if (vB >= size - 1 || vB < 0) {
          continue;
        }

        // Access height data
        const auto* ptr = parent->getTileData().get(TileDataType::eElevation)->getTypedPtr<float>();
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        height = ptr[vB + size * uB];
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        hP1 = ptr[vB + size * (uB + 1)];
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        hP2 = ptr[vB + 1 + size * uB];
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        hPP = ptr[vB + 1 + size * (uB + 1)];

        double interpol1 = (1.0 - uP) * height + uP * hP1;
        double interpol2 = (1.0 - uP) * hP2 + uP * hPP;
        height           = (1.0 - vP) * interpol1 + vP * interpol2;
        height *= planet->getHeightScale();

        // Detect hit as soon as a sample point below the surface is found
        if (sampleLngLatHeight.z < height) {
          // Ignore sampling, which already starts below a patch due to exaggeration
          if (first_sample) {
            break;
          }

          double lastWeight = lastSampleLngLatHeight.z - lastHeight;
          double curWeight  = height - sampleLngLatHeight.z;

          double sum = lastWeight + curWeight;

          lastWeight /= sum;
          curWeight /= sum;

          pos = lastSampleCartesian * (1.0 - lastWeight) + sampleCartesian * (1.0 - curWeight);

          // FOUND VALID INTERSECTION!!!
          return true;
        }
        first_sample = false;
      }
    }
    // Parent is not a leaf in cut -> push intersected children into queue:
    else {
      // Check all children for intersection
      for (int childIndex = 0; childIndex < 4; ++childIndex) {
        /* Get the new Child */
        child = parent->getChild(childIndex);
        if (child == nullptr) {
          continue;
        }

        double min_dist{};
        double max_dist{};
        bool intersects = intersectTileBounds(child, planet, origin, direction, min_dist, max_dist);
        if (intersects && max_dist > 0.0) {
          intersected_tiles.insert(std::pair<double, TileNode*>(min_dist, child));
        }
      }
    }
  }
  return false;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::lodbodies::utils
