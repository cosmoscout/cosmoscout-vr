////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "VoronoiGenerator.hpp"
#include "../logger.hpp"
#include "Arc.hpp"

#include <glm/glm.hpp>
#include <iomanip>
#include <limits>

namespace csp::measurementtools {

VoronoiGenerator::VoronoiGenerator()
    : mBeachline(this) {
}

void VoronoiGenerator::parse(std::vector<Site> const& sites) {
  mSites     = sites;
  mBeachline = Beachline(this);
  mSweepline = 0.0;
  mMaxY      = 0.0;
  mMinY      = 0.0;
  mVoronoiEdges.clear();
  mTriangulationEdges.clear();
  mTriangles.clear();
  mNeighbors.clear();

  if (sites.size() > 1) {
    for (auto site : sites) {
      if (site.mY > mMaxY) {
        mMaxY = site.mY;
      }
      if (site.mY < mMinY) {
        mMinY = site.mY;
      }
      mSiteEvents.push(site);
    }

    while (!mCircleEvents.empty() || !mSiteEvents.empty()) {
      double nextSite =
          mSiteEvents.empty() ? std::numeric_limits<double>::max() : mSiteEvents.top().mY;
      double nextCircle = mCircleEvents.empty() ? std::numeric_limits<double>::max()
                                                : mCircleEvents.top()->mPriority.mY;

      if (nextCircle < nextSite) {
        Circle* next = mCircleEvents.top();
        mCircleEvents.pop();
        mSweepline = next->mPriority.mY;
        process(next);
        delete next; // NOLINT(cppcoreguidelines-owning-memory): TODO
      } else {
        Site next = mSiteEvents.top();
        mSiteEvents.pop();

        // hackhack...
        if (!mSiteEvents.empty() && mSiteEvents.top().mY == next.mY &&
            mSiteEvents.top().mX == next.mX) {
          continue;
        }

        mSweepline = next.mY;
        process(next);
      }
    }

    finishEdges();
  }
}

double VoronoiGenerator::sweepLine() const {
  return mSweepline;
}

double VoronoiGenerator::maxY() const {
  return mMaxY;
}

double VoronoiGenerator::minY() const {
  return mMinY;
}

std::vector<Site> const& VoronoiGenerator::getSites() const {
  return mSites;
}

std::vector<Edge> const& VoronoiGenerator::getEdges() const {
  return mVoronoiEdges;
}

std::vector<Edge2> const& VoronoiGenerator::getTriangulation() const {
  return mTriangulationEdges;
}

std::vector<Triangle> const& VoronoiGenerator::getTriangles() const {
  return mTriangles;
}

std::map<uint16_t, std::vector<Site>> const& VoronoiGenerator::getNeighbors() const {
  return mNeighbors;
}

void VoronoiGenerator::addTriangulationEdge(Site const& site1, Site const& site2) {
  mTriangulationEdges.emplace_back(site1, site2);

  // site, place of site in mTriangles
  std::vector<std::pair<Site, int>> tri;
  int                               element = 0;

  // checks if triangle with site1 and site2 corners exist in mTriangles
  // if yes, saves the location in mTriangles and it's third corner
  for (auto const& t : mTriangles) {
    Site si1(0, 0, 0);
    Site si2(0, 0, 0);
    Site si3(0, 0, 0);
    std::tie(si1, si2, si3) = t;

    if (((site1.mAddr == si1.mAddr) || (site1.mAddr == si2.mAddr) || (site1.mAddr == si3.mAddr)) &&
        ((site2.mAddr == si1.mAddr) || (site2.mAddr == si2.mAddr) || (site2.mAddr == si3.mAddr))) {
      if ((si1.mAddr != site1.mAddr) && (si1.mAddr != site2.mAddr)) {
        tri.emplace_back(si1, element);
      } else if ((si2.mAddr != site1.mAddr) && (si2.mAddr != site2.mAddr)) {
        tri.emplace_back(si2, element);
      } else {
        tri.emplace_back(si3, element);
      }
    }
    element++;
  }

  // if sites have a common neighbor, the three form a triangle
  for (auto const& s1 : mNeighbors[site1.mAddr]) {
    for (auto const& s2 : mNeighbors[site2.mAddr]) {
      // common neighbor found
      if (s1.mAddr == s2.mAddr) {
        // emplace back triangle to vectors
        if (tri.empty()) {
          mTriangles.emplace_back(std::make_tuple(site1, site2, s1));
          tri.emplace_back(s1, static_cast<uint16_t>(mTriangles.size() - 1));
        }
        // if there are already triangles between site1 and site2
        else {
          // variables to remove elements from tri
          std::vector<int> remove;
          int              elementTri = 0;

          // examines every triangle
          for (auto const& p : tri) {
            // edges
            glm::dvec2 edge1 = glm::dvec2(site2.mX, site2.mY) - glm::dvec2(site1.mX, site1.mY);
            glm::dvec2 edge2 = glm::dvec2(s1.mX, s1.mY) - glm::dvec2(site1.mX, site1.mY);
            glm::dvec2 edge3 = glm::dvec2(p.first.mX, p.first.mY) - glm::dvec2(site1.mX, site1.mY);

            // cross products of edges to find out the order (clockwise, counterclockwise)
            double cross1 = edge1.x * edge2.y - edge1.y * edge2.x;
            double cross2 = edge1.x * edge3.y - edge1.y * edge3.x;
            double cross3 = edge2.x * edge3.y - edge2.y * edge3.x;

            // s1 and p.first are on the same side of edge1 (site1, site2)
            // they are overlapping each other -> one of them are unnecessary
            if ((cross1 > 0) == (cross2 > 0)) {
              // p_first is the outer site -> needs to be eliminated
              if ((cross1 > 0) == (cross3 > 0)) {
                // deletes unnecessary triangle from mTriangles and add new
                try {
                  mTriangles.at(p.second) = std::make_tuple(site1, site2, s1);
                  tri.emplace_back(s1, p.second);
                  remove.emplace_back(elementTri);
                } catch (std::exception const& e) {
                  logger().error("Triangle elimination in VoronoiGenerator: {}", e.what());
                }
              }
              // else: does not add unnecessary triangle
            }
            elementTri++;
          }

          // removes outdated elements from tri
          for (size_t i = remove.size(); i > 0; i--) {
            tri.erase(tri.begin() + i - 1);
          }
        }
      }
    }
  }

  mNeighbors[site1.mAddr].push_back(site2);
  mNeighbors[site2.mAddr].push_back(site1);
}

void VoronoiGenerator::removeTriangulationEdge(Site const& site1, Site const& site2) {
  // removes edge
  int i = 0;
  for (auto const& s : mTriangulationEdges) {
    if (((s.first == site1) && (s.second == site2)) ||
        ((s.first == site2) && (s.second == site1))) {
      mTriangulationEdges.erase(mTriangulationEdges.begin() + i);
    }

    i++;
  }

  // removes triangles
  i = 0;
  for (auto const& t : mTriangles) {
    Site s1(0, 0, 0);
    Site s2(0, 0, 0);
    Site s3(0, 0, 0);
    std::tie(s1, s2, s3) = t;
    if (((s1 == site1 || s2 == site1) && s3 == site2) ||
        ((s1 == site1 || s3 == site1) && s2 == site2) ||
        ((s2 == site1 || s3 == site1) && s1 == site2)) {
      mTriangles.erase(mTriangles.begin() + i);
    }
    i++;
  }

  // removes neigbours
  i = 0;
  for (auto const& s : mNeighbors[site1.mAddr]) {
    if (s == site2) {
      mNeighbors[site1.mAddr].erase(mNeighbors[site1.mAddr].begin() + i);
    }
    i++;
  }
  i = 0;
  for (auto const& s : mNeighbors[site2.mAddr]) {
    if (s == site1) {
      mNeighbors[site2.mAddr].erase(mNeighbors[site2.mAddr].begin() + i);
    }
    i++;
  }
}

void VoronoiGenerator::process(Circle* event) {

  if (event->mIsValid) {

    Arc* leftArc  = event->mArc->mLeftBreak ? event->mArc->mLeftBreak->mLeftArc : nullptr;
    Arc* rightArc = event->mArc->mRightBreak ? event->mArc->mRightBreak->mRightArc : nullptr;

    if (leftArc) {
      mVoronoiEdges.push_back(event->mArc->mLeftBreak->finishEdge(event->mCenter));
    }
    if (rightArc) {
      mVoronoiEdges.push_back(event->mArc->mRightBreak->finishEdge(event->mCenter));
    }

    mBeachline.removeArc(event->mArc);

    addCircleEvent(leftArc);
    addCircleEvent(rightArc);
  }
}

void VoronoiGenerator::process(Site const& event) {

  Arc* newArc = mBeachline.insertArcFor(event);

  addCircleEvent(newArc->mLeftBreak ? newArc->mLeftBreak->mLeftArc : nullptr);
  addCircleEvent(newArc->mRightBreak ? newArc->mRightBreak->mRightArc : nullptr);
}

void VoronoiGenerator::addCircleEvent(Arc* arc) {
  if (arc) {
    // NOLINTNEXTLINE(cppcoreguidelines-owning-memory): TODO
    auto* circle = new Circle(arc, sweepLine());
    if (circle->mIsValid) {
      mCircleEvents.push(circle);
    } else {
      delete circle; // NOLINT(cppcoreguidelines-owning-memory): TODO
    }
  }
}

void VoronoiGenerator::finishEdges() {
  mSweepline = 2 * mMaxY;
  mBeachline.finish(mVoronoiEdges);
}
} // namespace csp::measurementtools
