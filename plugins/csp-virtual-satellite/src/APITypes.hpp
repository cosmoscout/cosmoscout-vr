////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSP_VIRTUAL_SATELLITE_API_TYPES_HPP
#define CSP_VIRTUAL_SATELLITE_API_TYPES_HPP

#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace csp::virtualsatellite {

struct BeanStructuralElementInstanceReference {
  std::string uuid;
  std::string name;
};

struct BeanCategoryAssignmentReference {
  std::string uuid;
  std::string name;
};

struct BeanStructuralElementInstance {
  std::string                uuid;
  std::string                name;
  std::optional<std::string> type;
  std::optional<std::string> parent;
  std::string                assignedDiscipline;

  std::vector<BeanStructuralElementInstanceReference> children;
  std::vector<BeanStructuralElementInstanceReference> superSeis;

  std::vector<BeanCategoryAssignmentReference> categoryAssignments;
};

struct BeanProperty {
  std::string                uuid;
  bool                       isCalculated;
  bool                       override;
  std::optional<std::string> unitBean;
  std::string                propertyType;

  std::variant<std::monostate, int64_t, float, std::string> value;
};

struct CategoryAssignment {
  std::string                uuid;
  std::string                name;
  std::optional<std::string> type;

  BeanProperty colorBean;
  BeanProperty geometryFileBean;
  BeanProperty positionXBean;
  BeanProperty positionYBean;
  BeanProperty positionZBean;
  BeanProperty radiusBean;
  BeanProperty rotationXBean;
  BeanProperty rotationYBean;
  BeanProperty rotationZBean;
  BeanProperty shapeBean;
  BeanProperty sizeXBean;
  BeanProperty sizeYBean;
  BeanProperty sizeZBean;
  BeanProperty transparencyBean;
};

}; // namespace csp::virtualsatellite

#endif // CSP_VIRTUAL_SATELLITE_API_TYPES_HPP
