////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#include "APITypes.hpp"

#include "../../../src/cs-core/Settings.hpp"
#include <nlohmann/json.hpp>

namespace csp::virtualsatellite {

void from_json(nlohmann::json const& j, BeanStructuralElementInstanceReference& o) {
  cs::core::Settings::deserialize(j, "uuid", o.uuid);
  cs::core::Settings::deserialize(j, "name", o.name);
}

void from_json(nlohmann::json const& j, BeanCategoryAssignmentReference& o) {
  cs::core::Settings::deserialize(j, "uuid", o.uuid);
  cs::core::Settings::deserialize(j, "name", o.name);
}

void from_json(nlohmann::json const& j, BeanStructuralElementInstance& o) {
  cs::core::Settings::deserialize(j, "uuid", o.uuid);
  cs::core::Settings::deserialize(j, "name", o.name);
  cs::core::Settings::deserialize(j, "type", o.type);
  if (!j.at("parent").is_null()) {
    cs::core::Settings::deserialize(j, "parent", o.parent);
  }
  cs::core::Settings::deserialize(j, "assignedDiscipline", o.assignedDiscipline);
  cs::core::Settings::deserialize(j, "children", o.children);
  cs::core::Settings::deserialize(j, "superSeis", o.superSeis);
  cs::core::Settings::deserialize(j, "categoryAssignments", o.categoryAssignments);
}

void from_json(nlohmann::json const& j, BeanProperty& o) {
  cs::core::Settings::deserialize(j, "uuid", o.uuid);
  cs::core::Settings::deserialize(j, "isCalculated", o.isCalculated);
  cs::core::Settings::deserialize(j, "override", o.override);
  if (!j.contains("unitBean") || !j.at("unitBean").is_null()) {
    cs::core::Settings::deserialize(j, "unitBean", o.unitBean);
  }

  cs::core::Settings::deserialize(j, "propertyType", o.propertyType);
  if (j.at("value").is_null()) {
    o.value = std::monostate();
  } else if (o.propertyType == "int") {
    int64_t value;
    cs::core::Settings::deserialize(j, "value", value);
    o.value = value;
  } else if (o.propertyType == "float") {
    float value;
    cs::core::Settings::deserialize(j, "value", value);
    o.value = value;
  } else if (o.propertyType == "enum" || o.propertyType == "resource") {
    std::string value;
    cs::core::Settings::deserialize(j, "value", value);
    o.value = value;
  }
}

void from_json(nlohmann::json const& j, CategoryAssignment& o) {
  cs::core::Settings::deserialize(j, "uuid", o.uuid);
  cs::core::Settings::deserialize(j, "name", o.name);
  cs::core::Settings::deserialize(j, "type", o.type);

  cs::core::Settings::deserialize(j, "colorBean", o.colorBean);
  cs::core::Settings::deserialize(j, "geometryFileBean", o.geometryFileBean);
  cs::core::Settings::deserialize(j, "positionXBean", o.positionXBean);
  cs::core::Settings::deserialize(j, "positionYBean", o.positionYBean);
  cs::core::Settings::deserialize(j, "positionZBean", o.positionZBean);
  cs::core::Settings::deserialize(j, "radiusBean", o.radiusBean);
  cs::core::Settings::deserialize(j, "rotationXBean", o.rotationXBean);
  cs::core::Settings::deserialize(j, "rotationYBean", o.rotationYBean);
  cs::core::Settings::deserialize(j, "rotationZBean", o.rotationZBean);
  cs::core::Settings::deserialize(j, "shapeBean", o.shapeBean);
  cs::core::Settings::deserialize(j, "sizeXBean", o.sizeXBean);
  cs::core::Settings::deserialize(j, "sizeYBean", o.sizeYBean);
  cs::core::Settings::deserialize(j, "sizeZBean", o.sizeZBean);
  cs::core::Settings::deserialize(j, "transparencyBean", o.transparencyBean);
}

} // namespace csp::virtualsatellite