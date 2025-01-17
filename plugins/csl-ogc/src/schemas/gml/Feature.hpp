#ifndef CSL_OGC_GML_FEATURE
#define CSL_OGC_GML_FEATURE

#include "Base.hpp"
#include "DeprecatedTypes.hpp"

#include <optional>
#include <string>

#include "csl_ogc_export.hpp"

namespace ogc::schemas::gml {

struct BoundingShapeType;

struct CSL_OGC_EXPORT AbstractFeatureType : AbstractGMLType {
  // child elements
  std::optional<BoundingShapeType> boundedBy;
  std::optional<LocationPropertyType> location;

  ~AbstractFeatureType() override = default;
};

struct CSL_OGC_EXPORT FeaturePropertyType {
  // attributes
  OwnershipAttributeGroup ownershipAttributes;
  AssociationAttributeGroup associationAttributes;

  // child elements
  std::optional<AbstractFeatureType> abstractFeature;
};

struct CSL_OGC_EXPORT BoundingShapeType {
  // attributes
  std::optional<NilReasonType> nilReason;

  // child items
  std::variant<EnvelopeType, NilReasonType> envelope;
};

struct CSL_OGC_EXPORT EnvelopeWithTimePeriodType final : EnvelopeType {
  // attributes
  std::string frame = "#ISO-8601";

  // child elements
  TimePositionType beginPosition;
  TimePositionType endPosition;
};

struct CSL_OGC_EXPORT AbstractFeatureMemberType {
  // attributes
  OwnershipAttributeGroup ownershipAttributes;

  virtual ~AbstractFeatureMemberType() = default;
};

} // namespace ogc::schemas::gml

#endif // CSL_OGC_GML_FEATURE
