#ifndef CSL_OGC_GML_BASE
#define CSL_OGC_GML_BASE

#include "BasicTypes.hpp"
#include "DeprecatedTypes.hpp"

#include <optional>
#include <string>
#include <vector>

#include "../xlink/XLink.hpp"

#include "csl_ogc_export.hpp"

namespace ogc::schemas::gml {

struct StandardObjectProperties;
struct ReferenceType;

struct CSL_OGC_EXPORT AssociationAttributeGroup {
  // attributes
  xlink::SimpleAttrsGroup simpleAttributes;

  std::optional<NilReasonType> nilReason;
  std::optional<std::string>   remoteSchema;
};

struct CSL_OGC_EXPORT AbstractGMLType {
  // attributes
  std::string id;

  // child elements
  StandardObjectProperties standardObjectProperties;

  virtual ~AbstractGMLType() = default;
};

struct CSL_OGC_EXPORT StandardObjectProperties {
  // child elements
  std::vector<MetaDataPropertyType>    metaDataProperty;
  std::optional<StringOrRefType>       description;
  std::optional<ReferenceType>         descriptionReference;
  std::optional<CodeWithAuthorityType> identifier;
  std::vector<CodeType>                name;
};

struct CSL_OGC_EXPORT OwnershipAttributeGroup {
  // attributes
  bool owns = false;
};

struct AssociationRoleType {
  // attributes
  AssociationAttributeGroup associationAttributes;
  OwnershipAttributeGroup   ownershipAttributes;

  // content
  std::optional<std::any> content;
};

struct CSL_OGC_EXPORT ReferenceType {
  // attributes
  OwnershipAttributeGroup   ownershipAttributes;
  AssociationAttributeGroup associationAttributes;
};

struct CSL_OGC_EXPORT InlinePropertyType {
  // attributes
  OwnershipAttributeGroup ownershipAttributes;

  // content
  std::any content;
};

struct CSL_OGC_EXPORT AbstractMemberType {
  // attributes
  OwnershipAttributeGroup ownershipAttributes;

  virtual ~AbstractMemberType() = default;
};

enum class AggregationType { SET, BAG, SEQUENCE, ARRAY, RECORD, TABLE };

struct CSL_OGC_EXPORT AggregationAttributeGroup {
  // attributes
  AggregationType aggregationType;
};

struct CSL_OGC_EXPORT AbstractMetadataPropertyType {
  // attributes
  OwnershipAttributeGroup ownershipAttributes;

  virtual ~AbstractMetadataPropertyType() = default;
};

} // namespace ogc::schemas::gml

#endif // CSL_OGC_GML_BASE
