#ifndef CSL_OGC_GML_DEPRECATED_TYPES
#define CSL_OGC_GML_DEPRECATED_TYPES

#include <any>
#include <optional>

#include "Feature.hpp"
#include "Base.hpp"

#include "csl_ogc_export.hpp"

namespace ogc::schemas::gml {

struct AbstractMetaDataType;
struct DegreesType;
struct DefinitionProxyType;

struct CSL_OGC_EXPORT OperationPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributes;

  // child elements
  std::optional<AbstractCoordinateOperationType> abstractOperation;
};

struct CSL_OGC_EXPORT TemporalCSType final : AbstractCoordinateSystemType {};

struct CSL_OGC_EXPORT TemporalCSPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributes;

  // child elements
  std::optional<TemporalCSType> temporalCS;
};

struct CSL_OGC_EXPORT ObliqueCartesianCSType final : AbstractCoordinateSystemType {};

struct CSL_OGC_EXPORT ObliqueCartesianCSPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributes;

  // child elements
  std::optional<ObliqueCartesianCSType> obliqueCartesianCS;
};

struct CSL_OGC_EXPORT GeographicCRSType final : AbstractCRSType {
  // child elements
  EllipsoidalCSPropertyType usesEllipsoidalCS;
  GeodeticDatumPropertyType usesGeodeticDatum;
};

struct CSL_OGC_EXPORT GeographicCRSPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributes;

  // child elements
  std::optional<GeographicCRSType> geographicCRS;
};

struct CSL_OGC_EXPORT GeocentricCRSType final : AbstractCRSType {
  // child elements
  std::variant<CartesianCSPropertyType, SphericalCSPropertyType> usesCSProperty;
  GeodeticDatumPropertyType                                      usesGeodeticDatum;
};

struct CSL_OGC_EXPORT GeocentricCRSPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributes;

  // child elements
  std::optional<GeocentricCRSType> geocentricCRS;
};

enum class SuccessionType { SUBSTITUTION, DIVISION, FUSION, INITIATION };

using DecimalMinutesType = double;
using ArcMinutesType     = uint8_t;
using ArcSecondsType     = double;

struct CSL_OGC_EXPORT DMSAngleType {
  struct ArcFractionsType {
    ArcMinutesType                minutes;
    std::optional<ArcSecondsType> seconds;
  };

  // child elements
  DegreesType degrees;

  std::variant<std::monostate, DecimalMinutesType, ArcFractionsType> angle;
};

using DegreeValueType = uint16_t;

struct DegreesType {
  // attributes
  enum class DirectionType {
    N,
    E,
    S,
    W,
    PLUS,
    MINUS,
  } direction;

  // content
  DegreeValueType degrees;
};

using AngleChoiceType = std::variant<AngleType, DMSAngleType>;

struct CSL_OGC_EXPORT ArrayAssociationType {
  // attributes
  OwnershipAttributeGroup ownershipAttributes;

  // child elements
  std::vector<std::any> abstractObjects;
};

struct CSL_OGC_EXPORT StringOrRefType {
  // attributes
  AssociationAttributeGroup associationAttributes;

  // content
  std::string content;
};

struct CSL_OGC_EXPORT BagType final : AbstractGMLType {
  // child elements
  std::vector<AssociationRoleType>    members;
  std::optional<ArrayAssociationType> membersArray;
};

struct CSL_OGC_EXPORT ArrayType final : AbstractGMLType {
  // child elements
  std::optional<ArrayAssociationType> membersArray;
};

struct CSL_OGC_EXPORT MetaDataPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributes;
  std::string               about;

  // child elements
  std::optional<AbstractMetaDataType> abstractMetaData;
};

struct CSL_OGC_EXPORT AbstractMetaDataType {
  // attributes
  std::string id;
};

struct CSL_OGC_EXPORT GenericMetaDataType : AbstractMetaDataType {
  // child elements
  std::vector<std::any> children;
};

struct CSL_OGC_EXPORT LocationPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributes;

  // child elements
  std::variant<AbstractGeometryType, CodeType, StringOrRefType, NilReasonType> locationProperty;
};

struct CSL_OGC_EXPORT PriorityLocationPropertyType final : LocationPropertyType {
  // attributes
  std::string priority;
};

struct CSL_OGC_EXPORT FeatureArrayPropertyType {
  // child elements
  std::vector<AbstractFeatureType> abstractFeatures;
};

struct CSL_OGC_EXPORT BoundedFeatureType : AbstractFeatureType {
  // child elements
  StandardObjectProperties standardObjectProperties;
};

struct CSL_OGC_EXPORT AbstractFeatureCollectionType : AbstractFeatureType {
  // child elements
  std::vector<FeaturePropertyType>        featureMember;
  std::optional<FeatureArrayPropertyType> featureMemberArray;
};

struct CSL_OGC_EXPORT FeatureCollectionType final : AbstractFeatureCollectionType {};

struct CSL_OGC_EXPORT IndirectEntryType {
  // child elements
  DefinitionProxyType definitionProxy;
};

struct CSL_OGC_EXPORT DefinitionProxyType final : DefinitionType {
  // child elements
  ReferenceType definitionRef;
};

enum class CSL_OGC_EXPORT IncrementOrder {
  PLUS_X_PLUS_Y,
  PLUS_Y_PLUS_X,
  PLUS_X_MINUS_Y,
  MINUS_X_MINUS_Y,
};

struct CSL_OGC_EXPORT MovingObjectStatusType final : AbstractTimeSliceType {
  // child elements
  std::variant<GeometryPropertyType, DirectPositionType, CodeType, ReferenceType,
      LocationPropertyType>
      place;

  std::optional<MeasureType>           speed;
  std::optional<DirectionPropertyType> bearing;
  std::optional<MeasureType>           acceleration;
  std::optional<MeasureType>           elevation;
  std::optional<StringOrRefType>       status;
  std::optional<ReferenceType>         statusReference;
};

} // namespace ogc::schemas::gml

#endif // CSL_OGC_GML_DEPRECATED_TYPES
