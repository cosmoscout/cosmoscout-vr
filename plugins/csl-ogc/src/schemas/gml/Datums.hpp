#ifndef CSL_OGC_GML_DATUMS
#define CSL_OGC_GML_DATUMS

#include "Feature.hpp"

#include <any>
#include <optional>
#include <string>
#include <vector>

#include "csl_ogc_export.hpp"

namespace ogc::schemas::gml {

struct CSL_OGC_EXPORT AbstractDatumType : IdentifiedObjectType {
  // child elements
  std::optional<DomainOfValidity> domainOfValidity;
  std::vector<std::string>        scope;
  std::optional<CodeType>         anchorDefinition;
  std::optional<std::string>      realizationEpoch;
};

struct CSL_OGC_EXPORT DatumPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributeGroup;

  // child elements
  std::optional<AbstractDatumType> abstractDatum;
};

struct CSL_OGC_EXPORT GeodeticDatumType : AbstractDatumType {
  // child elements
  PrimeMeridianPropertyType primeMeridian;
  EllipsoidPropertyType     ellipsoid;
};

struct CSL_OGC_EXPORT GeodeticDatumPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributeGroup;

  // child elements
  std::optional<GeodeticDatumType> geodeticDatum;
};

using SecondDefiningParameter = std::variant<MeasureType, LengthType, bool>;

struct CSL_OGC_EXPORT EllipsoidType : IdentifiedObjectType {
  // child elements
  MeasureType             semiMajorAxis;
  SecondDefiningParameter secondDefiningParameter;
};

struct CSL_OGC_EXPORT EllipsoidPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributeGroup;

  // child elements
  std::optional<EllipsoidType> ellipsoid;
};

struct CSL_OGC_EXPORT PrimeMeridianType : IdentifiedObjectType {
  // child elements
  AngleType greenwichLongitude;
};

struct CSL_OGC_EXPORT PrimeMeridianPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributeGroup;

  // child elements
  std::optional<PrimeMeridianType> primeMeridian;
};

struct CSL_OGC_EXPORT EngineeringDatumType : AbstractDatumType {};

struct CSL_OGC_EXPORT EngineeringDatumPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributeGroup;

  // child elements
  std::optional<EngineeringDatumType> engineeringDatum;
};

struct CSL_OGC_EXPORT ImageDatumType : AbstractDatumType {
  // child elements
  CodeWithAuthorityType pixelInCell;
};

struct CSL_OGC_EXPORT ImageDatumPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributeGroup;

  // child elements
  std::optional<ImageDatumType> imageDatum;
};

struct CSL_OGC_EXPORT VerticalDatumType : AbstractDatumType {};

struct CSL_OGC_EXPORT VerticalDatumPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributeGroup;

  // child elements
  std::optional<VerticalDatumType> verticalDatum;
};

struct CSL_OGC_EXPORT TemporalDatumBaseType : AbstractDatumType {
  // attributes
  std::string id;

  // child elements
  std::vector<MetaDataPropertyType>   metaDataProperties;
  std::optional<StringOrRefType>      description;
  std::optional<ReferenceType>        descriptionReference;
  CodeWithAuthorityType               identifier;
  std::vector<CodeType>               names;
  std::optional<std::string>          remarks;
  std::optional<DomainOfValidityType> domainOfValidity;
  std::vector<std::string>            scope;
};

struct CSL_OGC_EXPORT TemporalDatumType : TemporalDatumBaseType {
  // child elements
  std::string origin;
};

struct CSL_OGC_EXPORT TemporalDatumPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributeGroup;

  // child elements
  std::optional<TemporalDatumType> temporalDatum;
};

} // namespace ogc::schemas::gml

#endif // CSL_OGC_GML_DATUMS
