#ifndef CSL_OGC_GML_COORDINATE_SYSTEMS
#define CSL_OGC_GML_COORDINATE_SYSTEMS

#include "BasicTypes.hpp"
#include "DeprecatedTypes.hpp"

#include <optional>
#include <string>
#include <vector>

#include "csl_ogc_export.hpp"

namespace ogc::schemas::gml {

struct CSL_OGC_EXPORT CoordinateSystemAxisType : IdentifiedObjectType {
  // attributes
  UomIdentifier uom;

  // child elements
  CodeType                             axisAbbrev;
  CodeWithAuthorityType                axisDirection;
  std::optional<double>                minimumValue;
  std::optional<double>                maximumValue;
  std::optional<CodeWithAuthorityType> rangeMeaning;
};

struct CSL_OGC_EXPORT CoordinateSystemAxisPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributeGroup;

  // child elements
  std::optional<CoordinateSystemAxisType> coordinateSystemAxis;
};

struct CSL_OGC_EXPORT AbstractCoordinateSystemType : IdentifiedObjectType {
  // attributes
  AggregationAttributeGroup aggregationAttributeGroup;

  // child elements
  std::vector<CoordinateSystemAxisPropertyType> axes;
};

struct CSL_OGC_EXPORT CoordinateSystemPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributeGroup;

  // child elements
  std::optional<AbstractCoordinateSystemType> abstractCoordinateSystem;
};

struct CSL_OGC_EXPORT EllipsoidalCSType : AbstractCoordinateSystemType {};

struct CSL_OGC_EXPORT EllipsoidalCSPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributeGroup;

  // child elements
  std::optional<EllipsoidalCSType> ellipsoidalCS;
};

struct CSL_OGC_EXPORT CartesianCSType : AbstractCoordinateSystemType {};

struct CSL_OGC_EXPORT CartesianCSPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributeGroup;

  // child elements
  std::optional<CartesianCSType> CartesianCS;
};

struct CSL_OGC_EXPORT VerticalCSType : AbstractCoordinateSystemType {};

struct CSL_OGC_EXPORT VerticalCSPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributeGroup;

  // child elements
  std::optional<VerticalCSType> VerticalCS;
};

struct CSL_OGC_EXPORT TimeCSType : AbstractCoordinateSystemType {};

struct CSL_OGC_EXPORT TimeCSPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributeGroup;

  // child elements
  std::optional<TimeCSType> TimeCS;
};

struct CSL_OGC_EXPORT LinearCSType : AbstractCoordinateSystemType {};

struct CSL_OGC_EXPORT LinearCSPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributeGroup;

  // child elements
  std::optional<LinearCSType> LinearCS;
};

struct CSL_OGC_EXPORT UserDefinedCSType : AbstractCoordinateSystemType {};

struct CSL_OGC_EXPORT UserDefinedCSPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributeGroup;

  // child elements
  std::optional<UserDefinedCSType> UserDefinedCS;
};

struct CSL_OGC_EXPORT SphericalCSType : AbstractCoordinateSystemType {};

struct CSL_OGC_EXPORT SphericalCSPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributeGroup;

  // child elements
  std::optional<SphericalCSType> SphericalCS;
};

struct CSL_OGC_EXPORT PolarCSType : AbstractCoordinateSystemType {};

struct CSL_OGC_EXPORT PolarCSPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributeGroup;

  // child elements
  std::optional<PolarCSType> PolarCS;
};

struct CSL_OGC_EXPORT CylindricalCSType : AbstractCoordinateSystemType {};

struct CSL_OGC_EXPORT CylindricalCSPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributeGroup;

  // child elements
  std::optional<CylindricalCSType> CylindricalCS;
};

struct CSL_OGC_EXPORT AffineCSType : AbstractCoordinateSystemType {};

struct CSL_OGC_EXPORT AffineCSPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributeGroup;

  // child elements
  std::optional<AffineCSType> AffineCS;
};



} // namespace ogc::schemas::gml

#endif // CSL_OGC_GML_COORDINATE_SYSTEMS
