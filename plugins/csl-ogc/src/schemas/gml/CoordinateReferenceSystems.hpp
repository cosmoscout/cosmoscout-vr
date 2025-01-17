#ifndef CSL_OGC_GML_COORDINATE_REFERENCE_SYSTEMS
#define CSL_OGC_GML_COORDINATE_REFERENCE_SYSTEMS

#include "BasicTypes.hpp"
#include "CoordinateOperations.hpp"
#include "DeprecatedTypes.hpp"

#include <optional>
#include <string>
#include <vector>

#include "csl_ogc_export.hpp"

namespace ogc::schemas::gml {

struct CSL_OGC_EXPORT SingleCRSPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributeGroup;

  // child elements
  std::optional<AbstractCRSType> abstractSingleCRS;
};

struct CSL_OGC_EXPORT AbstractGeneralDerivedCRSType : AbstractCRSType {
  // child elements
  GeneralConversionPropertyType conversion;
};

struct CSL_OGC_EXPORT CompoundCRSType : AbstractCRSType {
  // attributes
  AggregationAttributeGroup aggregationAttributeGroup;

  // child elements
  std::vector<SingleCRSPropertyType> componentReferenceSystems;
};

struct CSL_OGC_EXPORT CompoundCRSPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributeGroup;

  // child elements
  std::optional<CompoundCRSType> compoundCRS;
};

struct CSL_OGC_EXPORT GeodeticCRSType : AbstractCRSType {
  // child elements
  std::variant<EllipsoidalCSPropertyType, CartesianCSPropertyType, SphericalCSPropertyType>
                            coordinateSystem;
  GeodeticDatumPropertyType geodeticDatum;
};

struct CSL_OGC_EXPORT GeodeticCRSPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributeGroup;

  // child elements
  std::optional<GeodeticCRSType> geodeticCRS;
};

struct CSL_OGC_EXPORT VerticalCRSType : AbstractCRSType {
  // child elements
  VerticalCSPropertyType    verticalCS;
  VerticalDatumPropertyType verticalDatum;
};

struct CSL_OGC_EXPORT VerticalCRSPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributeGroup;

  // child elements
  std::optional<VerticalCRSType> verticalCRS;
};

struct CSL_OGC_EXPORT ProjectedCRSType : AbstractGeneralDerivedCRSType {
  // child elements
  std::variant<GeodeticCRSPropertyType, GeographicCRSPropertyType> baseCRS;
  CartesianCSPropertyType                                          cartesianCS;
};

struct CSL_OGC_EXPORT ProjectedCRSPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributeGroup;

  // child elements
  std::optional<ProjectedCRSType> projectedCRS;
};

struct CSL_OGC_EXPORT DerivedCRSType : AbstractGeneralDerivedCRSType {
  // child elements
  SingleCRSPropertyType        baseCRS;
  CodeWithAuthorityType        derivedCRSType;
  CoordinateSystemPropertyType coordinateSystem;
};

struct CSL_OGC_EXPORT DerivedCRSPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributeGroup;

  // child elements
  std::optional<DerivedCRSType> derivedCRS;
};

struct CSL_OGC_EXPORT EngineeringCRSType : AbstractCRSType {
  // child elements
  std::variant<AffineCSPropertyType, CartesianCSPropertyType, CylindricalCSPropertyType,
      LinearCSPropertyType, PolarCSPropertyType, SphericalCSPropertyType, UserDefinedCSPropertyType,
      CoordinateSystemPropertyType>
      coordinateSystem;

  EngineeringDatumPropertyType engineeringDatum;
};

struct EngineeringCRSPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributeGroup;

  // child elements
  std::optional<EngineeringCRSType> engineeringCRS;
};

struct CSL_OGC_EXPORT ImageCRSType : AbstractCRSType {
  // child elements
  std::variant<CartesianCSPropertyType, AffineCSPropertyType, ObliqueCartesianCSPropertyType>
      coordinateSystem;

  ImageDatumPropertyType imageDatum;
};

struct CSL_OGC_EXPORT ImageCRSPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributeGroup;

  // child elements
  std::optional<ImageCRSType> imageCRS;
};

struct CSL_OGC_EXPORT TemporalCRSType : AbstractCRSType {
  // child elements
  std::variant<TimeCSPropertyType, TemporalCSPropertyType> coordinateSystem;
  TemporalDatumPropertyType temporalDatum;
};

struct CSL_OGC_EXPORT TemporalCRSPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributeGroup;

  // child elements
  std::optional<TemporalCRSType> temporalCRS;
};

} // namespace ogc::schemas::gml

#endif // CSL_OGC_GML_COORDINATE_REFERENCE_SYSTEMS
