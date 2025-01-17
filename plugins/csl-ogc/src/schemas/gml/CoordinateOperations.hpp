#ifndef CSL_OGC_GML_COORDINATE_OPERATIONS
#define CSL_OGC_GML_COORDINATE_OPERATIONS

#include "Base.hpp"

#include <optional>
#include <string>
#include <vector>

#include "csl_ogc_export.hpp"

namespace ogc::schemas::gml {

struct CoordinateOperationAccuracyType;
struct CRSPropertyType;
struct OperationMethodPropertyType;
struct AbstractGeneralParameterValuePropertyType;
struct OperationParameterPropertyType;
struct OperationParameterGroupPropertyType;
struct FormulaCitationType;
struct AbstractGeneralOperationParameterPropertyType;

struct CSL_OGC_EXPORT AbstractCoordinateOperationType : IdentifiedObjectType {
  // child elements
  std::optional<DomainOfValidityType>          domainOfValidity;
  std::vector<std::string>                     scope;
  std::optional<std::string>                   operationVersion;
  std::vector<CoordinateOperationAccuracyType> coordinateOperationAccuracy;
  std::optional<CRSPropertyType>               sourceCRS;
  std::optional<CRSPropertyType>               targetCRS;

  ~AbstractCoordinateOperationType() override = default;
};

struct CSL_OGC_EXPORT CoordinateOperationAccuracyType {
  // attributes
  AssociationAttributeGroup associationAttributes;

  // child elements
  std::optional<gmd::AbstractDQ_PositionalAccuracyType> todo; // TODO
};

struct CSL_OGC_EXPORT CoordinateOperationPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributes;

  // child elements
  std::optional<AbstractCoordinateOperationType> abstractCoordinateOperation;
};

struct CSL_OGC_EXPORT SingleOperationPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributes;

  // child elements
  std::optional<AbstractCoordinateOperationType> abstractSingleOperation;
};

struct CSL_OGC_EXPORT AbstractGeneralConversionType : AbstractCoordinateOperationType {
  // attributes
  std::string id;

  // child elements
  std::vector<MetaDataPropertyType> metaDataProperty;
  std::optional<StringOrRefType>    description;
  std::optional<ReferenceType>      descriptionReference;
  CodeWithAuthorityType             identifier;
  std::vector<CodeType>             name;
  std::optional<std::string>        remarks;

  ~AbstractGeneralConversionType() override = default;
};

struct CSL_OGC_EXPORT GeneralConversionPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributes;

  // child elements
  std::optional<AbstractGeneralConversionType> abstractGeneralConversion;
};

struct CSL_OGC_EXPORT AbstractGeneralTransformationType : AbstractCoordinateOperationType {
  // attributes
  std::string id;

  // child elements
  std::vector<MetaDataPropertyType> metaDataProperty;
  std::optional<StringOrRefType>    description;
  std::optional<ReferenceType>      descriptionReference;
  CodeWithAuthorityType             identifier;
  std::vector<CodeType>             name;
  std::optional<std::string>        remarks;

  ~AbstractGeneralTransformationType() override = default;
};

struct CSL_OGC_EXPORT GeneralTransformationPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributes;

  // child elements
  std::optional<AbstractGeneralTransformationType> abstractGeneralTransformation;
};

struct CSL_OGC_EXPORT ConcatenatedOperationType final : AbstractCoordinateOperationType {
  // attributes
  AggregationAttributeGroup aggregationAttributes;

  // child elements
  std::vector<CoordinateOperationPropertyType> coordOperations;

  ~ConcatenatedOperationType() final = default;
};

struct CSL_OGC_EXPORT ConcatenatedOperationPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributes;

  // child elements
  std::optional<ConcatenatedOperationType> concatenatedOperation;
};

struct CSL_OGC_EXPORT PassThroughOperationType final : AbstractCoordinateOperationType {
  // attributes
  AggregationAttributeGroup aggregationAttributes;

  // child elements
  std::vector<uint64_t>           modifiedCoordinate;
  CoordinateOperationPropertyType coordOperation;

  ~PassThroughOperationType() final = default;
};

struct CSL_OGC_EXPORT PassThroughOperationPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributes;

  // child elements
  std::optional<PassThroughOperationType> passThroughOperation;
};

struct CSL_OGC_EXPORT ConversionType final : AbstractGeneralConversionType {
  // child items
  OperationMethodPropertyType                            method;
  std::vector<AbstractGeneralParameterValuePropertyType> parameterValues;

  ~ConversionType() final = default;
};

struct CSL_OGC_EXPORT ConversionPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributes;

  // child elements
  std::optional<ConversionType> conversion;
};

struct CSL_OGC_EXPORT TransformationType final : AbstractGeneralTransformationType {
  // child items
  OperationMethodPropertyType                            method;
  std::vector<AbstractGeneralParameterValuePropertyType> parameterValues;

  ~TransformationType() final = default;
};

struct CSL_OGC_EXPORT TransformationPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributes;

  // child elements
  std::optional<TransformationType> passThroughOperation;
};

struct CSL_OGC_EXPORT AbstractGeneralParameterValueType {
  virtual ~AbstractGeneralParameterValueType() = default;
};

struct CSL_OGC_EXPORT AbstractGeneralParameterValuePropertyType {
  // child elements
  AbstractGeneralParameterValueType abstractGeneralParameterValue;
};

struct CSL_OGC_EXPORT ParameterValueType final : AbstractGeneralParameterValueType {
  // child elements
  std::variant<MeasureType, DMSAngleType, std::string, uint64_t, bool, MeasureListType, IntegerList>
                                 value;
  OperationParameterPropertyType operationParameter;
};

struct CSL_OGC_EXPORT ParameterValueGroupType final : AbstractGeneralParameterValueType {
  // child elements
  std::vector<AbstractGeneralParameterValuePropertyType> parameterValue;
  OperationParameterGroupPropertyType                    group;
};

struct CSL_OGC_EXPORT OperationMethodType final : IdentifiedObjectType {
  // child elements
  std::variant<FormulaCitationType, CodeType>                formula;
  std::optional<uint64_t>                                    sourceDimensions;
  std::optional<uint64_t>                                    targetDimensions;
  std::vector<AbstractGeneralOperationParameterPropertyType> parameter;
};

struct CSL_OGC_EXPORT FormulaCitationType {
  // attributes
  AssociationAttributeGroup associationAttributes;

  // child elements
  std::optional<gmd::CI_Citation> todo; // TODO
};

struct CSL_OGC_EXPORT OperationMethodPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributes;

  // child elements
  std::optional<OperationMethodType> operationMethod;
};

struct CSL_OGC_EXPORT AbstractGeneralOperationParameterType : IdentifiedObjectType {
  // child elements
  std::optional<uint64_t> minimumOccurs;

  ~AbstractGeneralOperationParameterType() override = default;
};

struct CSL_OGC_EXPORT AbstractGeneralOperationParameterPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributes;

  // child elements
  std::optional<AbstractGeneralOperationParameterType> abstractGeneralOperationParameter;

  virtual ~AbstractGeneralOperationParameterPropertyType() = default;
};

struct CSL_OGC_EXPORT OperationParameterType final : AbstractGeneralOperationParameterType {
  ~OperationParameterType() final = default;
};

struct CSL_OGC_EXPORT OperationParameterPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributes;

  // child elements
  std::optional<OperationParameterType> operationParameter;
};

struct CSL_OGC_EXPORT OperationParameterGroupType final : AbstractGeneralOperationParameterType {
  // child elements
  std::optional<uint64_t>                                    maximumOccurs;
  std::vector<AbstractGeneralOperationParameterPropertyType> parameter;
};

struct CSL_OGC_EXPORT OperationParameterGroupPropertyType {
  // attributes
  AssociationAttributeGroup associationAttributes;

  // child elements
  std::optional<OperationParameterGroupType> operationParameter;
};

} // namespace ogc::schemas::gml

#endif // CSL_OGC_GML_COORDINATE_OPERATIONS
