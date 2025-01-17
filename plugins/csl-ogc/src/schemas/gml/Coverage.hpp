#ifndef CSL_OGC_GML_COVERAGE
#define CSL_OGC_GML_COVERAGE

#include "Feature.hpp"

#include <any>
#include <optional>
#include <string>
#include <vector>

#include "csl_ogc_export.hpp"

namespace ogc::schemas::gml {

struct DomainSetType;
struct RangeSetType;
struct CoverageFunctionType;
struct DataBlockType;
struct FileType;
struct MappingRuleType;
struct GridFunctionType;
struct SequenceRuleType;

struct CSL_OGC_EXPORT AbstractCoverageType : AbstractFeatureType {
  // child elements
  DomainSetType domainSet;
  RangeSetType  rangeSet;
};

struct CSL_OGC_EXPORT DiscreteCoverageType : AbstractCoverageType {
  // child elements
  std::optional<CoverageFunctionType> coverageFunction;
};

struct CSL_OGC_EXPORT AbstractContinuousCoverageType : AbstractCoverageType {
  // child elements
  std::optional<CoverageFunctionType> coverageFunction;
};

struct CSL_OGC_EXPORT DomainSetType {
  // attributes
  OwnershipAttributeGroup   ownershipAttributeGroup;
  AssociationAttributeGroup associationAttributeGroup;

  // child elements
  std::variant<std::monostate, AbstractGeometryType, AbstractTimeObjectType> domainSet;
};

struct CSL_OGC_EXPORT RangeSetType {
  // child elements
  std::optional<std::vector<ValueArrayType>, std::vector<std::any>, DataBlockType, FileType>
      rangeSet;
};

struct CSL_OGC_EXPORT DataBlockType {
  // child elements
  AssociationRoleType                                  rangeParameters;
  std::variant<CoordinatesType, DoubleOrNilReasonList> data;
};

struct CSL_OGC_EXPORT FileType {
  // child elements
  AssociationRoleType        rangeParameters;
  std::string                file;
  CodeType                   fileStructure;
  std::optional<std::string> mimeType;
  std::optional<std::string> compression;
};

struct CSL_OGC_EXPORT CoverageFunctionType {
  // child elements
  StringOrRefType  mappingRule;
  MappingRuleType  coverageMappingRule;
  GridFunctionType gridFunction;
};

struct CSL_OGC_EXPORT MappingRuleType {
  // child elements
  std::variant<std::string, ReferenceType> rule;
};

struct CSL_OGC_EXPORT GridFunctionType {
  // child elements
  std::optional<SequenceRuleType> sequenceRule;
  std::optional<IntegerList>      startPoint;
};

enum class SequenceRuleEnumeration {
  LINEAR,
  BOUSTROPHEDONIC,
  CENTER_DIAGONAL,
  SPIRAL,
  MORTON,
  HILBERT
};

using AxisDirection     = std::string;
using AxisDirectionList = std::vector<AxisDirection>;

struct CSL_OGC_EXPORT SequenceRuleType {
  // attributes
  std::optional<IncrementOrder>    order;
  std::optional<AxisDirectionList> axisOrder;

  // content
  SequenceRuleEnumeration content;
};

} // namespace ogc::schemas::gml

#endif // CSL_OGC_GML_COVERAGE
