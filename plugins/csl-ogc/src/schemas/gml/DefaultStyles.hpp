#ifndef CSL_OGC_GML_DEFAULT_STYLES
#define CSL_OGC_GML_DEFAULT_STYLES

#include "Base.hpp"

#include <any>
#include <optional>
#include <string>
#include <vector>

#include "csl_ogc_export.hpp"

namespace ogc::schemas::gml {

struct AbstractStyleType;
struct FeatureStylePropertyType;
struct GraphStylePropertyType;
struct FeatureStyleType;
struct GeometryStylePropertyType;
struct TopologyStylePropertyType;
struct LabelStylePropertyType;
struct StyleVariationType;
struct GeometryStyleType;
struct SymbolType;
struct TopologyStyleType;
struct LabelStyleType;
struct LabelType;
struct GraphStyleType;
struct GraphTypeType;
struct DrawingTypeType;
struct LineTypeType;
struct AesheticCriteriaType;

struct CSL_OGC_EXPORT DefaultStylePropertyType {
  // attributes
  std::optional<std::string> about;
  AssociationAttributeGroup  associationAttributeGroup;

  // child elements
  AbstractStyleType abstractStyle;
};

struct CSL_OGC_EXPORT AbstractStyleType : AbstractGMLType {};

struct CSL_OGC_EXPORT StyleType : AbstractStyleType {
  // child elements
  std::vector<FeatureStylePropertyType> featureStyles;
  std::optional<GraphStylePropertyType> graphStyle;
};

struct CSL_OGC_EXPORT FeatureStylePropertyType {
  // attributes
  std::optional<std::string> about;
  AssociationAttributeGroup  associationAttributeGroup;

  // child elements
  std::optional<FeatureStyleType> featureStyle;
};

enum struct CSL_OGC_EXPORT QueryGrammarEnumeration { XPATH, XQUERY, OTHER };

struct CSL_OGC_EXPORT FeatureStyleType : AbstractGMLType {
  // attributes
  std::optional<std::string>             featureType;
  std::optional<std::string>             baseType;
  std::optional<QueryGrammarEnumeration> queryGrammar;

  // child elements
  std::optional<std::string>             featureConstraints;
  std::vector<GeometryStylePropertyType> geometryStyles;
  std::vector<TopologyStylePropertyType> topologyStyles;
  std::optional<LabelStylePropertyType>  labelStyle;
};

struct CSL_OGC_EXPORT BaseStyleDescriptorType : AbstractGMLType {
  // child elements
  std::optional<ScaleType>        spatialResolution;
  std::vector<StyleVariationType> styleVariations;
  // TODO: smil20 elements
};

struct CSL_OGC_EXPORT GeometryStylePropertyType {
  // attributes
  std::optional<std::string> about;
  AssociationAttributeGroup  associationAttributeGroup;

  // child elements
  std::optional<GeometryStyleType> geometryStyle;
};

struct CSL_OGC_EXPORT GeometryStyleType : BaseStyleDescriptorType {
  // attributes
  std::optional<std::string> geometryProperty;
  std::optional<std::string> geometryType;

  // child elements
  std::variant<SymbolType, std::string> style;
  std::optional<LabelStylePropertyType> labelStyle;
};

struct CSL_OGC_EXPORT TopologyStylePropertyType {
  // attributes
  std::optional<std::string> about;
  AssociationAttributeGroup  associationAttributeGroup;

  // child elements
  std::optional<TopologyStyleType> topologyStyle;
};

struct CSL_OGC_EXPORT TopologyStyleType : BaseStyleDescriptorType {
  // attributes
  std::optional<std::string> topologyProperty;
  std::optional<std::string> topologyType;

  // child elements
  std::variant<SymbolType, std::string> style;
  std::optional<LabelStylePropertyType> labelStyle;
};

struct CSL_OGC_EXPORT LabelStylePropertyType {
  // attributes
  std::optional<std::string> about;
  AssociationAttributeGroup  associationAttributeGroup;

  // child elements
  std::optional<LabelStyleType> labelStyle;
};

struct CSL_OGC_EXPORT LabelStyleType : BaseStyleDescriptorType {
  // child elements
  std::string style;
  LabelType   label;
};

struct CSL_OGC_EXPORT GraphStylePropertyType {
  // attributes
  std::optional<std::string> about;
  AssociationAttributeGroup  associationAttributeGroup;

  // child elements
  std::optional<GraphStyleType> graphStyle;
};

struct CSL_OGC_EXPORT GraphStyleType : BaseStyleDescriptorType {
  // child elements
  std::optional<bool>               planar;
  std::optional<bool>               directed;
  std::optional<bool>               grid;
  std::optional<double>             minDistance;
  std::optional<double>             minAngle;
  std::optional<GraphTypeType>      graphType;
  std::optional<DrawingTypeType>    drawingType;
  std::optional<LineTypeType>       lineType;
  std::vector<AesheticCriteriaType> aestheticCriteria;
};

} // namespace ogc::schemas::gml

#endif // CSL_OGC_GML_DEFAULT_STYLES
