#ifndef CSL_OGC_GMLCOV_COVERAGE
#define CSL_OGC_GMLCOV_COVERAGE

#include <optional>
#include <string>
#include <vector>
#include <any>

#include "csl_ogc_export.hpp"

namespace ogc::schemas::gmlcov {

struct ExtensionType;
struct AbstractCoverageType;
struct AbstractDiscreteCoverageType;
struct AbstractContinuousCoverageType;
struct MetadataType;

using DataRecordPropertyType = std::any;  // Placeholder for the actual type.
using AbstractMetadataPropertyType = std::any;  // Placeholder for the actual type.

struct CSL_OGC_EXPORT AbstractCoverageType {
    // child elements
    std::optional<std::any> coverageFunction;  // Placeholder for the actual gml:coverageFunction type.
    DataRecordPropertyType rangeType;
    std::vector<MetadataType> metadata;
};

struct CSL_OGC_EXPORT MetadataType : public AbstractMetadataPropertyType {
    // child elements
    std::optional<ExtensionType> extension;
    // attributes
    // This inherits gml:AssociationAttributeGroup attributes.
};

struct CSL_OGC_EXPORT ExtensionType {
    // child elements
    std::vector<std::any> children;
};

struct CSL_OGC_EXPORT AbstractDiscreteCoverageType : public AbstractCoverageType {
    // No additional elements, inherits everything from AbstractCoverageType.
};

struct CSL_OGC_EXPORT AbstractContinuousCoverageType : public AbstractCoverageType {
    // No additional elements, inherits everything from AbstractCoverageType.
};

struct CSL_OGC_EXPORT MultiPointCoverage : public AbstractDiscreteCoverageType {
    // Inherits everything from AbstractDiscreteCoverageType.
};

struct CSL_OGC_EXPORT MultiCurveCoverage : public AbstractDiscreteCoverageType {
    // Inherits everything from AbstractDiscreteCoverageType.
};

struct CSL_OGC_EXPORT MultiSurfaceCoverage : public AbstractDiscreteCoverageType {
    // Inherits everything from AbstractDiscreteCoverageType.
};

struct CSL_OGC_EXPORT MultiSolidCoverage : public AbstractDiscreteCoverageType {
    // Inherits everything from AbstractDiscreteCoverageType.
};

struct CSL_OGC_EXPORT GridCoverage : public AbstractDiscreteCoverageType {
    // Inherits everything from AbstractDiscreteCoverageType.
};

struct CSL_OGC_EXPORT RectifiedGridCoverage : public AbstractDiscreteCoverageType {
    // Inherits everything from AbstractDiscreteCoverageType.
};

struct CSL_OGC_EXPORT ReferenceableGridCoverage : public AbstractDiscreteCoverageType {
    // Inherits everything from AbstractDiscreteCoverageType.
};

} // namespace ogc::schemas::gmlcov

#endif // CSL_OGC_GMLCOV_COVERAGE
