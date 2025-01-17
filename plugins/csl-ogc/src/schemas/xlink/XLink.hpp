#ifndef CSL_OGC_XLINK
#define CSL_OGC_XLINK

#include <optional>
#include <string>

#include "csl_ogc_export.hpp"

namespace ogc::schemas::xlink {

struct CSL_OGC_EXPORT SimpleAttrsGroup {
  // attributes
  std::optional<std::string> href;
  std::optional<std::string> role;
  std::optional<std::string> arcrole;
  std::optional<std::string> title;
  std::optional<std::string> show;
  std::optional<std::string> actuate;
  std::optional<std::string> about;
};

} // namespace ogc::schemas::xlink

#endif // CSL_OGC_XLINK
