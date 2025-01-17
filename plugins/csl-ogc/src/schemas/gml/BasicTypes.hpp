////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_OGC_GML_BASIC_TYPES
#define CSL_OGC_GML_BASIC_TYPES

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "csl_ogc_export.hpp"

namespace ogc::schemas::gml {

struct CodeType;
struct MeasureType;

enum class NilReasonEnumeration { Inapplicable, Missing, Template, Unknown, Withheld, Other };

using NilReasonType = std::variant<NilReasonEnumeration, std::string>;

enum class SignType { Minus, Plus };

using BooleanOrNilReason = std::variant<NilReasonType, bool>;
using DoubleOrNilReason  = std::variant<NilReasonType, double>;
using IntegerOrNilReason = std::variant<NilReasonType, int64_t>;
using NameOrNilReason    = std::variant<NilReasonType, std::string>;
using StringOrNilReason  = std::variant<NilReasonType, std::string>;

struct CSL_OGC_EXPORT CodeType {
  // attributes
  std::optional<std::string> codeSpace;

  // content
  std::string content;

  virtual ~CodeType() = default;
};

struct CSL_OGC_EXPORT CodeWithAuthorityType final : CodeType {
  CodeWithAuthorityType(std::string value, std::string codeSpace) {
    CodeType::codeSpace = std::move(codeSpace);
    content     = std::move(value);
  }

  std::string codeSpace() {
    return CodeType::codeSpace.value();
  }
};

struct CSL_OGC_EXPORT MeasureType {
  // attributes
  std::string uom;

  // content
  double content;
};

using UomIdentifier = std::string;

struct CSL_OGC_EXPORT CoordinatesType {
  // attributes
  std::string decimal = ".";
  std::string cs      = ",";
  std::string ts      = " ";

  // content
  std::string content;
};

using BooleanList            = std::vector<bool>;
using DoubleList             = std::vector<double>;
using IntegerList            = std::vector<int64_t>;
using NameList               = std::vector<std::string>;
using NCNameList             = std::vector<std::string>;
using QNameList              = std::vector<std::string>;
using BooleanOrNilReasonList = std::vector<BooleanOrNilReason>;
using NameOrNilReasonList    = std::vector<NameOrNilReason>;
using DoubleOrNilReasonList  = std::vector<DoubleOrNilReason>;
using IntegerOrNilReasonList = std::vector<IntegerOrNilReason>;

struct CSL_OGC_EXPORT CodeListType {
  // attributes
  std::optional<std::string> codeSpace;

  // content
  NameList content;
};

struct CSL_OGC_EXPORT CodeOrNilReasonListType {
  // attributes
  std::optional<std::string> codeSpace;

  // content
  NameOrNilReasonList content;
};

struct CSL_OGC_EXPORT MeasureListType {
  // attributes
  UomIdentifier uom;

  // content
  DoubleList content;
};

struct CSL_OGC_EXPORT MeasureOrNilReasonListType {
  // attributes
  UomIdentifier uom;

  // content
  DoubleOrNilReasonList content;
};

} // namespace ogc::schemas::gml

#endif // CSL_OGC_GML_BASIC_TYPES