////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef CSL_OGC_OGC_EXCEPTION_HPP
#define CSL_OGC_OGC_EXCEPTION_HPP

#include <exception>
#include <string>

namespace csl::ogc {

class OGCException : public std::exception {
 public:
  /// Get a short description of the error.
  std::string const& getText() const noexcept;

  /// Get a short description of the error.
  std::string const& getCode() const noexcept;

  /// Get type and description of the error as one string.
  const char* what() const noexcept override;

  inline bool operator!=(const OGCException& rhs) const {
    return mCode != rhs.mCode || mText != rhs.mText;
  }

 protected:
  OGCException(std::string code, std::string text);

 private:
  std::string mCode;
  std::string mText;
  std::string mMessage;
};

} // namespace csl::ogc

#endif // CSL_OGC_OGC_EXCEPTION_HPP
