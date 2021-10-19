////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2020 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "WebCoverageException.hpp"
#include "logger.hpp"
#include "utils.hpp"

namespace csp::wcsoverlays {

////////////////////////////////////////////////////////////////////////////////////////////////////

std::istream& operator>>(std::istream& in, WebCoverageException::Code& code) {
  std::string codeStr;
  code = WebCoverageException::Code::eNone;
  if (in >> codeStr) {
    if (codeStr == "InvalidFormat") {
      code = WebCoverageException::Code::eInvalidFormat;
    } else if (codeStr == "InvalidCRS") {
      code = WebCoverageException::Code::eInvalidCRS;
    } else if (codeStr == "LayerNotDefined") {
      code = WebCoverageException::Code::eLayerNotDefined;
    } else if (codeStr == "StyleNotDefined") {
      code = WebCoverageException::Code::eStyleNotDefined;
    } else if (codeStr == "LayerNotQueryable") {
      code = WebCoverageException::Code::eLayerNotQueryable;
    } else if (codeStr == "InvalidPoint") {
      code = WebCoverageException::Code::eInvalidPoint;
    } else if (codeStr == "CurrentUpdateSequence") {
      code = WebCoverageException::Code::eCurrentUpdateSequence;
    } else if (codeStr == "InvalidUpdateSequence") {
      code = WebCoverageException::Code::eInvalidUpdateSequence;
    } else if (codeStr == "MissingDimensionValue") {
      code = WebCoverageException::Code::eMissingDimensionValue;
    } else if (codeStr == "InvalidDimensionValue") {
      code = WebCoverageException::Code::eInvalidDimensionValue;
    } else if (codeStr == "OperationNotSupported") {
      code = WebCoverageException::Code::eOperationNotSupported;
    } else if (codeStr == "NoApplicableCode") {
      code = WebCoverageException::Code::eNoApplicableCode;
    }
  }
  return in;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::ostream& operator<<(std::ostream& out, WebCoverageException::Code& code) {
  if (code == WebCoverageException::Code::eInvalidFormat) {
    out << "InvalidFormat";
  } else if (code == WebCoverageException::Code::eInvalidCRS) {
    out << "InvalidCRS";
  } else if (code == WebCoverageException::Code::eLayerNotDefined) {
    out << "LayerNotDefined";
  } else if (code == WebCoverageException::Code::eStyleNotDefined) {
    out << "StyleNotDefined";
  } else if (code == WebCoverageException::Code::eLayerNotQueryable) {
    out << "LayerNotQueryable";
  } else if (code == WebCoverageException::Code::eInvalidPoint) {
    out << "InvalidPoint";
  } else if (code == WebCoverageException::Code::eCurrentUpdateSequence) {
    out << "CurrentUpdateSequence";
  } else if (code == WebCoverageException::Code::eInvalidUpdateSequence) {
    out << "InvalidUpdateSequence";
  } else if (code == WebCoverageException::Code::eMissingDimensionValue) {
    out << "MissingDimensionValue";
  } else if (code == WebCoverageException::Code::eInvalidDimensionValue) {
    out << "InvalidDimensionValue";
  } else if (code == WebCoverageException::Code::eOperationNotSupported) {
    out << "OperationNotSupported";
  } else if (code == WebCoverageException::Code::eNoApplicableCode) {
    out << "NoApplicableCode";
  } else {
    out << "UnknownCode";
  }
  return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WebCoverageException::WebCoverageException(VistaXML::TiXmlElement* element) {
  mCode = utils::getAttribute<Code>(element, "exceptionCode").value_or(Code::eNone);
  mText = utils::getElementValue<std::string>(element->FirstChildElement("ows:ExceptionText"))
              .value_or("No description given");

  std::stringstream message;
  message << mCode << ": " << mText;
  mMessage = message.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WebCoverageException::Code WebCoverageException::getCode() const {
  return mCode;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& WebCoverageException::getText() const {
  return mText;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* WebCoverageException::what() const noexcept {
  return mMessage.c_str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WebCoverageExceptionReport::WebCoverageExceptionReport(VistaXML::TiXmlDocument doc) {
  VistaXML::TiXmlElement* root = doc.FirstChildElement("ows:ExceptionReport");
  if (root == nullptr) {
    throw std::runtime_error("XML document has no 'ows:ExceptionReport' element as root");
  }

  for (VistaXML::TiXmlElement* exceptionElement = root->FirstChildElement("ows:Exception");
       exceptionElement; exceptionElement = exceptionElement->NextSiblingElement("ows:Exception")) {
    mExceptions.emplace_back(exceptionElement);
  }

  if (mExceptions.empty()) {
    mMessage = "No WCS exceptions found";
  } else if (mExceptions.size() == 1) {
    mMessage = mExceptions[0].what();
  } else {
    std::stringstream message;
    message << "Multiple WCS exceptions occurred: ";
    for (WebCoverageException const& e : mExceptions) {
      message << "'" << e.what() << "'";
      if (e != mExceptions.back()) {
        message << ", ";
      }
    }
    mMessage = message.str();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WebCoverageExceptionReport::WebCoverageExceptionReport(std::string const& xml)
    : WebCoverageExceptionReport(parseXml(xml)) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<WebCoverageException> const& WebCoverageExceptionReport::getExceptions() const {
  return mExceptions;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* WebCoverageExceptionReport::what() const noexcept {
  return mMessage.c_str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

VistaXML::TiXmlDocument WebCoverageExceptionReport::parseXml(std::string const& xml) {
  VistaXML::TiXmlDocument doc;
  doc.Parse(xml.c_str());
  if (doc.Error()) {
    std::stringstream message;
    message << "Parsing XML failed: " << doc.ErrorDesc();
    throw std::runtime_error(message.str());
  }
  return doc;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace csp::wcsoverlays
