////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2020 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "WebMapException.hpp"
#include "logger.hpp"
#include "utils.hpp"

namespace csp::wmsoverlays {

////////////////////////////////////////////////////////////////////////////////////////////////////

std::istream& operator>>(std::istream& in, WebMapException::Code& code) {
  std::string codeStr;
  code = WebMapException::Code::eNone;
  if (in >> codeStr) {
    if (codeStr == "InvalidFormat") {
      code = WebMapException::Code::eInvalidFormat;
    } else if (codeStr == "InvalidCRS") {
      code = WebMapException::Code::eInvalidCRS;
    } else if (codeStr == "LayerNotDefined") {
      code = WebMapException::Code::eLayerNotDefined;
    } else if (codeStr == "StyleNotDefined") {
      code = WebMapException::Code::eStyleNotDefined;
    } else if (codeStr == "LayerNotQueryable") {
      code = WebMapException::Code::eLayerNotQueryable;
    } else if (codeStr == "InvalidPoint") {
      code = WebMapException::Code::eInvalidPoint;
    } else if (codeStr == "CurrentUpdateSequence") {
      code = WebMapException::Code::eCurrentUpdateSequence;
    } else if (codeStr == "InvalidUpdateSequence") {
      code = WebMapException::Code::eInvalidUpdateSequence;
    } else if (codeStr == "MissingDimensionValue") {
      code = WebMapException::Code::eMissingDimensionValue;
    } else if (codeStr == "InvalidDimensionValue") {
      code = WebMapException::Code::eInvalidDimensionValue;
    } else if (codeStr == "OperationNotSupported") {
      code = WebMapException::Code::eOperationNotSupported;
    }
  }
  return in;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::ostream& operator<<(std::ostream& out, WebMapException::Code& code) {
  if (code == WebMapException::Code::eInvalidFormat) {
    out << "InvalidFormat";
  } else if (code == WebMapException::Code::eInvalidCRS) {
    out << "InvalidCRS";
  } else if (code == WebMapException::Code::eLayerNotDefined) {
    out << "LayerNotDefined";
  } else if (code == WebMapException::Code::eStyleNotDefined) {
    out << "StyleNotDefined";
  } else if (code == WebMapException::Code::eLayerNotQueryable) {
    out << "LayerNotQueryable";
  } else if (code == WebMapException::Code::eInvalidPoint) {
    out << "InvalidPoint";
  } else if (code == WebMapException::Code::eCurrentUpdateSequence) {
    out << "CurrentUpdateSequence";
  } else if (code == WebMapException::Code::eInvalidUpdateSequence) {
    out << "InvalidUpdateSequence";
  } else if (code == WebMapException::Code::eMissingDimensionValue) {
    out << "MissingDimensionValue";
  } else if (code == WebMapException::Code::eInvalidDimensionValue) {
    out << "InvalidDimensionValue";
  } else if (code == WebMapException::Code::eOperationNotSupported) {
    out << "OperationNotSupported";
  } else {
    out << "UnknownCode";
  }
  return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WebMapException::WebMapException(VistaXML::TiXmlElement* element) {
  mCode = utils::getAttribute<Code>(element, "code").value_or(Code::eNone);
  mText = utils::getElementValue<std::string>(element).value_or("No description given");

  std::stringstream message;
  message << mCode << ": " << mText;
  mMessage = message.str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WebMapException::Code WebMapException::getCode() const {
  return mCode;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::string const& WebMapException::getText() const {
  return mText;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* WebMapException::what() const noexcept {
  return mMessage.c_str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WebMapExceptionReport::WebMapExceptionReport(VistaXML::TiXmlDocument doc) {
  VistaXML::TiXmlElement* root = doc.FirstChildElement("ServiceExceptionReport");
  if (root == nullptr) {
    throw std::runtime_error("XML document has no ServiceExceptionReport element as root");
  }

  for (VistaXML::TiXmlElement* exceptionElement = root->FirstChildElement("ServiceException");
       exceptionElement;
       exceptionElement = exceptionElement->NextSiblingElement("ServiceException")) {
    mExceptions.emplace_back(exceptionElement);
  }

  if (mExceptions.empty()) {
    mMessage = "No WMS exceptions found";
  } else if (mExceptions.size() == 1) {
    mMessage = mExceptions[0].what();
  } else {
    std::stringstream message;
    message << "Multiple WMS exceptions occurred: ";
    for (WebMapException const& e : mExceptions) {
      message << "'" << e.what() << "'";
      if (e != mExceptions.back()) {
        message << ", ";
      }
    }
    mMessage = message.str();
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

WebMapExceptionReport::WebMapExceptionReport(std::string const& xml)
    : WebMapExceptionReport(parseXml(xml)) {
}

////////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<WebMapException> const& WebMapExceptionReport::getExceptions() const {
  return mExceptions;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

const char* WebMapExceptionReport::what() const noexcept {
  return mMessage.c_str();
}

////////////////////////////////////////////////////////////////////////////////////////////////////

VistaXML::TiXmlDocument WebMapExceptionReport::parseXml(std::string const& xml) {
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

} // namespace csp::wmsoverlays
