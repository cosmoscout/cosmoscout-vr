////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "Vector2f.hpp"

#include <cmath>

namespace csp::measurementtools {

Vector2f::Vector2f() = default;

Vector2f::Vector2f(Vector2f const& point) = default;

Vector2f::Vector2f(double x_in, double y_in)
    : mX(x_in)
    , mY(y_in) {
}

Vector2f Vector2f::normalize() const {
  double len = length();
  if (len != 0) {
    return Vector2f(mX / len, mY / len);
  }

  return Vector2f(0, 0);
}

double Vector2f::length() const {
  return (std::sqrt(mX * mX + mY * mY));
}

double Vector2f::lengthSquare() const {
  return mX * mX + mY * mY;
}

Vector2f& Vector2f::operator+=(Vector2f const& rhs) {
  mX += rhs.mX;
  mY += rhs.mY;
  return *this;
}

Vector2f& Vector2f::operator-=(Vector2f const& rhs) {
  mX -= rhs.mX;
  mY -= rhs.mY;
  return *this;
}

Vector2f& Vector2f::operator*=(double rhs) {
  mX *= rhs;
  mY *= rhs;
  return *this;
}

Vector2f& Vector2f::operator/=(double rhs) {
  mX /= rhs;
  mY /= rhs;
  return *this;
}

Vector2f operator+(Vector2f const& lhs, Vector2f const& rhs) {
  return (Vector2f(lhs.mX + rhs.mX, lhs.mY + rhs.mY));
}

Vector2f operator-(Vector2f const& lhs, Vector2f const& rhs) {
  return (Vector2f(lhs.mX - rhs.mX, lhs.mY - rhs.mY));
}

double operator*(Vector2f const& lhs, Vector2f const& rhs) {
  return (lhs.mX * rhs.mX + lhs.mY * rhs.mY);
}

Vector2f operator*(Vector2f const& lhs, double rhs) {
  return (Vector2f(lhs.mX * rhs, lhs.mY * rhs));
}

Vector2f operator*(double const& lhs, const Vector2f& rhs) {
  return rhs * lhs;
}

Vector2f operator/(Vector2f const& lhs, double rhs) {
  return (Vector2f(lhs.mX / rhs, lhs.mY / rhs));
}

bool operator==(Vector2f const& lhs, Vector2f const& rhs) {
  return (lhs.mX == rhs.mX && lhs.mY == rhs.mY);
}

bool operator!=(Vector2f const& lhs, Vector2f const& rhs) {
  return (lhs.mX != rhs.mX && lhs.mY != rhs.mY);
}

bool operator<(Vector2f const& lhs, Vector2f const& rhs) {
  return lhs.lengthSquare() < rhs.lengthSquare();
}

bool operator>(Vector2f const& lhs, Vector2f const& rhs) {
  return lhs.lengthSquare() > rhs.lengthSquare();
}

bool clockWise(Vector2f const& first, Vector2f const& second) {
  return (first.mX * second.mY - first.mY * second.mX) < 0;
}

std::ostream& operator<<(std::ostream& os, Vector2f const& rhs) {
  return os << "[" << rhs.mX << ", " << rhs.mY << "]";
}
} // namespace csp::measurementtools
