////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
//      and may be used under the terms of the MIT license. See the LICENSE file for details.     //
//                        Copyright: (c) 2019 German Aerospace Center (DLR)                       //
////////////////////////////////////////////////////////////////////////////////////////////////////

#ifndef CSP_MEASUREMENT_TOOLS_VECTOR2F_HPP
#define CSP_MEASUREMENT_TOOLS_VECTOR2F_HPP

#include <iostream>

namespace csp::measurementtools {

/// A struct representing a 2D-Vector.
/// It provides more math functions then the SFML Vector
/// and is more optimized for MARS.

struct Vector2f {
  /// Default ctor (0, 0).
  Vector2f();
  /// Ctor from a \a x and a \a y value.
  Vector2f(double x, double y);

  Vector2f(Vector2f const& point);
  Vector2f(Vector2f&& other) = default;

  Vector2f& operator=(Vector2f const& other) = default;
  Vector2f& operator=(Vector2f&& other) = default;

  ~Vector2f() = default;

  /// Sets the length of the vector to 1.
  Vector2f normalize() const;

  /// Returns the length of the vector.
  /// Use Vector2f::lengthSquare for comparing the length
  /// of vectors, because it's much faster.
  double length() const;

  /// Returns the squared length of the vector.
  double lengthSquare() const;

  /// Overload for the operator += with another vector.
  Vector2f& operator+=(Vector2f const& rhs);

  /// Overload for the operator -= with another vector.
  Vector2f& operator-=(Vector2f const& rhs);

  /// Overload for the operator *= with a scalar.
  Vector2f& operator*=(double rhs);

  /// Overload for the operator /= with a scalar.
  Vector2f& operator/=(double rhs);

  /// \name Data
  /// Members storing the information of the vector.
  ///@{
  double mX{}, mY{};
  ///@}
};

/// Addition of two vectors.
Vector2f operator+(Vector2f const& lhs, Vector2f const& rhs);

/// Subtraction of two vectors.
Vector2f operator-(Vector2f const& lhs, Vector2f const& rhs);

/// Scalar multiplication of two vectors.
double operator*(Vector2f const& lhs, Vector2f const& rhs);

/// Multiplication of a vector with a scalar.
Vector2f operator*(Vector2f const& lhs, double rhs);

/// Multiplication of a scalar with a vector.
Vector2f operator*(double const& lhs, const Vector2f& rhs);

/// Division of a vector by a scalar.
Vector2f operator/(Vector2f const& lhs, double rhs);

/// Comparision of two vectors.
bool operator==(Vector2f const& lhs, Vector2f const& rhs);

/// Comparision of two vectors.
bool operator!=(Vector2f const& lhs, Vector2f const& rhs);

/// Comparision of two vectors.
bool operator<(Vector2f const& lhs, Vector2f const& rhs);

/// Comparision of two vectors.
bool operator>(Vector2f const& lhs, Vector2f const& rhs);

/// Returns true, if second vector is rotated clockwise in reference to the first
bool clockWise(Vector2f const& first, Vector2f const& second);

/// Stream operator for a vector.
/// Creates an output like [x, y].
std::ostream& operator<<(std::ostream& os, Vector2f const& rhs);
} // namespace csp::measurementtools
#endif // CSP_MEASUREMENT_TOOLS_VECTOR2F_HPP
