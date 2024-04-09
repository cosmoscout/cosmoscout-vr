////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-License-Identifier: MIT

#ifndef HULST_MODE_HPP
#define HULST_MODE_HPP

#include <string>
#include <vector>

////////////////////////////////////////////////////////////////////////////////////////////////////
// This method writes the scattering coefficients and absorption coefficients based on            //
// Van de Hulst's anomalous diffraction approximation and the scattering approximation based on   //
// an atmosphere's turbidity.                                                                     //
////////////////////////////////////////////////////////////////////////////////////////////////////

int hulstMode(std::vector<std::string> const& arguments);

#endif // HULST_MODE_HPP