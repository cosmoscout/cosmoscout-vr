////////////////////////////////////////////////////////////////////////////////////////////////////
//                               This file is part of CosmoScout VR                               //
////////////////////////////////////////////////////////////////////////////////////////////////////

// SPDX-FileCopyrightText: German Aerospace Center (DLR) <cosmoscout@dlr.de>
// SPDX-FileCopyrightText: 1998 P. J. Flatau
// SPDX-FileCopyrightText: 1990 B. T. Draine
// SPDX-FileCopyrightText: 1983 Craig F. Bohren & Donald R. Huffman
// SPDX-License-Identifier: MIT

#ifndef BH_MIE_HPP
#define BH_MIE_HPP

#include <complex>
#include <vector>

/// This is the Bohren-Huffman Mie scattering subroutine to calculate scattering and absorption by a
/// homogenous isotropic sphere. The original code can be found here:
/// http://scatterlib.wikidot.com/mie
///
/// Original program taken from Bohren and Huffman (1983), Appendix A
/// Modified by B.T.Draine, Princeton Univ. Obs., 90/10/26 in order to compute <cos(theta)>
/// This code was translatted to C by P. J. Flatau Feb 1998.
/// Translation to C++ was done by S. Schneegans in Dec 2023.
///
/// Input:
///  x:     2*pi*r/lambda
///  cxref: (complex refractive index of sphere)/(real index of medium)
///  nang:  number of angles between 0 and 90 degrees (it will calculate 2*nang-1 directions from 0
///         to 180 deg.)
///
/// Output:
///  cxs1:  scattering intensity perpendicular to scattering plane
///  cxs2:  scattering intensity parallel to scattering plane
///  qext:  C_ext/pi*r**2 = efficiency factor for extinction
///  qsca:  C_sca/pi*r**2 = efficiency factor for scattering
///  qback: 4*pi*(dC_sca/domega)/pi*r**2 = backscattering efficiency
///  gsca:  <cos(theta)> for scattering
void bhmie(double x, std::complex<double> cxref, unsigned long nang,
    std::vector<std::complex<double>>& cxs1, std::vector<std::complex<double>>& cxs2, double* qext,
    double* qsca, double* qback, double* gsca);

#endif // _BH_MIE_H_
