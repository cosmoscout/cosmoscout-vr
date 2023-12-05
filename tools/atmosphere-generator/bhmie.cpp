#include "bhmie.hpp"

#include <cmath>
#include <cstdio>

#define CXONE std::complex<float>(1.0, 0.0)

void bhmie(float x, std::complex<float> cxref, unsigned long nang,
    std::vector<std::complex<float>>& cxs1, std::vector<std::complex<float>>& cxs2, float* qext,
    float* qsca, float* qback, float* gsca) {
  /* .. Array Arguments .. */
  /*      COMPLEX :: cxs1(2*nang-1), cxs2(2*nang-1)*/
  /* .. Local Scalars ..*/
  std::complex<float> cxan, cxan1, cxbn, cxbn1, cxxi, cxxi0, cxy, cxxi1;
  std::complex<float> cxtemp;
  float        apsi, apsi0, apsi1, chi, chi0, chi1, dang, fn, p, pii, rn, t, theta, xstop, ymod;
  double       dn, dx, psi, psi0, psi1;
  unsigned int j, jj, n, nmx, nn, nstop;

  pii = 4.E0 * atan(1.E0);
  dx  = x;
  cxy = std::complex<float>(x, 0.0) * cxref;

  /* Series expansion terminated after NSTOP terms */
  xstop = x + 4.E0 * pow(x, 0.3333) + 2.0;
  nstop = xstop;
  ymod  = std::abs(cxy);
  nmx   = std::max(xstop, ymod) + 15;

  /* .. Local Arrays ..*/
  std::vector<std::complex<float>> cxd(nmx + 1);
  std::vector<float> amu(nang + 1), pi(nang + 1), pi0(nang + 1), pi1(nang + 1), tau(nang + 1);

  dang = .5E0 * pii / (float)(nang - 1);
  for (j = 1; j <= nang; j++) {

    theta  = (float)(j - 1) * dang;
    amu[j] = cos(theta);
  }

  /* Logarithmic derivative D(J) calculated by downward recurrence
      beginning with initial value (0.,0.) at J=NMX */

  cxd[nmx] = std::complex<float>(0.E0, 0.E0);
  nn       = nmx - 1;

  for (n = 1; n <= nn; n++) {
    rn = nmx - n + 1;
    /*        cxd(nmx-n) = (rn/cxy) - (1.E0/(cxd(nmx-n+1)+rn/cxy)) */
    cxtemp       = cxd[nmx - n + 1] + std::complex<float>(rn, 0.0) / cxy;
    cxtemp       = CXONE / cxtemp;
    cxd[nmx - n] = std::complex<float>(rn, 0.0) / cxy - cxtemp;
  }

  for (j = 1; j <= nang; j++) {
    pi0[j] = 0.E0;
    pi1[j] = 1.E0;
  }
  nn = 2 * nang - 1;
  for (j = 1; j <= nn; j++) {
    cxs1[j] = std::complex<float>(0.E0, 0.E0);
    cxs2[j] = std::complex<float>(0.E0, 0.E0);
  }

  /* Riccati-Bessel functions with real argument X
      calculated by upward recurrence */

  psi0  = cos(dx);
  psi1  = sin(dx);
  chi0  = -sin(x);
  chi1  = cos(x);
  apsi0 = psi0;
  apsi1 = psi1;
  cxxi0 = std::complex<float>(apsi0, -chi0);
  cxxi1 = std::complex<float>(apsi1, -chi1);
  *qsca = 0.E0;
  *gsca = 0.E0;

  for (n = 1; n <= nstop; n++) {

    dn   = n;
    rn   = n;
    fn   = (2.E0 * rn + 1.E0) / (rn * (rn + 1.E0));
    psi  = (2.E0 * dn - 1.E0) * psi1 / dx - psi0;
    apsi = psi;
    chi  = (2.E0 * rn - 1.E0) * chi1 / x - chi0;
    cxxi = std::complex<float>(apsi, -chi);
    /* Store previous values of AN and BN for use
        in computation of g=<cos(theta)> */
    if (n > 1) {
      cxan1 = cxan;
      cxbn1 = cxbn;
    }

    /* Compute AN and BN:*/
    /*        cxan = (cxd(n)/cxref+rn/x)*apsi - apsi1; */

    cxan = cxd[n] / cxref;
    cxan = cxan + std::complex<float>(rn / x, 0.0);
    cxan = cxan * std::complex<float>(apsi, 0.0);
    cxan = cxan - std::complex<float>(apsi1, 0.0);

    /*        cxan = cxan/((cxd(n)/cxref+rn/x)*cxxi-cxxi1); */
    cxtemp = cxd[n] / cxref;
    cxtemp = cxtemp + std::complex<float>(rn / x, 0.0);
    cxtemp = cxtemp * cxxi;
    cxtemp = cxtemp - cxxi1;
    cxan   = cxan / cxtemp;

    /*        cxbn = (cxref*cxd(n)+rn/x)*apsi - apsi1; */
    cxbn = cxref * cxd[n];
    cxbn = cxbn + std::complex<float>(rn / x, 0.0);
    cxbn = cxbn * std::complex<float>(apsi, 0.0);
    cxbn = cxbn - std::complex<float>(apsi1, 0.0);
    /*        cxbn = cxbn/((cxref*cxd(n)+rn/x)*cxxi-cxxi1); */
    cxtemp = cxref * cxd[n];
    cxtemp = cxtemp + std::complex<float>(rn / x, 0.0);
    cxtemp = cxtemp * cxxi;
    cxtemp = cxtemp - cxxi1;
    cxbn   = cxbn / cxtemp;

    /* Augment sums for *qsca and g=<cos(theta)> */
    /*        *qsca = *qsca + (2.*rn+1.)*(cabs(cxan)**2+cabs(cxbn)**2); */
    *qsca = *qsca +
            (2. * rn + 1.) * (std::abs(cxan) * std::abs(cxan) + std::abs(cxbn) * std::abs(cxbn));
    *gsca = *gsca + ((2. * rn + 1.) / (rn * (rn + 1.))) *
                        (cxan.real() * cxbn.real() + cxan.imag() * cxbn.imag());

    if (n > 1) {
      *gsca = *gsca + ((rn - 1.) * (rn + 1.) / rn) *
                          (cxan1.real() * cxan.real() + cxan1.imag() * cxan.imag() +
                              cxbn1.real() * cxbn.real() + cxbn1.imag() * cxbn.imag());
    }

    for (j = 1; j <= nang; j++) {
      jj     = 2 * nang - j;
      pi[j]  = pi1[j];
      tau[j] = rn * amu[j] * pi[j] - (rn + 1.E0) * pi0[j];
      p      = pow(-1.0, n - 1);
      /*          cxs1[j] = cxs1[j] + fn*(cxan*pi[j]+cxbn*tau[j]); */
      cxtemp  = cxan * std::complex<float>(pi[j], 0.0);
      cxtemp  = cxtemp + cxbn * std::complex<float>(tau[j], 0.0);
      cxtemp  = std::complex<float>(fn, 0.0) * cxtemp;
      cxs1[j] = cxs1[j] + cxtemp;
      t       = pow(-1.0, n);
      /*          cxs2[j] = cxs2[j] + fn*(cxan*tau[j]+cxbn*pi[j]); */
      cxtemp  = cxan * std::complex<float>(tau[j], 0.0);
      cxtemp  = cxtemp + cxbn * std::complex<float>(pi[j], 0.0);
      cxtemp  = std::complex<float>(fn, 0.0) * cxtemp;
      cxs2[j] = cxs2[j] + cxtemp;

      if (j != jj) {
        /*            cxs1[jj] = cxs1[jj] + fn*(cxan*pi(j)*p+cxbn*tau(j)*t);*/
        cxtemp   = cxan * std::complex<float>(pi[j] * p, 0.0);
        cxtemp   = cxtemp + cxbn * std::complex<float>(tau[j] * t, 0.0);
        cxtemp   = std::complex<float>(fn, 0.0) * cxtemp;
        cxs1[jj] = cxs1[jj] + cxtemp;

        /*            cxs2[jj] = cxs2[jj] + fn*(cxan*tau(j)*t+cxbn*pi(j)*p); */
        cxtemp   = cxan * std::complex<float>(tau[j] * t, 0.0);
        cxtemp   = cxtemp + cxbn * std::complex<float>(pi[j] * p, 0.0);
        cxtemp   = std::complex<float>(fn, 0.0) * cxtemp;
        cxs2[jj] = cxs2[jj] + cxtemp;
      }
    }

    psi0  = psi1;
    psi1  = psi;
    apsi1 = psi1;
    chi0  = chi1;
    chi1  = chi;
    cxxi1 = std::complex<float>(apsi1, -chi1);

    /*  For each angle J, compute pi_n+1
        from PI = pi_n , PI0 = pi_n-1 */

    for (j = 1; j <= nang; j++) {
      pi1[j] = ((2. * rn + 1.) * amu[j] * pi[j] - (rn + 1.) * pi0[j]) / rn;
      pi0[j] = pi[j];
    }
  } /*end of big for */

  /*  Have summed sufficient terms.
       Now compute *qsca,*qext,*qback,and *gsca */
  *gsca = 2. * *gsca / *qsca;
  *qsca = (2.E0 / (x * x)) * *qsca;
  *qext = (4.E0 / (x * x)) * cxs1[1].real();

  *qback = (4.E0 / (x * x)) * std::abs(cxs1[2 * nang - 1]) * std::abs(cxs1[2 * nang - 1]);
}
