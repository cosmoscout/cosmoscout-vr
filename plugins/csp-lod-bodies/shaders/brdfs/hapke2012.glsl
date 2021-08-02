// The Hapke BRDF to represent regolith, as shown in 10.1002/2013JE004580 (DOI).

// w: The average single-scattering albedo of a particle of the regolith,
// i.e. the ratio of reflected energy to reflected plus absorbed energy at a particle, in range [0, 1].

// b: Shape-controlling parameter for the average single particle scattering function in range [0, 1].
// Controls the amplitude of the lobes of backward and forward scattering. A value of 0 is isotropic scattering.

// c: Ratio between strength of backward and forward scattering in range [-1, 1].
// A value of -1 increases forward scattering and 1 backward scattering.

// B_C0: The amplitude of the contribution of coherent backscattering to the opposition effect in range [0, inf (nonsense)].

// h_C: The angular width parameter for the contribution of coherent backscattering to the opposition effect.
// It controls at which phase angles the opposition effect kicks in, in range [0, TODO].

// B_S0: Analogous to B_C0 but for the shadow hiding contribution.

// h_S: Analogous to h_C but for the shadow hiding contribution.

// theta_p: Average angle between the macroscopic surface normal and the normal of a surface facet in range [0, 90].
// It is basically the surface roughness parameter.

// phi: Filling factor to control the porosity of the regolith in range [0, 1]. Hard to explain in simple terms.
// Basically the opposite of how densly the regolith is structured. A higher value corresponds to higher density.

// Some suggestions for some regoliths:
// - Lunar regolith: w = 0.32357, b = 0.23955, c = 0.30452, B_C0 = 0.0, h_C = 1.0, B_S0 = 1.80238, h_S = 0.07145, theta_p = 23.4, phi = 0.3

float $BRDF(vec3 N, vec3 L, vec3 V)
{
  float PI = 3.14159265358979323846;
  
  float cos_g = dot(L, V);
  float cos_i = dot(L, N);
  float cos_e = dot(V, N);
  
  float g = acos(cos_g);
  float i = acos(cos_i);
  float e = acos(cos_e);
  
  float my_0 = cos_i;
  float my = cos_e;
  
  float sin_i = sin(i);
  float sin_e = sin(e);
  
  // The way to calculate Psi in the paper is unstable for some reason,
  // therefore we calculate Psi by projecting L and V on a plane with N as normal and getting
  // the cosine of the angle between the projections.
  //float Psi = acos((cos_g - cos_e * cos_i) / (sin_e * sin_i));
  float cos_Psi = dot(normalize(L - cos_i * N), normalize(V - cos_e * N));
  cos_Psi = clamp(cos_Psi, -1, 1);
  float Psi = acos(cos_Psi);
  
  float PsiHalf = Psi / 2;
  float f_Psi = exp(-2 * tan(PsiHalf));
  float sin_PsiHalf = sin(PsiHalf);
  float sin2_PsiHalf = sin_PsiHalf * sin_PsiHalf;
  float PsiPerPI = Psi / PI;
  
  float tan_theta_p = tan($theta_p * PI / 180);
  float tan2_theta_p = tan_theta_p * tan_theta_p;
  
  // The tan_theta_p is zero when theta_p is zero.
  // A zero for theta_p is somewhat unrealistic.
  float cot_theta_p = 1 / tan_theta_p;
  float cot2_theta_p = cot_theta_p * cot_theta_p;
  
  float tan_i = tan(i);
  float tan_e = tan(e);
  
  // Here, tan_i and tan_e are zero each when i and e are zero each.
  // Because 1 / 0.0 is positive infinity we should check the usages.
  // The results are only used as factors for exponential functions,
  // where the whole expressions for the arguments are negated and
  // therefore result in negative infinity. Thus overall, these
  // exponential functions simply result in zero. No fix needed.
  float cot_i = 1 / tan_i;
  float cot_e = 1 / tan_e;
  
  float cot2_i = cot_i * cot_i;
  float cot2_e = cot_e * cot_e;
  
  float E_1_i = exp(-2 / PI * cot_theta_p * cot_i);
  float E_1_e = exp(-2 / PI * cot_theta_p * cot_e);
  float E_2_i = exp(-1 / PI * cot2_theta_p * cot2_i);
  float E_2_e = exp(-1 / PI * cot2_theta_p * cot2_e);
  
  float chi_theta_p = 1 / sqrt(1 + PI * tan2_theta_p);
  float eta_i = chi_theta_p * (cos_i + sin_i * tan_theta_p * E_2_i / (2 - E_1_i));
  float eta_e = chi_theta_p * (cos_e + sin_e * tan_theta_p * E_2_e / (2 - E_1_e));
  
  float my_0e = chi_theta_p;
  float my_e = chi_theta_p;
  float S;
  // Don't panic, I've checked it 3 times.
  if (i <= e) {
    my_0e *= cos_i + sin_i * tan_theta_p * (cos_Psi * E_2_e + sin2_PsiHalf * E_2_i) / (2 - E_1_e - PsiPerPI * E_1_i);
    my_e *= cos_e + sin_e * tan_theta_p * (E_2_e - sin2_PsiHalf * E_2_i) / (2 - E_1_e - PsiPerPI * E_1_i);
    S = my_e / eta_e * my_0 / eta_i * chi_theta_p / (1 - f_Psi + f_Psi * chi_theta_p * (my_0 / eta_i));
  }
  else {
    my_0e *= cos_i + sin_i * tan_theta_p * (E_2_i - sin2_PsiHalf * E_2_e) / (2 - E_1_i - PsiPerPI * E_1_e);
    my_e *= cos_e + sin_e * tan_theta_p * (cos_Psi * E_2_i + sin2_PsiHalf * E_2_e) / (2 - E_1_i - PsiPerPI * E_1_e);
    S = my_e / eta_e * my_0 / eta_i * chi_theta_p / (1 - f_Psi + f_Psi * chi_theta_p * (my / eta_e));
  }
  
  float KphiTerm = 1.209 * pow($phi, 2.0 / 3);
  // TODO: This goes into complex numbers already within [0, 1]. Figure out how to handle this.
  float K = -log(1 - KphiTerm) / KphiTerm;
  
  float gHalf = g / 2;
  float tan_gHalf = tan(gHalf);
  float tan_gHalfPerh_c = tan_gHalf / $h_C;
  float B_C = (1 + (1 - exp(-tan_gHalfPerh_c)) / tan_gHalfPerh_c) / (2 * pow(1 + tan_gHalfPerh_c, 2));
  // When g = 0, the division in the upper part causes a NaN.
  if (isnan(B_C)) {
    B_C = 1;
  }
  
  float r_0Term = sqrt(1 - $w);
  float r_0 = (1 - r_0Term) / (1 + r_0Term);
  
  float LS = my_0e / (my_0e + my_e);
  
  float b2 = $b * $b;
  // An approximation from the paper.
  //c = 3.29 * exp(-17.4 * b2) - 0.908;
  float oneMinusb2 = 1 - b2;
  float twobcos_g = 2 * $b * cos_g;
  float p_g = (1 + $c) / 2 * oneMinusb2 / pow(1 - twobcos_g + b2, 1.5) + (1 - $c) / 2 * oneMinusb2 / pow(1 + twobcos_g + b2, 1.5);
  
  float B_S = 1 / (1 + tan_gHalf / $h_S);
  
  float x_i = my_0e / K;
  float x_e = my_e / K;
  float H_i = 1 / (1 - $w * x_i * (r_0 + (1 - 2 * r_0 * x_i) / 2 * log((1 + x_i) / x_i)));
  float H_e = 1 / (1 - $w * x_e * (r_0 + (1 - 2 * r_0 * x_e) / 2 * log((1 + x_e) / x_e)));
  
  float M = H_i * H_e - 1;
  
  return LS * K * $w / (4 * PI) * (p_g * (1 + $B_S0 * B_S) + M) * (1 + $B_C0 * B_C) * S / cos_i;
}