// Oren-Nayar BRDF to represent ideal diffuse and rough surfaces,
// as defined in 10.1145/192161.192213 (DOI).

// rho: Reflectivity of the surface in range [0, 1].

// sigma: Standard deviation of the angle in degrees between the macroscopic surface normal and the
// microscopic normals of the surface facets. It is the roughness parameter so to say.
// It ranges from 0 to 90 where a value of zero degenerates the BRDF to Lambertian reflectance.
// You'll be happy with sigma = 30 in most cases.

// Some suggestions for some bodies:
// - Moon: rho = 0.11, sigma = 37

float $BRDF(vec3 N, vec3 L, vec3 V)
{
  float PI = 3.14159265358979323846;
  
  float cos_theta_i = dot(N, L);
  float theta_i = acos(cos_theta_i);
  
  float cos_theta_r = dot(N, V);
  float theta_r = acos(cos_theta_r);
  
  // Project L and V on a plane with N as the normal and get the cosine of the angle between the projections.
  float cos_diff_phi = dot(normalize(V - cos_theta_r * N), normalize(L - cos_theta_i * N));
  
  float alpha = max(theta_i, theta_r);
  float beta = min(theta_i, theta_r);
  float beta_1  = 2 * beta / PI;
  
  float sigma2 = pow($sigma * PI / 180, 2);
  float sigma_term = sigma2 / (sigma2 + 0.09);
  
  float C1 = 1 - 0.5 * (sigma2 / (sigma2 + 0.33));
  
  float C2 = 0.45 * sigma_term;
  if (cos_diff_phi >= 0) {
    C2 *= sin(alpha);
  }
  else {
    C2 *= sin(alpha) - pow(beta_1, 3);
  }
  
  float C3 = 0.125 * sigma_term * pow(4 * alpha * beta / (PI * PI), 2);
  
  float L1 = C1 + cos_diff_phi * C2 * tan(beta) + (1 - abs(cos_diff_phi)) * C3 * tan((alpha + beta) / 2);
  
  float L2 = 0.17 * $rho * sigma2 / (sigma2 + 0.13) * (1 - cos_diff_phi * pow(beta_1, 2));
  
  // I've extracted rho divided by Pi from L1 and L2 for clarity and performance,
  // in contrast to how the formula looks like in the paper.
  return $rho / PI * (L1 + L2);
}