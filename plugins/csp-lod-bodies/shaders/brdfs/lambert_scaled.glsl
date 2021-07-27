// Lambertian reflectance to represent ideal diffuse surfaces, scaled by Pi.

// rho: Reflectivity of the surface in range [0, 1].

float $BRDF(vec3 N, vec3 L, vec3 V)
{
    return $rho;
}