// The Hapke BRDF to represent regolith, as defined in 10.1029/JB086iB04p03039 (DOI).

// w: The average single-scattering albedo of a particle of the regolith,
// i.e. the ratio of reflected energy to reflected plus absorbed energy at a particle, in range [0, 1].

// h: The density parameter which controls the sharpness of the opposition peak.
// A lower value describes a denser regolith, which leads to a sharper peak of the function.
// "Sharpness" in this context is about how strong the opposition effect is at particular phase angles.
// The range goes from 0 (excluded) to infinity (nonsense).

// B_0: The amplitude of the opposition surge, i.e. simply how strong the effect is, in range [0, inf (nonsense)].

// Some suggestions for some regoliths:
// - Lunar regolith: w = 0.32, h = 0.4, B_0 = 1.8

float $BRDF(vec3 N, vec3 L, vec3 V)
{
	float PI = 3.14159265358979323846;
	
    float g = acos(dot(L, V));
	float sin_g = sin(g);
	float cos_g = cos(g);
    float abs_g = abs(g);
    float tan_abs_g = tan(abs_g);
	float my_0 = dot(L, N);
	float my = dot(V, N);
	
	// The following term was suggested for B_0 in the original paper,
	// however, in later publications it is recommended to not use it anymore.
	//float B_0 = exp(-pow(w, 2) / 2);
	
	float B_g_ePart = exp(-$h / tan_abs_g);
	float B_g = $B_0;
	if (abs_g <= PI / 2) {
		B_g *= 1 - tan_abs_g / (2 * $h) * (3 - B_g_ePart) * (1 - B_g_ePart);
	}
	else {
		B_g *= 0;
	}
	
	float gamma = sqrt(1 - $w);
	float H_my = (1 + 2 * my) / (1 + 2 * gamma * my);
	float H_my_0 = (1 + 2 * my_0) / (1 + 2 * gamma * my_0);
	
	float f_LS = $w / (4 * PI * (my_0 + my));
	
	// This function was suggested in the original paper and fits the lunar surface,
	// as it is derived from Apollo samples. Other regoliths might have different functions.
	float P_g = 0.8 * PI * ((sin_g + (PI - g) * cos_g) / PI + pow(1 - cos_g, 2) / 10);
	
    return f_LS * ((1 + B_g) * P_g + H_my_0 * H_my - 1);
}