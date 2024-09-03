#version 330 core

// Compute the irradiance within the glossy
// BRDF lobe aligned with a hard-coded wi
// that will equal our surface normal direction.
// Our surface normal direction is normalize(fs_Pos).
// Check implementation here: https://learnopengl.com/PBR/IBL/Specular-IBL

in vec3 fs_Pos;
out vec4 out_Col;
uniform samplerCube u_EnvironmentMap;
uniform float u_Roughness;

const float PI = 3.14159265359;

// Random number generator
float RadicalInverse_VdC(uint bits) 
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000
}
// ----------------------------------------------------------------------------
vec2 Hammersley(uint i, uint N)
{
    return vec2(float(i)/float(N), RadicalInverse_VdC(i));
} 

vec3 ImportanceSampleGGX(vec2 Xi, vec3 N, float roughness)
{
    float a = roughness*roughness;
	
    // Find an angle from 0 to 2pi about the hemisphere pole
    float phi = 2.0 * PI * Xi.x;
    // Combine the normal distribution function of GGX with a method proposed by Epic Games
    float cosTheta = sqrt((1.0 - Xi.y) / (1.0 + (a*a - 1.0) * Xi.y));
    // Trigonometric identity
    float sinTheta = sqrt(1.0 - cosTheta*cosTheta);
	
    // from spherical coordinates to cartesian coordinates
    vec3 wh;
    wh.x = cos(phi) * sinTheta;
    wh.y = sin(phi) * sinTheta;
    wh.z = cosTheta;
	
    // from tangent-space vector to world-space sample vector
    vec3 up        = abs(N.z) < 0.999 ? vec3(0.0, 0.0, 1.0) : vec3(1.0, 0.0, 0.0);
    vec3 tangent   = normalize(cross(up, N));
    vec3 bitangent = cross(N, tangent);
	
    vec3 whW = tangent * wh.x + bitangent * wh.y + N * wh.z;
    return normalize(whW);
}  

float DistributionGGX(vec3 N, vec3 wh, float roughness) {
    // Trowbridge-Reitz/GGX approximation
    float alpha = roughness * roughness;
    float alpha_squared = alpha * alpha;
    float dotProduct = max(dot(N, wh), 0.0);
    float dot_squared = dotProduct * dotProduct;

    return alpha_squared / (PI * pow(dot_squared * (alpha_squared - 1.f) + 1.f, 2));
}

void main() {
    const uint SAMPLE_COUNT = 1024u;
    float totalWeight = 0.0;   
    vec3 Li = vec3(0.0);    
    vec3 N = normalize(fs_Pos);
    // isotropic lobe assumption
    // Write in a separate line for clarity
    vec3 wo = N;

    for(uint i = 0u; i < SAMPLE_COUNT; ++i)
    {
        vec2 Xi = Hammersley(i, SAMPLE_COUNT);
        vec3 wh  = ImportanceSampleGGX(Xi, N, u_Roughness);
        // Reflect wo about wh
        vec3 wi  = normalize(2.0 * dot(wo, wh) * wh - wo);

        // Already scaled by cosine
        float NdotWi = max(dot(N, wi), 0.0);
        if(NdotWi > 0.0)
        {   
            // This D has nothing to do with the BRDF
            // It's just a technique to reduce the artifacts for the convolution
            // We can reduce this artifact by sampling a mip level of the environment map based on the integral's PDF and the roughness
            float D = DistributionGGX(N, wh, u_Roughness);
            float nDotwh  = max(dot(N, wh), 0.0);
            float woDotwh = max(dot(wh, wo), 0.0);
            float pdf = D * nDotwh / (4.0 * woDotwh) + 0.0001;
            
            float resolution = 1024.0; // resolution of env map cube face
            float saTexel  = 4.0 * PI / (6.0 * resolution * resolution);
            float saSample = 1.0 / (float(SAMPLE_COUNT) * pdf + 0.0001);
            float mipLevel = u_Roughness == 0.0 ? 0.0 : 0.5 * log2(saSample / saTexel);

            Li += textureLod(u_EnvironmentMap, wi, mipLevel).rgb * NdotWi;
            // Samples with less influence on the final result (for small NdotWi) contribute less to the final weight.
            totalWeight += NdotWi;
        }
    }

    Li = Li / totalWeight;

    out_Col = vec4(Li, 1.0);
}
