#version 330 core
#extension GL_ARB_shading_language_packing : enable

uniform vec3 u_CamPos;

// PBR material attributes
uniform vec3 u_Albedo;
uniform float u_Metallic;
uniform float u_Roughness;
uniform float u_AmbientOcclusion;
// Texture maps for controlling some of the attribs above, plus normal mapping
uniform sampler2D u_AlbedoMap;
uniform sampler2D u_MetallicMap;
uniform sampler2D u_RoughnessMap;
uniform sampler2D u_AOMap;
uniform sampler2D u_NormalMap;
// If true, use the textures listed above instead of the GUI slider values
uniform bool u_UseAlbedoMap;
uniform bool u_UseMetallicMap;
uniform bool u_UseRoughnessMap;
uniform bool u_UseAOMap;
uniform bool u_UseNormalMap;

// Image-based lighting
uniform samplerCube u_DiffuseIrradianceMap;
uniform samplerCube u_GlossyIrradianceMap;
uniform sampler2D u_BRDFLookupTexture;
// For rendering glints
uniform sampler2D u_GlintNoiseTexture;
uniform int _Glint2023NoiseMapSize;

// Varyings
in vec3 fs_Pos;
in vec3 fs_Nor; // Surface normal
in vec3 fs_Tan; // Surface tangent
in vec3 fs_Bit; // Surface bitangent
in vec2 fs_UV;
out vec4 out_Col;

const float PI = 3.14159f;
const float DEG2RAD = 0.01745329251;
const float RAD2DEG = 57.2957795131;

const bool useGlint = true;
const float _ScreenSpaceScale = 2.5;
const float _LogMicrofacetDensity = 10.0;
const float _MicrofacetRoughness = 0.5;
const float _DensityRandomization = 10.0;

//=======================================================================================
// TOOLS
//=======================================================================================
// Set the input material attributes to texture-sampled values
// if the indicated booleans are TRUE
void handleMaterialMaps(inout vec3 albedo, inout float metallic,
                        inout float roughness, inout float ambientOcclusion,
                        inout vec3 normal) {
    if(u_UseAlbedoMap) {
        albedo = pow(texture(u_AlbedoMap, fs_UV).rgb, vec3(2.2));
    }
    if(u_UseMetallicMap) {
        metallic = texture(u_MetallicMap, fs_UV).r;
    }
    if(u_UseRoughnessMap) {
        roughness = texture(u_RoughnessMap, fs_UV).r;
    }
    if(u_UseAOMap) {
        ambientOcclusion = texture(u_AOMap, fs_UV).r;
    }
    if(u_UseNormalMap) {
        // TODO: Apply normal mapping
        normal = 2.f * texture(u_NormalMap, fs_UV).rgb - 1.f;
        mat3 TBN = mat3(fs_Tan, fs_Bit, fs_Nor);
        normal = normalize(TBN * normal);
    }
}

vec3 fresnelSchlickRoughness(float cosTheta, vec3 R, float roughness)
{
    return R + (max(vec3(1.0 - roughness), R) - R) * pow(clamp(1.0 - cosTheta, 0.0, 1.0), 5.0);
} 

float erfinv(float x)
{
	float w, p;
	w = -log((1.0 - x) * (1.0 + x));
	if (w < 5.000000)
	{
		w = w - 2.500000;
		p = 2.81022636e-08;
		p = 3.43273939e-07 + p * w;
		p = -3.5233877e-06 + p * w;
		p = -4.39150654e-06 + p * w;
		p = 0.00021858087 + p * w;
		p = -0.00125372503 + p * w;
		p = -0.00417768164 + p * w;
		p = 0.246640727 + p * w;
		p = 1.50140941 + p * w;
	}
	else
	{
		w = sqrt(w) - 3.000000;
		p = -0.000200214257;
		p = 0.000100950558 + p * w;
		p = 0.00134934322 + p * w;
		p = -0.00367342844 + p * w;
		p = 0.00573950773 + p * w;
		p = -0.0076224613 + p * w;
		p = 0.00943887047 + p * w;
		p = 1.00167406 + p * w;
		p = 2.83297682 + p * w;
	}
	return p * x;
}

vec3 sampleNormalDistribution(vec3 u, float mu, float sigma)
{
	float x0 = sigma * 1.414213f * erfinv(2.0 * u.x - 1.0) + mu;
	float x1 = sigma * 1.414213f * erfinv(2.0 * u.y - 1.0) + mu;
	float x2 = sigma * 1.414213f * erfinv(2.0 * u.z - 1.0) + mu;
	return vec3(x0, x1, x2);
}

vec3 pcg3dFloat(uvec3 v)
{
	v = v * 1664525u + 1013904223u;

	v.x += v.y * v.z;
	v.y += v.z * v.x;
	v.z += v.x * v.y;

	v ^= v >> 16u;

	v.x += v.y * v.z;
	v.y += v.z * v.x;
	v.z += v.x * v.y;

	return v * (1.0 / 4294967296.0);
}

float HashWithoutSine13(vec3 p3)
{
	p3 = fract(p3 * .1031);
	p3 += dot(p3, p3.yzx + 33.33);
	return fract((p3.x + p3.y) * p3.z);
}

mat2 Inverse(mat2 A)
{
    return mat2(A[1][1], -A[0][1], -A[1][0], A[0][0]) / determinant(A);
}

void GetGradientEllipse(vec2 duvdx, vec2 duvdy, out vec2 ellipseMajor, out vec2 ellipseMinor)
{
    mat2 J = mat2(duvdx, duvdy);
    // Maybe this can be replaced by the glsl function 'inverse'?
    J = Inverse(J);
    J = J * transpose(J);

    float a = J[0][0];
    float b = J[0][1];
    float c = J[1][0];
    float d = J[1][1];

    float T = a + d;
    float D = a * d - b * c;
    float L1 = T / 2.0 - sqrt(T * T / 3.99999 - D);
    float L2 = T / 2.0 + sqrt(T * T / 3.99999 - D);

    vec2 A0 = vec2(L1 - d, c);
    vec2 A1 = vec2(L2 - d, c);
    float r0 = 1.0 / sqrt(L1);
    float r1 = 1.0 / sqrt(L2);
    ellipseMajor = normalize(A0) * r0;
    ellipseMinor = normalize(A1) * r1;
}

vec2 RotateUV(vec2 uv, float rotation, vec2 mid)
{
	return vec2(
		cos(rotation) * (uv.x - mid.x) + sin(rotation) * (uv.y - mid.y) + mid.x,
		cos(rotation) * (uv.y - mid.y) - sin(rotation) * (uv.x - mid.x) + mid.y
		);
}

float BilinearLerp(vec4 values, vec2 valuesLerp)
{
	// Values XY = float4(00, 01, 10, 11)
	float resultX = mix(values.x, values.z, valuesLerp.x);
	float resultY = mix(values.y, values.w, valuesLerp.x);
	float result = mix(resultX, resultY, valuesLerp.y);
	return result;
}

float Remap(float s, float a1, float a2, float b1, float b2)
{
	return b1 + (s - a1) * (b2 - b1) / (a2 - a1);
}

float Remap01To(float s, float b1, float b2)
{
	return b1 + s * (b2 - b1);
}

float RemapTo01(float s, float a1, float a2)
{
	return (s - a1) / (a2 - a1);
}

vec4 RemapTo01(vec4 s, float a1, float a2)
{
	return (s - a1) / (a2 - a1);
}

vec4 GetBarycentricWeightsTetrahedron(vec3 p, vec3 v1, vec3 v2, vec3 v3, vec3 v4)
{
    vec3 c11 = v1 - v4, c21 = v2 - v4, c31 = v3 - v4, c41 = v4 - p;

    vec2 m1 = c31.yz / c31.x;
    vec2 c12 = c11.yz - c11.x * m1, c22 = c21.yz - c21.x * m1, c32 = c41.yz - c41.x * m1;

    vec4 uvwk = vec4(0.0);
    float m2 = c22.y / c22.x;
    uvwk.x = (c32.x * m2 - c32.y) / (c12.y - c12.x * m2);
    uvwk.y = -(c32.x + c12.x * uvwk.x) / c22.x;
    uvwk.z = -(c41.x + c21.x * uvwk.y + c11.x * uvwk.x) / c31.x;
    uvwk.w = 1.0 - uvwk.z - uvwk.y - uvwk.x;

    return uvwk;
}

void UnpackFloatParallel4(vec4 input, out vec4 a, out vec4 b)
{
    uvec4 uintInput = floatBitsToUint(input);

    // Unpack each component separately and combine them into vec4 results
    a = vec4(
        unpackHalf2x16(uintInput.x >> 16),  // Unpack the upper 16 bits of x
        unpackHalf2x16(uintInput.y >> 16)   // Unpack the upper 16 bits of y
    );
    
    b = vec4(
        unpackHalf2x16(uintInput.x & 0xFFFFu),  // Unpack the lower 16 bits of x
        unpackHalf2x16(uintInput.y & 0xFFFFu)   // Unpack the lower 16 bits of y
    );
}


//=======================================================================================
// GLINTS TEST NOVEMBER 2022
//=======================================================================================
void CustomRand4Texture(vec2 slope, vec2 slopeRandOffset, out vec4 outUniform, out vec4 outGaussian, out vec2 slopeLerp)
{
    ivec2 size = ivec2(_Glint2023NoiseMapSize);
    vec2 slope2 = abs(slope) / _MicrofacetRoughness;
    slope2 = slope2 + (slopeRandOffset * vec2(size));
    slopeLerp = fract(slope2);
    ivec2 slopeCoord = ivec2(floor(slope2)) % size;

    vec4 packedRead = texture(u_GlintNoiseTexture, slopeCoord);
    UnpackFloatParallel4(packedRead, outUniform, outGaussian);
}

float GenerateAngularBinomialValueForSurfaceCell(vec4 randB, vec4 randG, vec2 slopeLerp, float footprintOneHitProba, float binomialSmoothWidth, float footprintMean, float footprintSTD, float microfacetCount)
{
    vec4 gating;
    if (binomialSmoothWidth > 0.0000001)
        gating = clamp(RemapTo01(randB, footprintOneHitProba + binomialSmoothWidth, footprintOneHitProba - binomialSmoothWidth), 0.0, 1.0);
    else
        gating = step(footprintOneHitProba, randB);  // Equivalent to randB < footprintOneHitProba

    vec4 gauss = randG * footprintSTD + footprintMean;
    gauss = clamp(floor(gauss), 0.0, microfacetCount);
    vec4 results = gating * (1.0 + gauss);
    float result = BilinearLerp(results, slopeLerp);
    return result;
}

float SampleGlintGridSimplex(vec2 uv, uint gridSeed, vec2 slope, float footprintArea, float targetNDF, float gridWeight)
{
    // Get surface space glint simplex grid cell
    const mat2 gridToSkewedGrid = mat2(1.0, -0.57735027, 0.0, 1.15470054);
    vec2 skewedCoord = gridToSkewedGrid * uv;
    ivec2 baseId = ivec2(floor(skewedCoord));
    vec3 temp = vec3(fract(skewedCoord), 0.0);
    temp.z = 1.0 - temp.x - temp.y;
    float s = step(0.0, -temp.z);
    float s2 = 2.0 * s - 1.0;
    ivec2 glint0 = baseId + ivec2(s, s);
    ivec2 glint1 = baseId + ivec2(s, 1.0 - s);
    ivec2 glint2 = baseId + ivec2(1.0 - s, s);
    vec3 barycentrics = vec3(-temp.z * s2, s - temp.y * s2, s - temp.x * s2);

    // Generate per surface cell random numbers
    vec3 rand0 = pcg3dFloat(uvec3(glint0 + ivec2(2147483648), gridSeed)); // TODO : optimize away manual seeds
    vec3 rand1 = pcg3dFloat(uvec3(glint1 + ivec2(2147483648), gridSeed));
    vec3 rand2 = pcg3dFloat(uvec3(glint2 + ivec2(2147483648), gridSeed));

    // Get per surface cell per slope cell random numbers
    vec4 rand0SlopesB, rand1SlopesB, rand2SlopesB, rand0SlopesG, rand1SlopesG, rand2SlopesG;
    vec2 slopeLerp0, slopeLerp1, slopeLerp2;
    CustomRand4Texture(slope, rand0.yz, rand0SlopesB, rand0SlopesG, slopeLerp0);
    CustomRand4Texture(slope, rand1.yz, rand1SlopesB, rand1SlopesG, slopeLerp1);
    CustomRand4Texture(slope, rand2.yz, rand2SlopesB, rand2SlopesG, slopeLerp2);

    // Compute microfacet count with randomization
    vec3 logDensityRand = clamp(sampleNormalDistribution(vec3(rand0.x, rand1.x, rand2.x), _LogMicrofacetDensity, _DensityRandomization), 0.0, 50.0); // TODO : optimize sampleNormalDist
    vec3 microfacetCount = max(vec3(0.0), footprintArea * exp(logDensityRand));
    vec3 microfacetCountBlended = microfacetCount * gridWeight;

    // Compute binomial properties
    float hitProba = _MicrofacetRoughness * targetNDF; // probability of hitting desired half vector in NDF distribution
    vec3 footprintOneHitProba = (1.0 - pow(vec3(1.0 - hitProba), microfacetCountBlended)); // probability of hitting at least one microfacet in footprint
    vec3 footprintMean = (microfacetCountBlended - 1.0) * hitProba; // Expected value of number of hits in the footprint given already one hit
    vec3 footprintSTD = sqrt((microfacetCountBlended - 1.0) * hitProba * (1.0 - hitProba)); // Standard deviation of number of hits in the footprint given already one hit
    vec3 binomialSmoothWidth = 0.1 * clamp(footprintOneHitProba * 10.0, 0.0, 1.0) * clamp((1.0 - footprintOneHitProba) * 10.0, 0.0, 1.0);

    // Generate numbers of reflecting microfacets
    float result0, result1, result2;
    result0 = GenerateAngularBinomialValueForSurfaceCell(rand0SlopesB, rand0SlopesG, slopeLerp0, footprintOneHitProba.x, binomialSmoothWidth.x, footprintMean.x, footprintSTD.x, microfacetCountBlended.x);
    result1 = GenerateAngularBinomialValueForSurfaceCell(rand1SlopesB, rand1SlopesG, slopeLerp1, footprintOneHitProba.y, binomialSmoothWidth.y, footprintMean.y, footprintSTD.y, microfacetCountBlended.y);
    result2 = GenerateAngularBinomialValueForSurfaceCell(rand2SlopesB, rand2SlopesG, slopeLerp2, footprintOneHitProba.z, binomialSmoothWidth.z, footprintMean.z, footprintSTD.z, microfacetCountBlended.z);

    // Interpolate result for glint grid cell
    vec3 results = vec3(result0, result1, result2) / microfacetCount;
    float result = dot(results, barycentrics);
    return result;
}

void GetAnisoCorrectingGridTetrahedron(bool centerSpecialCase, inout float thetaBinLerp, float ratioLerp, float lodLerp, out vec3 p0, out vec3 p1, out vec3 p2, out vec3 p3)
{
    if (centerSpecialCase == true) // SPECIAL CASE (no anisotropy, center of blending pattern, different triangulation)
    {
        vec3 a = vec3(0, 1, 0);
        vec3 b = vec3(0, 0, 0);
        vec3 c = vec3(1, 1, 0);
        vec3 d = vec3(0, 1, 1);
        vec3 e = vec3(0, 0, 1);
        vec3 f = vec3(1, 1, 1);
        if (lodLerp > 1.0 - ratioLerp) // Upper pyramid
        {
            if (RemapTo01(lodLerp, 1.0 - ratioLerp, 1.0) > thetaBinLerp) // Left-up tetrahedron (a, e, d, f)
            {
                p0 = a; p1 = e; p2 = d; p3 = f;
            }
            else // Right-down tetrahedron (f, e, c, a)
            {
                p0 = f; p1 = e; p2 = c; p3 = a;
            }
        }
        else // Lower tetrahedron (b, a, c, e)
        {
            p0 = b; p1 = a; p2 = c; p3 = e;
        }
    }
    else // NORMAL CASE
    {
        vec3 a = vec3(0, 1, 0);
        vec3 b = vec3(0, 0, 0);
        vec3 c = vec3(0.5, 1, 0);
        vec3 d = vec3(1, 0, 0);
        vec3 e = vec3(1, 1, 0);
        vec3 f = vec3(0, 1, 1);
        vec3 g = vec3(0, 0, 1);
        vec3 h = vec3(0.5, 1, 1);
        vec3 i = vec3(1, 0, 1);
        vec3 j = vec3(1, 1, 1);
        if (thetaBinLerp < 0.5 && thetaBinLerp * 2.0 < ratioLerp) // Prism A
        {
            if (lodLerp > 1.0 - ratioLerp) // Upper pyramid
            {
                if (RemapTo01(lodLerp, 1.0 - ratioLerp, 1.0) > RemapTo01(thetaBinLerp * 2.0, 0.0, ratioLerp)) // Left-up tetrahedron (a, f, h, g)
                {
                    p0 = a; p1 = f; p2 = h; p3 = g;
                }
                else // Right-down tetrahedron (c, a, h, g)
                {
                    p0 = c; p1 = a; p2 = h; p3 = g;
                }
            }
            else // Lower tetrahedron (b, a, c, g)
            {
                p0 = b; p1 = a; p2 = c; p3 = g;
            }
        }
        else if (1.0 - ((thetaBinLerp - 0.5) * 2.0) > ratioLerp) // Prism B
        {
            if (lodLerp < 1.0 - ratioLerp) // Lower pyramid
            {
                if (RemapTo01(lodLerp, 0.0, 1.0 - ratioLerp) > RemapTo01(thetaBinLerp, 0.5 - (1.0 - ratioLerp) * 0.5, 0.5 + (1.0 - ratioLerp) * 0.5)) // Left-up tetrahedron (b, g, i, c)
                {
                    p0 = b; p1 = g; p2 = i; p3 = c;
                }
                else // Right-down tetrahedron (d, b, c, i)
                {
                    p0 = d; p1 = b; p2 = c; p3 = i;
                }
            }
            else // Upper tetrahedron (c, g, h, i)
            {
                p0 = c; p1 = g; p2 = h; p3 = i;
            }
        }
        else // Prism C
        {
            if (lodLerp > 1.0 - ratioLerp) // Upper pyramid
            {
                if (RemapTo01(lodLerp, 1.0 - ratioLerp, 1.0) > RemapTo01((thetaBinLerp - 0.5) * 2.0, 1.0 - ratioLerp, 1.0)) // Left-up tetrahedron (c, j, h, i)
                {
                    p0 = c; p1 = j; p2 = h; p3 = i;
                }
                else // Right-down tetrahedron (e, i, c, j)
                {
                    p0 = e; p1 = i; p2 = c; p3 = j;
                }
            }
            else // Lower tetrahedron (d, e, c, i)
            {
                p0 = d; p1 = e; p2 = c; p3 = i;
            }
        }
    }
}

vec4 SampleGlints2023NDF(vec3 localHalfVector, float targetNDF, float maxNDF, vec2 uv, vec2 duvdx, vec2 duvdy)
{
    // ACCURATE PIXEL FOOTPRINT ELLIPSE
    vec2 ellipseMajor, ellipseMinor;
    GetGradientEllipse(duvdx, duvdy, ellipseMajor, ellipseMinor);
    float ellipseRatio = length(ellipseMajor) / length(ellipseMinor);

    // SHARED GLINT NDF VALUES
    float halfScreenSpaceScaler = _ScreenSpaceScale * 0.5;
    float footprintArea = length(ellipseMajor) * halfScreenSpaceScaler * length(ellipseMinor) * halfScreenSpaceScaler * 4.0;
    vec2 slope = localHalfVector.xy; // Orthogrtaphic slope projected grid
    float rescaledTargetNDF = targetNDF / maxNDF;

    // MANUAL LOD COMPENSATION
    float lod = log2(length(ellipseMinor) * halfScreenSpaceScaler);
    float lod0 = floor(lod);
    float lod1 = lod0 + 1.0;
    float divLod0 = pow(2.0, lod0);
    float divLod1 = pow(2.0, lod1);
    float lodLerp = fract(lod);
    float footprintAreaLOD0 = pow(exp2(lod0), 2.0);
    float footprintAreaLOD1 = pow(exp2(lod1), 2.0);

    // MANUAL ANISOTROPY RATIO COMPENSATION
    float ratio0 = max(pow(2.0, floor(log2(ellipseRatio))), 1.0);
    float ratio1 = ratio0 * 2.0;
    float ratioLerp = clamp(Remap(ellipseRatio, ratio0, ratio1, 0.0, 1.0), 0.0, 1.0);

    // MANUAL ANISOTROPY ROTATION COMPENSATION
    vec2 v1 = vec2(0.0, 1.0);
    vec2 v2 = normalize(ellipseMajor);
    float theta = atan(v2.y, v2.x) * RAD2DEG;
    float thetaGrid = 90.0 / max(ratio0, 2.0);
    float thetaBin = floor(theta / thetaGrid) * thetaGrid;
    thetaBin = thetaBin + (thetaGrid / 2.0);
    float thetaBin0 = theta < thetaBin ? thetaBin - thetaGrid / 2.0 : thetaBin;
    float thetaBinH = thetaBin0 + thetaGrid / 4.0;
    float thetaBin1 = thetaBin0 + thetaGrid / 2.0;
    float thetaBinLerp = Remap(theta, thetaBin0, thetaBin1, 0.0, 1.0);
    thetaBin0 = thetaBin0 <= 0.0 ? 180.0 + thetaBin0 : thetaBin0;

    // TETRAHEDRONIZATION OF ROTATION + RATIO + LOD GRID
    bool centerSpecialCase = (ratio0 == 1.0);
    vec2 divLods = vec2(divLod0, divLod1);
    vec2 footprintAreas = vec2(footprintAreaLOD0, footprintAreaLOD1);
    vec2 ratios = vec2(ratio0, ratio1);
    vec4 thetaBins = vec4(thetaBin0, thetaBinH, thetaBin1, 0.0); // added 0.0 for center singularity case
    vec3 tetraA, tetraB, tetraC, tetraD;
    GetAnisoCorrectingGridTetrahedron(centerSpecialCase, thetaBinLerp, ratioLerp, lodLerp, tetraA, tetraB, tetraC, tetraD);
    if (centerSpecialCase) // Account for center singularity in barycentric computation
        thetaBinLerp = Remap01To(thetaBinLerp, 0.0, ratioLerp);
    vec4 tetraBarycentricWeights = GetBarycentricWeightsTetrahedron(vec3(thetaBinLerp, ratioLerp, lodLerp), tetraA, tetraB, tetraC, tetraD); // Compute barycentric coordinates within chosen tetrahedron

    // PREPARE NEEDED ROTATIONS
    tetraA.x *= 2.0; tetraB.x *= 2.0; tetraC.x *= 2.0; tetraD.x *= 2.0;
    if (centerSpecialCase) // Account for center singularity (if center vertex => no rotation)
    {
        tetraA.x = (tetraA.y == 0.0) ? 3.0 : tetraA.x;
        tetraB.x = (tetraB.y == 0.0) ? 3.0 : tetraB.x;
        tetraC.x = (tetraC.y == 0.0) ? 3.0 : tetraC.x;
        tetraD.x = (tetraD.y == 0.0) ? 3.0 : tetraD.x;
    }
    vec2 uvRotA = RotateUV(uv, thetaBins[int(tetraA.x)] * DEG2RAD, vec2(0.0));
    vec2 uvRotB = RotateUV(uv, thetaBins[int(tetraB.x)] * DEG2RAD, vec2(0.0));
    vec2 uvRotC = RotateUV(uv, thetaBins[int(tetraC.x)] * DEG2RAD, vec2(0.0));
    vec2 uvRotD = RotateUV(uv, thetaBins[int(tetraD.x)] * DEG2RAD, vec2(0.0));

    // SAMPLE GLINT GRIDS
    uint gridSeedA = uint(HashWithoutSine13(vec3(log2(divLods[int(tetraA.z)]), mod(thetaBins[int(tetraA.x)], 360.0), ratios[int(tetraA.y)])) * 4294967296.0);
    uint gridSeedB = uint(HashWithoutSine13(vec3(log2(divLods[int(tetraB.z)]), mod(thetaBins[int(tetraB.x)], 360.0), ratios[int(tetraB.y)])) * 4294967296.0);
    uint gridSeedC = uint(HashWithoutSine13(vec3(log2(divLods[int(tetraC.z)]), mod(thetaBins[int(tetraC.x)], 360.0), ratios[int(tetraC.y)])) * 4294967296.0);
    uint gridSeedD = uint(HashWithoutSine13(vec3(log2(divLods[int(tetraD.z)]), mod(thetaBins[int(tetraD.x)], 360.0), ratios[int(tetraD.y)])) * 4294967296.0);
    float sampleA = SampleGlintGridSimplex(uvRotA / divLods[int(tetraA.z)] / vec2(1.0, ratios[int(tetraA.y)]), gridSeedA, slope, ratios[int(tetraA.y)] * footprintAreas[int(tetraA.z)], rescaledTargetNDF, tetraBarycentricWeights.x);
    float sampleB = SampleGlintGridSimplex(uvRotB / divLods[int(tetraB.z)] / vec2(1.0, ratios[int(tetraB.y)]), gridSeedB, slope, ratios[int(tetraB.y)] * footprintAreas[int(tetraB.z)], rescaledTargetNDF, tetraBarycentricWeights.y);
    float sampleC = SampleGlintGridSimplex(uvRotC / divLods[int(tetraC.z)] / vec2(1.0, ratios[int(tetraC.y)]), gridSeedC, slope, ratios[int(tetraC.y)] * footprintAreas[int(tetraC.z)], rescaledTargetNDF, tetraBarycentricWeights.z);
    float sampleD = SampleGlintGridSimplex(uvRotD / divLods[int(tetraD.z)] / vec2(1.0, ratios[int(tetraD.y)]), gridSeedD, slope, ratios[int(tetraD.y)] * footprintAreas[int(tetraD.z)], rescaledTargetNDF, tetraBarycentricWeights.w);
    return vec4((sampleA + sampleB + sampleC + sampleD) * (1.0 / _MicrofacetRoughness) * maxNDF);
}

void main()
{
    // Get the correct values
    vec3  N                = fs_Nor;
    vec3  albedo           = u_Albedo;
    float metallic         = u_Metallic;
    float roughness        = u_Roughness;
    float ambientOcclusion = u_AmbientOcclusion;

    handleMaterialMaps(albedo, metallic, roughness, ambientOcclusion, N);

    // Calculate the lighting
    vec3 wo = normalize(u_CamPos - fs_Pos);
    // Reflecting wo about N
    vec3 wi = reflect(-wo, N);
    vec3 wh = normalize(wi + wo);

    /**
        The diffuse part.
    */
    vec3 R = mix(vec3(0.04), albedo, metallic);
    // TODO: But N should be wh? How to get wh? Isotropic lobe assumption?
    vec3 kS = fresnelSchlickRoughness(max(dot(N, wo), 0.0), R, roughness); 
    vec3 kD = 1.0 - kS;
    // The texture sampling correspond to PI * (the whole thing in the diffuse integral, including its cosine term) / num samples
    // Therefore we need to time it by kD and albedo to get the diffuse Lo
    // Also given that the hemisphere is oriented along N, we pass N as the fs_Pos for the convolution
    vec3 diffuse    = texture(u_DiffuseIrradianceMap, N).rgb * albedo;

    /**
        The specular part.
    */
    // The Li part of the split sum approximation.
    const float MAX_REFLECTION_LOD = 4.0;
    vec3 Li = textureLod(u_GlossyIrradianceMap, wi, roughness * MAX_REFLECTION_LOD).rgb;

    vec3 specular;
    if (useGlint) {
        specular = Li * SampleGlints2023NDF(wh, 0.5, 1.0, fs_UV, dFdx(fs_UV), dFdy(fs_UV)).rgb;
    }
    else {
        // The BRDF part of the split sum approximation.
        vec2 brdf = texture(u_BRDFLookupTexture, vec2(max(dot(N, wo), 0.0), roughness)).rg;
        // Complete split sum approximation.
        specular = Li * (kS * brdf.x + brdf.y);
    }

    // Looks like the ref on GitHub does not have this term
    vec3 ambient = (kD * diffuse + specular) * ambientOcclusion; 

    // if you wanna add point light to the scene as well, write:
    // vec3 col = ambient + Lo;
    // where Lo is the point light radiance 
    vec3 col = ambient; 

    // Reinhard operator: c' = c / (c + 1)
    col = col / (col + 1.0);

    // Gamma correction: c' = c^(1/gamma)
    col = pow(col, vec3(1.0 / 2.2));

    out_Col = vec4(col, 1.f);
}
