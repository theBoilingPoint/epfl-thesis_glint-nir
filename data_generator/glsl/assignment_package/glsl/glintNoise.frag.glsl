#version 330 core

uniform uint _FrameSize;
// Range of seed should be [0, 99]
uniform uint _Seed;

in vec3 fs_Pos;
in vec2 fs_UV;

out vec4 out_Col;

void main() {
    out_Col = vec4(1.0);
}

// #version 330 core
// #extension GL_ARB_shading_language_packing : enable

// const float GAUSSIAN_AVG = 0.0;
// const float GAUSSIAN_STD = 1.0;

// uniform uint _FrameSize;
// // Range of seed should be [0, 99]
// uniform uint _Seed;

// in vec3 fs_Pos;
// in vec2 fs_UV;

// out vec4 out_Col;

// uint CoordToFlatId(uvec2 coord)
// {
//     return coord.y * _FrameSize + coord.x;
// }

// float PackFloats(float a, float b)
// {
//     // Convert float to 16-bit half-float
//     uint a16 = floatBitsToUint(packHalf2x16(vec2(a, 0.0))) & 0xFFFFu;
//     uint b16 = floatBitsToUint(packHalf2x16(vec2(b, 0.0))) & 0xFFFFu;

//     // Combine the two 16-bit values into a 32-bit unsigned int
//     uint abPacked = (a16 << 16) | b16;

//     // Convert the packed 32-bit unsigned int back to float
//     return uintBitsToFloat(abPacked);
// }

// uint WangHash(uint seed)
// {
//     seed = (seed ^ 61u) ^ (seed >> 16);
//     seed *= 9u;
//     seed = seed ^ (seed >> 4);
//     seed *= 0x27d4eb2du;
//     seed = seed ^ (seed >> 15);
//     return seed;
// }

// void RandXorshift(inout uint rngState)
// {
//     // Xorshift algorithm from George Marsaglia's paper
//     rngState ^= (rngState << 13);
//     rngState ^= (rngState >> 17);
//     rngState ^= (rngState << 5);
// }

// float RandXorshiftFloat(inout uint rngState)
// {
//     RandXorshift(rngState);
//     float res = float(rngState) * (1.0 / 4294967296.0);
//     return res;
// }

// float Erf(float x)
// {
//     // Save the sign of x
//     int sign = 1;
//     if (x < 0)
//         sign = -1;
//     x = abs(x);

//     // A&S formula 7.1.26
//     float t = 1.0 / (1.0 + 0.3275911 * x);
//     float y = 1.0 - (((((1.061405429 * t + -1.453152027) * t) + 1.421413741)
//         * t + -0.284496736) * t + 0.254829592) * t * exp(-x * x);

//     return sign * y;
// }

// float ErfInv(float x)
// {
//     float w, p;
//     w = -log((1.0 - x) * (1.0 + x));
//     if (w < 5.000000)
//     {
//         w = w - 2.500000;
//         p = 2.81022636e-08;
//         p = 3.43273939e-07 + p * w;
//         p = -3.5233877e-06 + p * w;
//         p = -4.39150654e-06 + p * w;
//         p = 0.00021858087 + p * w;
//         p = -0.00125372503 + p * w;
//         p = -0.00417768164 + p * w;
//         p = 0.246640727 + p * w;
//         p = 1.50140941 + p * w;
//     }
//     else
//     {
//         w = sqrt(w) - 3.000000;
//         p = -0.000200214257;
//         p = 0.000100950558 + p * w;
//         p = 0.00134934322 + p * w;
//         p = -0.00367342844 + p * w;
//         p = 0.00573950773 + p * w;
//         p = -0.0076224613 + p * w;
//         p = 0.00943887047 + p * w;
//         p = 1.00167406 + p * w;
//         p = 2.83297682 + p * w;
//     }
//     return p * x;
// }

// float InvCDF(float U, float mu, float sigma)
// {
//     float x = sigma * sqrt(2.0) * ErfInv(2.0 * U - 1.0) + mu;
//     return x;
// }

// void main()
// {
//     uvec2 size = uvec2(_FrameSize);

//     // Generate random numbers for this cell and the next ones in X and Y
//     uvec2 pixelCoord00 = uvec2(fs_UV * vec2(_FrameSize));
//     uint rngState00 = WangHash(CoordToFlatId(pixelCoord00 * 123u) + size.x * size.y * _Seed);
//     float u00 = RandXorshiftFloat(rngState00);
//     float g00 = InvCDF(RandXorshiftFloat(rngState00), GAUSSIAN_AVG, GAUSSIAN_STD);

//     uvec2 pixelCoord01 = (pixelCoord00 + uvec2(0, 1)) % size;
//     uint rngState01 = WangHash(CoordToFlatId(pixelCoord01 * 123u) + size.x * size.y * _Seed);
//     float u01 = RandXorshiftFloat(rngState01);
//     float g01 = InvCDF(RandXorshiftFloat(rngState01), GAUSSIAN_AVG, GAUSSIAN_STD);

//     uvec2 pixelCoord10 = (pixelCoord00 + uvec2(1, 0)) % size;
//     uint rngState10 = WangHash(CoordToFlatId(pixelCoord10 * 123u) + size.x * size.y * _Seed);
//     float u10 = RandXorshiftFloat(rngState10);
//     float g10 = InvCDF(RandXorshiftFloat(rngState10), GAUSSIAN_AVG, GAUSSIAN_STD);

//     uvec2 pixelCoord11 = (pixelCoord00 + uvec2(1, 1)) % size;
//     uint rngState11 = WangHash(CoordToFlatId(pixelCoord11 * 123u) + size.x * size.y * _Seed);
//     float u11 = RandXorshiftFloat(rngState11);
//     float g11 = InvCDF(RandXorshiftFloat(rngState11), GAUSSIAN_AVG, GAUSSIAN_STD);

//     // Pack 8 values into 4
//     out_Col = vec4(PackFloats(u00, g00), PackFloats(u01, g01), PackFloats(u10, g10), PackFloats(u11, g11));
// }