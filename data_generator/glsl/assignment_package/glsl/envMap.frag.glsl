#version 330 core

in vec3 fs_Pos;

uniform uint backgroundType;

out vec4 out_Col;

uniform samplerCube u_EnvironmentMap;

void main() {
    // vec3 envColor = texture(u_EnvironmentMap, fs_Pos).rgb;
    // Uncomment the line below if you want to see different mip levels
    // of your glossy irradiance map. Change 1.2 to anything up to 4
    // to see the different mip levels.
    vec3 envColor = textureLod(u_EnvironmentMap, fs_Pos, 1.2).rgb;

    // Reinhard op + gamma correction
    envColor = envColor / (envColor + vec3(1.0));
    envColor = pow(envColor, vec3(1.0/2.2));

    out_Col = vec4(envColor, 1.0);
    if (backgroundType == 1u) {
        // 1 is the index for a black background
        out_Col = vec4(vec3(0.0), 1.0);
    }
    else if (backgroundType == 2u) {
        // 2 is the index for a white background
        out_Col = vec4(1.0);
    }
    else if (backgroundType == 3u) {
        // 3 is the index for a transparent background
        out_Col = vec4(0.0);
    }
}
