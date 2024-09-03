#version 330 core

// Compute the irradiance across the entire
// hemisphere aligned with a surface normal
// pointing in the direction of fs_Pos.
// Thus, our surface normal direction
// is normalize(fs_Pos).

in vec3 fs_Pos;
out vec4 out_Col;
uniform samplerCube u_EnvironmentMap;

const float PI = 3.14159265359;

// Followed: https://learnopengl.com/PBR/IBL/Diffuse-irradiance
// Correspond to: pi * (the whole thing in the diffuse integral, including its cosine term) / num samples
void main()
{
    vec3 irradiance = vec3(0.0);  

    vec3 normal = normalize(fs_Pos);
    vec3 up    = vec3(0.0, 1.0, 0.0);
    vec3 right = normalize(cross(up, normal));
    up         = normalize(cross(normal, right));

    float sampleDelta = 0.025;
    float nrSamples = 0.0; 
    for(float phi = 0.0; phi < 2.0 * PI; phi += sampleDelta)
    {
        for(float theta = 0.0; theta < 0.5 * PI; theta += sampleDelta)
        {
            // spherical to cartesian (in tangent space)
            vec3 wi = vec3(sin(theta) * cos(phi),  sin(theta) * sin(phi), cos(theta));
            // tangent space to world
            vec3 wiW = wi.x * right + wi.y * up + wi.z * normal; 

            irradiance += texture(u_EnvironmentMap, wiW).rgb * cos(theta) * sin(theta);
            nrSamples++;
        }
    }
    irradiance = PI * irradiance * (1.0 / float(nrSamples));

    out_Col = vec4(irradiance, 1.0);
}
