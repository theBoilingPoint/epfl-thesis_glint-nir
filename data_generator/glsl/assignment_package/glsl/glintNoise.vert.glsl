#version 330 core

uniform mat4 u_Model;
uniform mat4 u_ViewProj;

in vec3 vs_Pos;
in vec2 vs_UV;

out vec3 fs_Pos;
out vec2 fs_UV;

void main()
{
    fs_Pos = (u_ViewProj * u_Model * vec4(vs_Pos, 1.0)).xyz;
    fs_UV = vs_UV;

    gl_Position =  u_ViewProj * vec4(fs_Pos, 1.0);
}