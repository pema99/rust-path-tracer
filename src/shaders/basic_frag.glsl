#version 140

in vec2 v_tex_coords;
out vec4 color;

uniform samplerBuffer texture;
uniform float width;
uniform float height;

void main() {
    vec2 uv = v_tex_coords.xy*0.5+0.5;
    uv.y = 1.0-uv.y;
    ivec2 puv = ivec2(uv * vec2(width, height));
    int idx = (puv.y*int(width)+puv.x);
    color.r = texelFetch(texture, idx*3+0).r;
    color.g = texelFetch(texture, idx*3+1).r;
    color.b = texelFetch(texture, idx*3+2).r;
    color.rgb = pow(color.rgb, vec3(2.2));
}