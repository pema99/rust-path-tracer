#version 140

in vec2 v_tex_coords;
out vec4 color;
uniform sampler2D framebuffer;

void main() {
    vec2 uv = v_tex_coords.xy*0.5+0.5;
    uv.y = 1.0-uv.y;
    color = texture(framebuffer, uv);
    color.rgb = pow(color.rgb, vec3(2.2));
}