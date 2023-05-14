struct VertexOut {
    @location(0) uv: vec2<f32>,
    @builtin(position) position: vec4<f32>,
};

struct Uniforms {
    width: u32,
    height: u32,
};

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@group(0) @binding(1)
var<storage> render_buffer: array<f32>;

var<private> v_positions: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0),
    vec2<f32>(-1.0, 1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(1.0, 1.0),
    vec2<f32>(1.0, -1.0),
    vec2<f32>(-1.0, -1.0),
);

@vertex
fn vs_main(@builtin(vertex_index) v_idx: u32) -> VertexOut {
    var out: VertexOut;
    out.position = vec4<f32>(v_positions[v_idx], 0.0, 1.0);
    out.uv = v_positions[v_idx] * 0.5 + 0.5;
    return out;
}

fn aces(x: f32) -> f32 {
  var a = 2.51;
  var b = 0.03;
  var c = 2.43;
  var d = 0.59;
  var e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0);
}

fn rheinhard(x: f32) -> f32 {
    return x / (x + 1.0);
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    var uv = in.uv;
    uv.y = 1.0 - uv.y;
    var puv: vec2<u32> = vec2<u32>(uv * vec2<f32>(f32(uniforms.width), f32(uniforms.height)));
    var idx: u32 = (puv.y*u32(uniforms.width)+puv.x);
    var color: vec4<f32> = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    color.r = render_buffer[idx*3u+0u];
    color.g = render_buffer[idx*3u+1u];
    color.b = render_buffer[idx*3u+2u];
    return vec4<f32>(color.rgb, 1.0);
}