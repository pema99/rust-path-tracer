struct VertexOut {
    @location(0) uv: vec2<f32>,
    @builtin(position) position: vec4<f32>,
};

struct Uniforms {
    width: u32,
    height: u32,
    tonemapping: u32,
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

fn aces(x: vec3<f32>) -> vec3<f32> {
  var a = 2.51;
  var b = 0.03;
  var c = 2.43;
  var d = 0.59;
  var e = 0.14;
  return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

fn reinhard(x: vec3<f32>) -> vec3<f32> {
    return x / (x + 1.0);
}

fn neutralCurve(x: vec3<f32>, a: f32, b: f32, c: f32, d: f32, e: f32, f: f32) -> vec3<f32> {
    return ((x * (a * x + c * b) + d * e) / (x * (a * x + b) + d * f)) - e / f;
}

fn neutralTonemap(x: vec3<f32>) -> vec3<f32> {
    // Tonemap
    var a: f32 = 0.2;
    var b: f32 = 0.29;
    var c: f32 = 0.24;
    var d: f32 = 0.272;
    var e: f32 = 0.02;
    var f: f32 = 0.3;
    var whiteLevel: f32 = 5.3;
    var whiteClip: f32 = 1.0;

    var whiteScale: vec3<f32> = vec3<f32>(1.0) / neutralCurve(vec3<f32>(whiteLevel), a, b, c, d, e, f);
    var x = neutralCurve(x * whiteScale, a, b, c, d, e, f);
    x *= whiteScale;

    // Post-curve white point adjustment
    x /= vec3<f32>(whiteClip);

    return x;
}

fn unchartedPartial(x: vec3<f32>) -> vec3<f32>
{
    var A = 0.15f;
    var B = 0.50f;
    var C = 0.10f;
    var D = 0.20f;
    var E = 0.02f;
    var F = 0.30f;
    return ((x*(A*x+C*B)+D*E)/(x*(A*x+B)+D*F))-E/F;
}

fn uncharted(v: vec3<f32>) -> vec3<f32>
{
    var exposure_bias = 2.0;
    var curr = unchartedPartial(v * exposure_bias);

    var W = vec3(11.2);
    var white_scale = vec3<f32>(1.0) / unchartedPartial(W);
    return curr * white_scale;
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

    var tonemapped = color.rgb;
    switch (uniforms.tonemapping) {
        case 1u: { // Reinhard
            tonemapped = reinhard(tonemapped);
        }
        case 2u: { // ACES
            tonemapped = aces(tonemapped);
        }
        case 3u: { // Neutral
            tonemapped = neutralTonemap(tonemapped);
        }
        case 4u: { // Uncharted
            tonemapped = uncharted(tonemapped);
        }
        default: {
            // No tonemapping
        }
    }

    return vec4<f32>(tonemapped, 1.0);
}