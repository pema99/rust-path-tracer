use spirv_std::glam::Vec3;
#[allow(unused_imports)]
use spirv_std::num_traits::Float;

pub const EPS: f32 = 0.001;

pub fn uniform_sample_sphere(r1: f32, r2: f32) -> Vec3 {
    let cos_phi = 2.0 * r1 - 1.0;
    let sin_phi = (1.0 - cos_phi * cos_phi).sqrt();
    let theta = 2.0 * core::f32::consts::PI * r2;
    Vec3::new(sin_phi * theta.cos(), cos_phi, sin_phi * theta.cos())
}

pub fn uniform_sample_hemisphere(r1: f32, r2: f32) -> Vec3 {
    let sin_theta = (1.0 - r1 * r1).sqrt();
    let phi = 2.0 * core::f32::consts::PI * r2;
    let x = sin_theta * phi.cos();
    let z = sin_theta * phi.sin();
    Vec3::new(x, r1, z)
}

pub fn cosine_sample_hemisphere(r1: f32, r2: f32) -> Vec3 {
    let theta = r1.sqrt().acos();
    let phi = 2.0 * core::f32::consts::PI * r2;
    Vec3::new(
        theta.sin() * phi.cos(),
        theta.cos(),
        theta.sin() * phi.sin(),
    )
}

pub fn create_cartesian(up: Vec3) -> (Vec3, Vec3, Vec3) {
    let arbitrary = Vec3::new(0.1, 0.5, 0.9);
    let temp_vec = up.cross(arbitrary).normalize();
    let right = temp_vec.cross(up).normalize();
    let forward = up.cross(right).normalize();
    (up, right, forward)
}

pub fn reflect(i: Vec3, normal: Vec3) -> Vec3 {
    i - normal * 2.0 * i.dot(normal)
}

pub fn ggx_distribution(normal: Vec3, halfway: Vec3, roughness: f32) -> f32 {
    let numerator = roughness * roughness;
    let n_dot_h = normal.dot(halfway).max(0.0);
    let mut denominator = (n_dot_h * n_dot_h) * (numerator - 1.0) + 1.0;
    denominator = (core::f32::consts::PI * (denominator * denominator)).max(EPS);
    numerator / denominator
}

pub fn sample_ggx(r1: f32, r2: f32, reflection_direction: Vec3, roughness: f32) -> Vec3 {
    let a = roughness * roughness;

    let phi = 2.0 * core::f32::consts::PI * r1;
    let cos_theta = ((1.0 - r2) / (r2 * (a * a - 1.0) + 1.0)).sqrt();
    let sin_theta = (1.0 - cos_theta * cos_theta).sqrt();

    let halfway = Vec3::new(phi.cos() * sin_theta, phi.sin() * sin_theta, cos_theta);

    let up = if reflection_direction.z.abs() < 0.999 {
        Vec3::new(0.0, 0.0, 1.0)
    } else {
        Vec3::new(1.0, 0.0, 0.0)
    };
    let tangent = up.cross(reflection_direction).normalize();
    let bitangent = reflection_direction.cross(tangent);

    (tangent * halfway.x + bitangent * halfway.y + reflection_direction * halfway.z).normalize()
}

pub fn geometry_schlick_ggx(normal: Vec3, view_direction: Vec3, roughness: f32) -> f32 {
    let numerator = normal.dot(view_direction).max(0.0);
    let r = (roughness * roughness) / 8.0;
    let denominator = numerator * (1.0 - r) + r;
    numerator / denominator
}

pub fn geometry_smith(
    normal: Vec3,
    view_direction: Vec3,
    light_direction: Vec3,
    roughness: f32,
) -> f32 {
    geometry_schlick_ggx(normal, view_direction, roughness)
        * geometry_schlick_ggx(normal, light_direction, roughness)
}

pub fn fresnel_schlick(cos_theta: f32, f0: Vec3) -> Vec3 {
    f0 + (Vec3::ONE - f0) * (1.0 - cos_theta).powi(5)
}
