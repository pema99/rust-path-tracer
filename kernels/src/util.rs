use spirv_std::glam::Vec3;
#[allow(unused_imports)]
use spirv_std::num_traits::Float;

pub const EPS: f32 = 0.001;

#[allow(dead_code)]
pub fn uniform_sample_sphere(r1: f32, r2: f32) -> Vec3 {
    let cos_phi = 2.0 * r1 - 1.0;
    let sin_phi = (1.0 - cos_phi * cos_phi).sqrt();
    let theta = 2.0 * core::f32::consts::PI * r2;
    Vec3::new(sin_phi * theta.cos(), cos_phi, sin_phi * theta.cos())
}

#[allow(dead_code)]
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

#[allow(dead_code)]
pub fn refract(i: Vec3,  normal: Vec3, in_ior: f32, out_ior: f32) -> Vec3 {
    let eta = in_ior / out_ior;
    let n_dot_i = normal.dot(i);
    let k = 1.0 - eta * eta * (1.0 - n_dot_i.powi(2));
    if k < 0.0 {
        Vec3::ZERO
    } else {
        eta * i - (eta * n_dot_i + k.sqrt()) * normal
    }
}

pub fn ggx_distribution(normal: Vec3, halfway: Vec3, roughness: f32) -> f32 {
    let numerator = roughness * roughness;
    let n_dot_h = normal.dot(halfway).max(0.0);
    let mut denominator = (n_dot_h * n_dot_h) * (numerator - 1.0) + 1.0;
    denominator = (core::f32::consts::PI * (denominator * denominator)).max(EPS);
    numerator / denominator
}

// https://blog.selfshadow.com/publications/s2013-shading-course/karis/s2013_pbs_epic_notes_v2.pdf
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

// GGX distribution with respect to microsurface normal (NOT halfway vector)
// https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf equation 33
#[allow(dead_code)]
pub fn ggx_distribution_microsurface_normal(
    m_dot_n: f32, // microsurface normal dot macrosurface normal
    roughness: f32
) -> f32 {
    let a_g = roughness * roughness;
    let a_g2 = a_g * a_g;
    let theta_m = m_dot_n.acos();
    let numerator = a_g2 * positive_characteristic(m_dot_n);
    let denominator = core::f32::consts::PI * theta_m.cos().powi(4) * (a_g2 + theta_m.tan().powi(2)).powi(2);
    numerator / denominator
}

// GGX distribution multiplied with the dot of micro and macrosurface normals.
// This what sample_ggx_microsurface_normal samples w.r.t
// https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf equation 24
#[allow(dead_code)]
pub fn ggx_pdf_microsurface_normal(
    microsurface_normal: Vec3,
    macrosurface_normal: Vec3,
    roughness: f32
) -> f32 {
    let m_dot_n = microsurface_normal.dot(macrosurface_normal);
    ggx_distribution_microsurface_normal(m_dot_n, roughness) * m_dot_n.abs()
}

// Function for sampling GGX(m)*|m.n| with respect to microsurface normal
// https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf equation 35-36
pub fn sample_ggx_microsurface_normal(
    r1: f32,
    r2: f32,
    macrosurface_normal: Vec3,
    roughness: f32
) -> Vec3 {
    let a_g = roughness * roughness;

    let theta_m = ((a_g * r1.sqrt()) / (1.0 - r1).sqrt()).atan();
    let phi_m = 2.0 * core::f32::consts::PI * r2;

    let m = Vec3::new(
        theta_m.sin() * phi_m.cos(),
        theta_m.cos(),
        theta_m.sin() * phi_m.sin());

    let (up, nt, nb) = create_cartesian(macrosurface_normal);
    Vec3::new(
        m.x * nb.x + m.y * up.x + m.z * nt.x,
        m.x * nb.y + m.y * up.y + m.z * nt.y,
        m.x * nb.z + m.y * up.z + m.z * nt.z,
    ).normalize()
}

// PDF for sampling GGX(m)*|m.n| in case of reflection
// https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf equation 38, 13 and 14
#[allow(dead_code)]
pub fn ggx_pdf_reflection(
    incoming_direction: Vec3, // aka. light direction
    outgoing_direction: Vec3, // aka. scattered direction
    macrosurface_normal: Vec3,
    pdf_microsurface_normal: f32, // pdf of sampling the specific microsurface normal
) -> f32 {
    let h_r_hat = incoming_direction.dot(macrosurface_normal).signum() * (incoming_direction + outgoing_direction);
    let h_r = h_r_hat.normalize();
    let jacobian = 1.0 / (4.0 * outgoing_direction.dot(h_r).abs());
    pdf_microsurface_normal * jacobian
}

// PDF for sampling GGX(m)*|m.n| in case of refraction
// https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf equation 38, 13 and 14
#[allow(dead_code)]
pub fn ggx_pdf_refraction(
    incoming_direction: Vec3, // aka. light direction
    outgoing_direction: Vec3, // aka. scattered direction
    in_ior: f32,
    out_ior: f32,
    pdf_microsurface_normal: f32, // pdf of sampling the specific microsurface normal
) -> f32 {
    let h_t_hat = -(in_ior * incoming_direction + out_ior * outgoing_direction);
    let h_t = h_t_hat.normalize();
    let o_dot_h_t = outgoing_direction.dot(h_t);
    let numerator = out_ior.powi(2) * o_dot_h_t.abs();
    let denominator = (in_ior * incoming_direction.dot(h_t) + out_ior * (outgoing_direction.dot(h_t))).powi(2);
    let jacobian = numerator / denominator;
    pdf_microsurface_normal * jacobian
}

// GGX geometry function w.r.t microsurface normal. Not the Schlick-GGX approximation.
// https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf equation 34
pub fn geometry_ggx_microfacet_normal(
    macrosurface_normal: Vec3,
    microsurface_normal: Vec3,
    view_direction: Vec3,
    roughness: f32
) -> f32 {
    let a_g = roughness * roughness;
    let a_g2 = a_g * a_g;

    let v_dot_n = view_direction.dot(macrosurface_normal);
    let v_dot_m = view_direction.dot(microsurface_normal);
    let numerator = 2.0 * positive_characteristic(v_dot_m / v_dot_n);

    let theta_v = view_direction.dot(macrosurface_normal).acos();
    let denominator = 1.0 + (1.0 + a_g2 * theta_v.tan().powi(2)).sqrt();

    numerator / denominator
}

// Geometry-Smith term w.r.t microsurface normal. Not the Schlick-GGX approximation.
// https://www.cs.cornell.edu/~srm/publications/EGSR07-btdf.pdf equation 23
#[allow(dead_code)]
pub fn geometry_smith_microfacet_normal(
    macrosurface_normal: Vec3,
    microsurface_normal: Vec3,
    view_direction: Vec3,
    light_direction: Vec3,
    roughness: f32,
) -> f32 {
    geometry_ggx_microfacet_normal(macrosurface_normal, microsurface_normal, view_direction, roughness)
        * geometry_ggx_microfacet_normal(macrosurface_normal, microsurface_normal, light_direction, roughness)
}

// Schlick-GGX geometry function from https://learnopengl.com/pbr/theory
pub fn geometry_schlick_ggx(normal: Vec3, view_direction: Vec3, roughness: f32) -> f32 {
    let numerator = normal.dot(view_direction).max(0.0);
    let r = (roughness * roughness) / 8.0;
    let denominator = numerator * (1.0 - r) + r;
    numerator / denominator
}

// Geometry-Smith term based on Schlick-GGX from https://learnopengl.com/pbr/theory
pub fn geometry_smith_schlick_ggx(
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

pub fn fresnel_schlick_scalar(in_ior: f32, out_ior: f32, cos_theta: f32) -> f32 {
    let f0 = ((in_ior - out_ior) / (in_ior + out_ior)).powi(2);
    f0 + (1.0 - f0) * (1.0 - cos_theta).powi(5)
}

pub fn barycentric(p: Vec3, a: Vec3, b: Vec3, c: Vec3) -> Vec3 {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let d00 = v0.dot(v0);
    let d01 = v0.dot(v1);
    let d11 = v1.dot(v1);
    let d20 = v2.dot(v0);
    let d21 = v2.dot(v1);
    let denom = d00 * d11 - d01 * d01;
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    Vec3::new(1.0 - v - w, v, w)
}

pub fn power_heuristic(p1: f32, p2: f32) -> f32 {
    let p1_2 = p1 * p1;
    p1_2 / (p1_2 + p2 * p2)
}

#[allow(dead_code)]
pub fn balance_heuristic(p1: f32, p2: f32) -> f32 {
    p1 / (p1 + p2)
}

pub fn positive_characteristic(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else {
        0.0
    }
}

pub fn mask_nan(v: Vec3) -> Vec3 {
    if v.is_finite() {
        v
    } else {
        Vec3::ZERO
    }
}

pub fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a * (1.0 - t) + b * t
}