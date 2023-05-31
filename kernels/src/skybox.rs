use spirv_std::glam::{Vec2, Vec3, Vec4, Vec4Swizzles};
#[allow(unused_imports)]
use spirv_std::num_traits::Float;

use crate::util;

// Constants
const RAY_SCATTER_COEFF: Vec3 = Vec3::new(58e-7, 135e-7, 331e-7);
const RAY_EFFECTIVE_COEFF: Vec3 = RAY_SCATTER_COEFF; // Rayleight doesn't absorb light
const MIE_SCATTER_COEFF: Vec3 = Vec3::new(2e-5, 2e-5, 2e-5);
const MIE_EFFECTIVE_COEFF: Vec3 = Vec3::new(2e-5 * 1.1, 2e-5 * 1.1, 2e-5 * 1.1); // Approximate absorption as a factor of scattering
const EARTH_RADIUS: f32 = 6360e3;
const ATMOSPHERE_RADIUS: f32 = 6380e3;
const H_RAY: f32 = 8e3;
const H_MIE: f32 = 12e2;
const CENTER: Vec3 = Vec3::new(0.0, -EARTH_RADIUS, 0.0); // earth center point

fn escape(p: Vec3, d: Vec3, r: f32) -> f32 {
    let v = p - CENTER;
    let b = v.dot(d);
    let det = b * b - v.dot(v) + r * r;
    if det < 0.0 {
        return -1.0;
    }
    let det = det.sqrt();
    let t1 = -b - det;
    let t2 = -b + det;
    if t1 >= 0.0 {
        return t1;
    }
    return t2;
}

fn densities_rm(p: Vec3) -> Vec2 {
    let h = ((p - CENTER).length() - EARTH_RADIUS).max(0.0);
    let exp_h_ray = (-h / H_RAY).exp();
    let exp_h_mie = (-h / H_MIE).exp();
    Vec2::new(exp_h_ray, exp_h_mie)
}

fn scatter_depth_int(o: Vec3, d: Vec3, l: f32) -> Vec2 {
    // Approximate by combining 2 samples
    densities_rm(o) * (l / 2.) + densities_rm(o + d * l) * (l / 2.)
}

fn scatter_in(origin: Vec3, direction: Vec3, depth: f32, steps: u32, sundir: Vec3) -> (Vec3, Vec3) {
    let depth = depth / steps as f32;

    let mut i_r = Vec3::ZERO;
    let mut i_m = Vec3::ZERO;
    let mut total_depth_rm = Vec2::ZERO;

    let mut i = 0;
    while i < steps {
        let p = origin + direction * (depth * i as f32);
        let d_rm = densities_rm(p) * depth;
        total_depth_rm += d_rm;

        // Calculate optical depth
        let depth_rm_sum =
            total_depth_rm + scatter_depth_int(p, sundir, escape(p, sundir, ATMOSPHERE_RADIUS));

        // Calculate exponent part of both integrals
        let a =
            (-RAY_EFFECTIVE_COEFF * depth_rm_sum.x - MIE_EFFECTIVE_COEFF * depth_rm_sum.y).exp();

        i_r += a * d_rm.x;
        i_m += a * d_rm.y;
        i += 1;
    }

    (i_r, i_m)
}

pub fn scatter(sundir: Vec4, origin: Vec3, direction: Vec3) -> Vec3 {
    let (i_r, i_m) = scatter_in(
        origin,
        direction,
        escape(origin, direction, ATMOSPHERE_RADIUS),
        12,
        sundir.xyz(),
    );

    let mu = direction.dot(sundir.xyz());
    let res = sundir.w
        * (1. + mu * mu)
        * (
            // 3/16pi = 0.597
            i_r * RAY_EFFECTIVE_COEFF * 0.0597
                + i_m * MIE_SCATTER_COEFF * 0.0196 / (1.58 - 1.52 * mu).powf(1.5)
        );

    return util::mask_nan(Vec3::new(res.x.sqrt(), res.y.sqrt(), res.z.sqrt())).powf(2.2); // gamma -> linear since we render in linear
}
