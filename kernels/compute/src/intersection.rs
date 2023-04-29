use shared_structs::{BVHNode};
#[allow(unused_imports)]
use spirv_std::num_traits::Float;
use spirv_std::glam::{UVec4, Vec4, Vec3, Vec4Swizzles};

use crate::vec::FixedVec;

// Adapted from raytri.c
fn muller_trumbore(ro: Vec3, rd: Vec3, a: Vec3, b: Vec3, c: Vec3, out_t: &mut f32) -> bool
{
    *out_t = 0.0;

    let edge1 = b - a;
    let edge2 = c - a;
 
    // begin calculating determinant - also used to calculate U parameter
    let pv = rd.cross(edge2);
 
    // if determinant is near zero, ray lies in plane of triangle
    let det = edge1.dot(pv);
 
    if det.abs() < 1e-6 {
        return false;
    }
 
    let inv_det = 1.0 / det;
 
    // calculate distance from vert0 to ray origin
    let tv = ro - a;
 
    // calculate U parameter and test bounds
    let u = tv.dot(pv) * inv_det;
    if u < 0.0 || u > 1.0 {
        return false;
    }
 
    // prepare to test V parameter
    let qv = tv.cross(edge1);
 
    // calculate V parameter and test bounds
    let v = rd.dot(qv) * inv_det;
    if v < 0.0 || u + v > 1.0 {
        return false;
    }
 
    let t = edge2.dot(qv) * inv_det;
    if t < 0.0 {
        return false;
    }
    *out_t = t;
 
    return true;
}

pub struct TraceResult {
    pub t: f32,
    pub triangle: UVec4,
    pub hit: bool,
}

impl Default for TraceResult {
    fn default() -> Self {
        Self {
            t: 1000000.0,
            triangle: UVec4::splat(0),
            hit: false,
        }
    }
}

#[allow(dead_code)]
fn intersect_slow_as_shit(
    vertex_buffer: &[Vec4],
    index_buffer: &[UVec4],
    ro: Vec3,
    rd: Vec3
) -> TraceResult {
    let mut result = TraceResult::default();
    for i in 0..index_buffer.len() {
        let triangle = index_buffer[i];
        let a = vertex_buffer[triangle.x as usize].xyz();
        let b = vertex_buffer[triangle.y as usize].xyz();
        let c = vertex_buffer[triangle.z as usize].xyz();

        let mut t = 0.0;
        if muller_trumbore(ro, rd, a, b, c, &mut t) && t > 0.001 && t < result.t {
            result.t = result.t.min(t);
            result.triangle = triangle;
            result.hit = true;
        }
    }
    result
}

fn intersect_aabb(aabb_min: Vec3, aabb_max: Vec3, ro: Vec3, rd: Vec3) -> bool {
    let tx1 = (aabb_min.x - ro.x) / rd.x; 
    let tx2 = (aabb_max.x - ro.x) / rd.x;
    let mut tmin = tx1.min(tx2);
    let mut tmax = tx1.max(tx2);
    let ty1 = (aabb_min.y - ro.y) / rd.y;
    let ty2 = (aabb_max.y - ro.y) / rd.y;
    tmin = tmin.max(ty1.min(ty2));
    tmax = tmax.min(ty1.max(ty2));
    let tz1 = (aabb_min.z - ro.z) / rd.z;
    let tz2 = (aabb_max.z - ro.z) / rd.z;
    tmin = tmin.max(tz1.min(tz2));
    tmax = tmax.min(tz1.max(tz2));
    if tmax >= tmin && tmax > 0.0 {
        //tmin
        true
    } else { 
        //1e30f
        false
    }
}

pub struct BVHReference<'a> {
    pub nodes: &'a [BVHNode],
    pub indirect_indices: &'a [u32],
}

impl<'a> BVHReference<'a> {
    pub fn intersect(&self, vertex_buffer: &[Vec4], index_buffer: &[UVec4], ro: Vec3, rd: Vec3) -> TraceResult {
        let mut stack = FixedVec::<usize, 16>::new();
        stack.push(0);

        let mut result = TraceResult::default();
        while !stack.is_empty() {
            let node_index = stack.pop().unwrap();
            let node = &self.nodes[node_index];
            if !intersect_aabb(node.aabb_min(), node.aabb_max(), ro, rd) {
                continue;
            }

            if node.is_leaf() {
                for i in 0..node.triangle_count() {
                    let indirect_index = self.indirect_indices[(node.first_triangle_index() + i) as usize];
                    let triangle = index_buffer[indirect_index as usize];
                    let a = vertex_buffer[triangle.x as usize].xyz();
                    let b = vertex_buffer[triangle.y as usize].xyz();
                    let c = vertex_buffer[triangle.z as usize].xyz();
    
                    let mut t = 0.0;
                    if muller_trumbore(ro, rd, a, b, c, &mut t) && t > 0.001 && t < result.t {
                        result.t = result.t.min(t);
                        result.triangle = triangle;
                        result.hit = true;
                    }
                }
            } else {
                stack.push(node.right_node_index() as usize);
                stack.push(node.left_node_index() as usize);
            }
        }

        result
    }
}
