use glam::{UVec4, Vec3, Vec4, Vec4Swizzles};
use gpgpu::{GpuBuffer, BufOps};
use shared_structs::{BVHNode};

use crate::trace::FW;

// TODO: Use triangle buffer directly instead of 2 indirections

trait BVHNodeExtensions {
    fn update_aabb(&mut self, vertices: &[Vec4], indices: &[UVec4], indirect_indices: &[u32]);
}

impl BVHNodeExtensions for BVHNode {
    fn update_aabb(&mut self, vertices: &[Vec4], indices: &[UVec4], indirect_indices: &[u32]) {
        let mut aabb_min = Vec3::splat(f32::INFINITY);
        let mut aabb_max = Vec3::splat(f32::NEG_INFINITY);

        for i in 0..self.triangle_count() {
            let indirect_index = indirect_indices[(self.first_triangle_index() + i) as usize];
            let index = indices[indirect_index as usize];
            let v0 = vertices[index.x as usize].xyz();
            let v1 = vertices[index.y as usize].xyz();
            let v2 = vertices[index.z as usize].xyz();

            aabb_min = aabb_min.min(v0.min(v1).min(v2));
            aabb_max = aabb_max.max(v0.max(v1).max(v2));
        }

        self.set_aabb_min(&aabb_min);
        self.set_aabb_max(&aabb_max);
    }
}

pub struct BVH<'fw> {
    pub nodes: Vec<BVHNode>,
    pub indirect_indices: Vec<u32>,
    pub nodes_buffer: GpuBuffer<'fw, BVHNode>,
    pub indirect_indices_buffer: GpuBuffer<'fw, u32>,
}

impl<'fw> BVH<'fw> {
    pub fn build(vertices: &[Vec4], indices: &[UVec4]) -> BVH<'fw> {
        let mut indirect_indices: Vec<u32> = (0..indices.len() as u32).collect();
        let centroids = indices
            .iter()
            .map(|ind| {
                let v0 = vertices[ind.x as usize].xyz();
                let v1 = vertices[ind.y as usize].xyz();
                let v2 = vertices[ind.z as usize].xyz();
                (v0 + v1 + v2) / 3.0
            })
            .collect::<Vec<_>>();

        let mut nodes = vec![BVHNode::default(); indices.len() * 2 - 1];
        let mut node_count = 1;

        let root = &mut nodes[0];
        root.set_first_triangle_index(0);
        root.set_triangle_count(indices.len() as u32);
        root.update_aabb(vertices, indices, &indirect_indices);

        let mut stack = vec![0];
        while !stack.is_empty() {
            // get the next root node
            let node_idx = stack.pop().unwrap();
            let node = &mut nodes[node_idx];
            if node.triangle_count() <= 2 {
                continue;
            }

            // calculate the max axis
            let extent = node.aabb_max() - node.aabb_min();
            let mut axis = 0;
            if extent.y > extent.x {
                axis = 1;
            }
            if extent.z > extent[axis] {
                axis = 2;
            }

            // split along the axis
            let split = node.aabb_min()[axis] + extent[axis] * 0.5; // this is shitty

            // partition the triangles
            let mut a = node.first_triangle_index();
            let mut b = a + node.triangle_count() - 1;
            while a <= b {
                let centroid = centroids[indirect_indices[a as usize] as usize][axis];
                if centroid < split {
                    a += 1;
                } else {
                    indirect_indices.swap(a as usize, b as usize);
                    b -= 1;
                }
            }

            // if either side is empty (no split), then we're done
            let left_count = a - node.first_triangle_index();
            if left_count == 0 || left_count == node.triangle_count() {
                continue;
            }

            // create children
            let prev_triangle_idx = node.first_triangle_index();
            let prev_triangle_count = node.triangle_count();
            let left_idx = node_count;
            let right_idx = node_count + 1;
            node_count += 2;
            node.set_left_node_index(left_idx as u32);
            node.set_triangle_count(0);
            nodes[left_idx].set_first_triangle_index(prev_triangle_idx);
            nodes[left_idx].set_triangle_count(left_count);
            nodes[right_idx].set_first_triangle_index(a);
            nodes[right_idx].set_triangle_count(prev_triangle_count - left_count);
            nodes[left_idx].update_aabb(vertices, indices, &indirect_indices);
            nodes[right_idx].update_aabb(vertices, indices, &indirect_indices);

            // push children onto the stack
            stack.push(right_idx);
            stack.push(left_idx);
        }

        nodes.truncate(node_count);
        let nodes_buffer = GpuBuffer::from_slice(&FW, &nodes);
        let indirect_indices_buffer = GpuBuffer::from_slice(&FW, &indirect_indices);
        Self {
            nodes,
            indirect_indices,
            nodes_buffer,
            indirect_indices_buffer,
        }
    }
}
