use glam::{UVec4, Vec3, Vec4, Vec4Swizzles};
use gpgpu::{GpuBuffer, BufOps};
use shared_structs::{BVHNode};

use crate::trace::FW;

// TODO: Use triangle buffer directly instead of 2 indirections

trait BVHNodeExtensions {
    fn encapsulate(&mut self, point: &Vec3);
    fn area(&self) -> f32;
}

impl BVHNodeExtensions for BVHNode {
    fn encapsulate(&mut self, point: &Vec3) {
        self.set_aabb_min(&self.aabb_min().min(*point));
        self.set_aabb_max(&self.aabb_max().max(*point));
    }

    fn area(&self) -> f32 {
        let extent = self.aabb_max() - self.aabb_min();
        extent.x * extent.y + extent.y * extent.z + extent.z * extent.x
    }
}

pub struct BVH<'fw> {
    pub nodes: Vec<BVHNode>,
    pub indirect_indices: Vec<u32>,
    pub nodes_buffer: GpuBuffer<'fw, BVHNode>,
    pub indirect_indices_buffer: GpuBuffer<'fw, u32>,
}

pub struct BVHBuilder<'a> {
    sah_samples: usize,
    vertices: &'a [Vec4],
    indices: &'a [UVec4],
    centroids: Vec<Vec3>,
    indirect_indices: Vec<u32>,
    nodes: Vec<BVHNode>,
}

impl<'a> BVHBuilder<'a> {
    pub fn new(vertices: &'a [Vec4], indices: &'a [UVec4]) -> Self {
        let indirect_indices: Vec<u32> = (0..indices.len() as u32).collect();
        let centroids = indices
            .iter()
            .map(|ind| {
                let v0 = vertices[ind.x as usize].xyz();
                let v1 = vertices[ind.y as usize].xyz();
                let v2 = vertices[ind.z as usize].xyz();
                (v0 + v1 + v2) / 3.0
            })
            .collect::<Vec<_>>();
        let nodes = vec![BVHNode::default(); indices.len() * 2 - 1];

        Self {
            sah_samples: 128,
            vertices,
            indices,
            centroids,
            indirect_indices,
            nodes,
        }
    }

    pub fn sah_samples(mut self, sah_samples: usize) -> Self {
        self.sah_samples = sah_samples;
        self
    }

    fn update_node_aabb<'b>(&mut self, node_idx: usize) {
        let node = &mut self.nodes[node_idx];
        let mut aabb_min = Vec3::splat(f32::INFINITY);
        let mut aabb_max = Vec3::splat(f32::NEG_INFINITY);

        for i in 0..node.triangle_count() {
            let indirect_index = self.indirect_indices[(node.first_triangle_index() + i) as usize];
            let index = self.indices[indirect_index as usize];
            let v0 = self.vertices[index.x as usize].xyz();
            let v1 = self.vertices[index.y as usize].xyz();
            let v2 = self.vertices[index.z as usize].xyz();

            aabb_min = aabb_min.min(v0.min(v1).min(v2));
            aabb_max = aabb_max.max(v0.max(v1).max(v2));
        }

        node.set_aabb_min(&aabb_min);
        node.set_aabb_max(&aabb_max);
    }

    fn calculate_surface_area_heuristic(&self, node: &BVHNode, axis: usize, split: f32) -> f32 {
        let mut left_box = BVHNode::default();
        let mut right_box = BVHNode::default();
        let mut left_tri_count = 0;
        let mut right_tri_count = 0;
    
        for i in 0..node.triangle_count() {
            let indirect_index = self.indirect_indices[(node.left_node_index() + i) as usize];
            let index = self.indices[indirect_index as usize];
            let v0 = self.vertices[index.x as usize].xyz();
            let v1 = self.vertices[index.y as usize].xyz();
            let v2 = self.vertices[index.z as usize].xyz();
            let centroid = self.centroids[indirect_index as usize];
    
            if centroid[axis] < split {
                left_box.encapsulate(&v0);
                left_box.encapsulate(&v1);
                left_box.encapsulate(&v2);
                left_tri_count += 1;
            } else {
                right_box.encapsulate(&v0);
                right_box.encapsulate(&v1);
                right_box.encapsulate(&v2);
                right_tri_count += 1;
            }
        }
        
        let result = left_box.area() * left_tri_count as f32 + right_box.area() * right_tri_count as f32;
        if result > 0.0 {
            result
        } else {
            f32::INFINITY
        }
    }

    pub fn build<'b>(&mut self) -> BVH<'b> {
        let mut node_count = 1;

        let root = &mut self.nodes[0];
        root.set_first_triangle_index(0);
        root.set_triangle_count(self.indices.len() as u32);
        self.update_node_aabb(0);

        let mut stack = vec![0];
        while !stack.is_empty() {
            // get the next root node
            let node_idx = stack.pop().unwrap();
            let node = &self.nodes[node_idx];

            // calculate the best split (SAH)
            let mut best_axis = 0;
            let mut best_split = 0.0;
            let mut best_cost = f32::INFINITY;
            for axis in 0..3 {
                // find bounds of centroids in this node
                let mut bounds_min = f32::INFINITY;
                let mut bounds_max = f32::NEG_INFINITY;
                for i in 0..node.triangle_count() {
                    let indirect_index = self.indirect_indices[(node.first_triangle_index() + i) as usize];
                    let centroid = self.centroids[indirect_index as usize];
                    bounds_min = bounds_min.min(centroid[axis]);
                    bounds_max = bounds_max.max(centroid[axis]);
                }

                // skip if completely flat
                if bounds_min == bounds_max {
                    continue;
                }

                // check splits uniformly along the axis
                let scale = (bounds_max - bounds_min) / self.sah_samples as f32;
                for i in 1..self.sah_samples {
                    let candidate = bounds_min + scale * i as f32;
                    let cost = self.calculate_surface_area_heuristic(node, axis, candidate);
                    if cost < best_cost {
                        best_axis = axis;
                        best_split = candidate;
                        best_cost = cost;
                    }
                }
            }

            // if the parent node is cheaper, don't split
            let parent_cost = node.area() * node.triangle_count() as f32;
            if parent_cost <= best_cost {
                continue;
            }

            // partition the triangles
            let mut a = node.first_triangle_index();
            let mut b = a + node.triangle_count() - 1;
            while a <= b {
                let centroid = self.centroids[self.indirect_indices[a as usize] as usize][best_axis];
                if centroid < best_split {
                    a += 1;
                } else {
                    self.indirect_indices.swap(a as usize, b as usize);
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
            self.nodes[node_idx].set_left_node_index(left_idx as u32);
            self.nodes[node_idx].set_triangle_count(0);
            self.nodes[left_idx].set_first_triangle_index(prev_triangle_idx);
            self.nodes[left_idx].set_triangle_count(left_count);
            self.nodes[right_idx].set_first_triangle_index(a);
            self.nodes[right_idx].set_triangle_count(prev_triangle_count - left_count);
            self.update_node_aabb(left_idx);
            self.update_node_aabb(right_idx);

            // push children onto the stack
            stack.push(right_idx);
            stack.push(left_idx);
        }

        self.nodes.truncate(node_count);
        let nodes_buffer = GpuBuffer::from_slice(&FW, &self.nodes);
        let indirect_indices_buffer = GpuBuffer::from_slice(&FW, &self.indirect_indices);
        BVH {
            nodes: self.nodes.clone(),
            indirect_indices: self.indirect_indices.clone(),
            nodes_buffer,
            indirect_indices_buffer,
        }
    }
}