use bytemuck::{Pod, Zeroable};
use glam::{UVec4, Vec3, Vec4, Vec4Swizzles};

// TODO: Use triangle buffer directly instead of 2 indirections

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Default)]
pub struct BVHNode {
    pub aabb_min: Vec3,
    pub aabb_max: Vec3,
    pub left_node_or_triangle_index: u32, // left_node if triangle_count is 0, triangle_index if tri_count is 1
    pub triangle_count: u32,
}

impl BVHNode {
    fn update_aabb(&mut self, vertices: &[Vec4], indices: &[UVec4], indirect_indices: &[u32]) {
        self.aabb_min = Vec3::splat(f32::INFINITY);
        self.aabb_max = Vec3::splat(f32::NEG_INFINITY);

        for i in 0..self.triangle_count {
            let indirect_index = indirect_indices[(self.left_node_or_triangle_index + i) as usize];
            let index = indices[indirect_index as usize];
            let v0 = vertices[index.x as usize].xyz();
            let v1 = vertices[index.y as usize].xyz();
            let v2 = vertices[index.z as usize].xyz();

            self.aabb_min = self.aabb_min.min(v0.min(v1).min(v2));
            self.aabb_max = self.aabb_max.max(v0.max(v1).max(v2));
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.triangle_count > 0
    }
}

pub struct BVH {
    pub nodes: Vec<BVHNode>,
    pub indirect_indices: Vec<u32>,
}

impl BVH {
    pub fn build(vertices: &[Vec4], indices: &[UVec4]) -> BVH {
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

        let mut root = &mut nodes[0];
        root.left_node_or_triangle_index = 0;
        root.triangle_count = indices.len() as u32;
        root.update_aabb(vertices, indices, &indirect_indices);

        let mut stack = vec![0];
        while !stack.is_empty() {
            // get the next root node
            let node_idx = stack.pop().unwrap();
            let mut node = &mut nodes[node_idx];
            if node.triangle_count <= 2 {
                continue;
            }

            // calculate the max axis
            let extent = node.aabb_max - node.aabb_min;
            let mut axis = 0;
            if extent.y > extent.x {
                axis = 1;
            }
            if extent.z > extent[axis] {
                axis = 2;
            }

            // split along the axis
            let split = node.aabb_min[axis] + extent[axis] * 0.5; // this is shitty

            // partition the triangles
            let mut a = node.left_node_or_triangle_index;
            let mut b = a + node.triangle_count - 1;
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
            let left_count = a - node.left_node_or_triangle_index;
            if left_count == 0 || left_count == node.triangle_count {
                continue;
            }

            // create children
            let prev_triangle_idx = node.left_node_or_triangle_index;
            let prev_triangle_count = node.triangle_count;
            let left_idx = node_count;
            let right_idx = node_count + 1;
            node_count += 2;
            node.left_node_or_triangle_index = left_idx as u32;
            node.triangle_count = 0;
            nodes[left_idx].left_node_or_triangle_index = prev_triangle_idx;
            nodes[left_idx].triangle_count = left_count;
            nodes[right_idx].left_node_or_triangle_index = a;
            nodes[right_idx].triangle_count = prev_triangle_count - left_count;
            nodes[left_idx].update_aabb(vertices, indices, &indirect_indices);
            nodes[right_idx].update_aabb(vertices, indices, &indirect_indices);

            // push children onto the stack
            stack.push(right_idx);
            stack.push(left_idx);
        }

        nodes.truncate(node_count - 1);
        Self {
            nodes,
            indirect_indices,
        }
    }
}