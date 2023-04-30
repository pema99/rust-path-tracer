const KERNEL: &[u8] = include_bytes!(env!("compute.spv"));

lazy_static::lazy_static! {
    pub static ref FW: gpgpu::Framework = gpgpu::Framework::default();
}

use glam::{UVec2, UVec4, Vec4, Mat4};
use gpgpu::{
    BufOps, DescriptorSet, GpuBuffer, GpuBufferUsage, GpuUniformBuffer, Kernel, Program, Shader,
};
use russimp::{scene::{PostProcess::*, Scene}, node::Node};
pub use shared_structs::TracingConfig;
use std::sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    Arc, Mutex,
};

use crate::bvh::{BVH, BVHBuilder};

struct PathTracingKernel<'fw>(Kernel<'fw>);

impl<'fw> PathTracingKernel<'fw> {
    fn new(
        config_buffer: &GpuUniformBuffer<'fw, TracingConfig>,
        rng_buffer: &GpuBuffer<'fw, UVec2>,
        output_buffer: &GpuBuffer<'fw, Vec4>,
        world: &World<'fw>,
    ) -> Self {
        let shader = Shader::from_spirv_bytes(&FW, KERNEL, Some("compute"));
        let bindings = DescriptorSet::default()
            .bind_uniform_buffer(config_buffer)
            .bind_buffer(rng_buffer, GpuBufferUsage::ReadWrite)
            .bind_buffer(output_buffer, GpuBufferUsage::ReadWrite)
            .bind_buffer(&world.vertex_buffer, GpuBufferUsage::ReadOnly)
            .bind_buffer(&world.index_buffer, GpuBufferUsage::ReadOnly)
            .bind_buffer(&world.normal_buffer, GpuBufferUsage::ReadOnly)
            .bind_buffer(&world.bvh.nodes_buffer, GpuBufferUsage::ReadOnly)
            .bind_buffer(&world.bvh.indirect_indices_buffer, GpuBufferUsage::ReadOnly);
        let program = Program::new(&shader, "main_material").add_descriptor_set(bindings);
        let kernel = Kernel::new(&FW, program);

        Self(kernel)
    }
}

#[cfg(feature = "oidn")]
fn denoise_image(config: &TracingConfig, input: &mut [f32]) {
    let device = oidn::Device::new();
    oidn::RayTracing::new(&device)
        .srgb(true)
        .image_dimensions(config.width as usize, config.height as usize)
        .filter_in_place(input)
        .expect("Filter config error!");
}

struct World<'fw> {
    vertex_buffer: GpuBuffer<'fw, Vec4>,
    index_buffer: GpuBuffer<'fw, UVec4>,
    normal_buffer: GpuBuffer<'fw, Vec4>,
    bvh: BVH<'fw>,
}

impl<'fw> World<'fw> {
    fn from_path(path: &str) -> Self {
        let blend = Scene::from_file(
            path,
            vec![
                JoinIdenticalVertices,
                Triangulate,
                SortByPrimitiveType,
                GenerateSmoothNormals,
                GenerateUVCoords,
                TransformUVCoords,
            ],
        )
        .unwrap();

        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut normals = Vec::new();

        fn walk_node_graph(scene: &Scene, node: &Node, trs: Mat4, vertices: &mut Vec<Vec4>, indices: &mut Vec<UVec4>, normals: &mut Vec<Vec4>) {
            let node_trs = Mat4::from_cols_array_2d(&[
                [node.transformation.a1, node.transformation.b1, node.transformation.c1, node.transformation.d1],
                [node.transformation.a2, node.transformation.b2, node.transformation.c2, node.transformation.d2],
                [node.transformation.a3, node.transformation.b3, node.transformation.c3, node.transformation.d3],
                [node.transformation.a4, node.transformation.b4, node.transformation.c4, node.transformation.d4],
            ]);
            let new_trs = node_trs * trs;

            for mesh_idx in node.meshes.iter() {
                let mesh = &scene.meshes[*mesh_idx as usize];
                let triangle_offset = vertices.len() as u32;
                for v in &mesh.vertices {
                    let vert = new_trs.mul_vec4(Vec4::new(v.x, v.y, v.z, 1.0));
                    vertices.push(Vec4::new(vert.x, vert.z, vert.y, 1.0));
                }
                for f in &mesh.faces {
                    assert_eq!(f.0.len(), 3);
                    indices.push(UVec4::new(triangle_offset + f.0[0], triangle_offset + f.0[2], triangle_offset + f.0[1], *mesh_idx as u32));
                }
                for n in &mesh.normals {
                    let norm = new_trs.mul_vec4(Vec4::new(n.x, n.y, n.z, 0.0)).normalize();
                    normals.push(Vec4::new(norm.x, norm.z, norm.y, 0.0));
                }
            }

            for child in node.children.borrow().iter() {
                walk_node_graph(scene, &child, new_trs, vertices, indices, normals);
            }
        }

        if let Some(root) = blend.root.as_ref() {
            walk_node_graph(&blend, root, Mat4::IDENTITY, &mut vertices, &mut indices, &mut normals);
        }

        // time the build:
        let now = std::time::Instant::now();
        let bvh = BVHBuilder::new(&vertices, &indices).sah_samples(128).build();
        println!("BVH build time: {:?}", now.elapsed());
        println!("BVH node count: {}", bvh.nodes.len());

        let vertex_buffer = GpuBuffer::from_slice(&FW, &vertices);
        let index_buffer = GpuBuffer::from_slice(&FW, &indices);
        let normal_buffer = GpuBuffer::from_slice(&FW, &normals);
        Self {
            vertex_buffer,
            index_buffer,
            normal_buffer,
            bvh,
        }
    }
}

pub fn trace(
    framebuffer: Arc<Mutex<Vec<f32>>>,
    running: Arc<AtomicBool>,
    samples: Arc<AtomicU32>,
    #[allow(unused_variables)] denoise: Arc<AtomicBool>,
    sync_rate: Arc<AtomicU32>,
    config: TracingConfig,
) {
    let world = World::from_path("scene.blend");

    let mut rng = rand::thread_rng();
    let mut rng_data = Vec::new();
    for _ in 0..(config.width * config.height) {
        rng_data.push(UVec2::new(
            rand::Rng::gen(&mut rng),
            rand::Rng::gen(&mut rng),
        ));
    }

    let pixel_count = (config.width * config.height) as u64;
    let config_buffer = GpuUniformBuffer::from_slice(&FW, &[config]);
    let rng_buffer = GpuBuffer::from_slice(&FW, &rng_data);
    let output_buffer = GpuBuffer::with_capacity(&FW, pixel_count);

    let mut image_buffer_raw: Vec<Vec4> = vec![Vec4::ZERO; pixel_count as usize];
    let mut image_buffer: Vec<f32> = vec![0.0; pixel_count as usize * 3];

    let rt = PathTracingKernel::new(&config_buffer, &rng_buffer, &output_buffer, &world);

    while running.load(Ordering::Relaxed) {
        let sync_rate = sync_rate.load(Ordering::Relaxed);
        for _ in 0..sync_rate {
            rt.0.enqueue(config.width.div_ceil(8), config.height.div_ceil(8), 1);
        }
        samples.fetch_add(sync_rate, Ordering::Relaxed);

        output_buffer.read_blocking(&mut image_buffer_raw).unwrap();
        let sample_count = samples.load(Ordering::Relaxed) as f32;
        for (i, col) in image_buffer_raw.iter().enumerate() {
            image_buffer[i * 3 + 0] = col.x / sample_count;
            image_buffer[i * 3 + 1] = col.y / sample_count;
            image_buffer[i * 3 + 2] = col.z / sample_count;
        }

        #[cfg(feature = "oidn")]
        if denoise.load(Ordering::Relaxed) {
            denoise_image(&config, &mut image_buffer);
        }

        // Readback
        match framebuffer.lock() {
            Ok(mut fb) => fb.copy_from_slice(image_buffer.as_slice()),
            Err(_) => {},
        }
    }
}
