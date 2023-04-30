const KERNEL: &[u8] = include_bytes!(env!("compute.spv"));

lazy_static::lazy_static! {
    pub static ref FW: gpgpu::Framework = gpgpu::Framework::default();
}

use glam::{UVec2, UVec4, Vec4};
use gnuplot::{AutoOption, AxesCommon, Coordinate::Graph, Figure, PlotOption::{Caption, self}};
use gpgpu::{
    BufOps, DescriptorSet, GpuBuffer, GpuBufferUsage, GpuUniformBuffer, Kernel, Program, Shader,
};
use russimp::scene::{PostProcess::*, Scene};
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
                PreTransformVertices,
                GenerateUVCoords,
                TransformUVCoords,
            ],
        )
        .unwrap();

        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let mut normals = Vec::new();

        for (idx, mesh) in blend.meshes.iter().enumerate() {
            for v in &mesh.vertices {
                vertices.push(Vec4::new(v.x, v.z, v.y, 0.0));
            }
            for f in &mesh.faces {
                assert_eq!(f.0.len(), 3);
                indices.push(UVec4::new(f.0[0], f.0[2], f.0[1], idx as u32));
            }
            for n in &mesh.normals {
                normals.push(Vec4::new(n.x, n.z, n.y, 0.0));
            }
        }

        // time the build:
        let now = std::time::Instant::now();
        let bvh = BVHBuilder::new(&vertices, &indices).sah_samples(128).build();
        println!("BVH build time: {:?}", now.elapsed());
        println!("BVH node count: {}", bvh.nodes.len());

        /*
        let mut fg = Figure::new();
        let mut ax = fg
            .axes3d()
            .set_title("A plot", &[])
            .set_x_label("x", &[])
            .set_y_label("y", &[])
            .set_z_label("z", &[]);

        for node in bvh.nodes.iter().filter(|x| true) {
            let min = node.aabb_min();
            let max = node.aabb_max();

            let points = vec![
                (min.x, min.y, min.z),
                (min.x, min.y, max.z),
                (min.x, max.y, max.z),
                (min.x, min.y, max.z),
                (max.x, min.y, max.z),
                (max.x, max.y, max.z),
                (max.x, min.y, max.z),
                (max.x, min.y, min.z),
                (max.x, max.y, min.z),
                (max.x, min.y, min.z),
                (min.x, min.y, min.z),
                (min.x, max.y, min.z),
                (min.x, max.y, max.z),
                (max.x, max.y, max.z),
                (max.x, max.y, min.z),
                (min.x, max.y, min.z),
            ];
            // draw a wireframe cube out of lines:
            ax.lines(
                points.iter().map(|(x, y, z)| *x),
                points.iter().map(|(x, y, z)| *y),
                points.iter().map(|(x, y, z)| *z),
                &[],
            );
        }

        for tri in indices {
            let v0 = vertices[tri.x as usize];
            let v1 = vertices[tri.y as usize];
            let v2 = vertices[tri.z as usize];
            let points = vec![
                (v0.x, v0.y, v0.z),
                (v1.x, v1.y, v1.z),
                (v2.x, v2.y, v2.z),
                (v0.x, v0.y, v0.z),
            ];
            ax.lines(
                points.iter().map(|(x, y, z)| *x),
                points.iter().map(|(x, y, z)| *y),
                points.iter().map(|(x, y, z)| *z),
                &[PlotOption::Color("#FF0000")],
            );
        }

        fg.show().unwrap();
        std::process::exit(0);*/

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
