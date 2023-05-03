const KERNEL: &[u8] = include_bytes!(env!("kernels.spv"));

lazy_static::lazy_static! {
    pub static ref FW: gpgpu::Framework = gpgpu::Framework::default();
}

use glam::{UVec2, Vec4};
use gpgpu::{
    BufOps, DescriptorSet, GpuBuffer, GpuBufferUsage, GpuUniformBuffer, Kernel, Program, Shader, Sampler, SamplerWrapMode, SamplerFilterMode
};
pub use shared_structs::TracingConfig;
use std::sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    Arc,
};
use parking_lot::RwLock;

use crate::asset::World;

struct PathTracingKernel<'fw>(Kernel<'fw>);

impl<'fw> PathTracingKernel<'fw> {
    fn new(
        config_buffer: &GpuUniformBuffer<'fw, TracingConfig>,
        rng_buffer: &GpuBuffer<'fw, UVec2>,
        output_buffer: &GpuBuffer<'fw, Vec4>,
        world: &World<'fw>,
    ) -> Self {
        let shader = Shader::from_spirv_bytes(&FW, KERNEL, Some("compute"));
        let sampler = Sampler::new(&FW, SamplerWrapMode::ClampToEdge, SamplerFilterMode::Linear);
        let bindings = DescriptorSet::default()
            .bind_uniform_buffer(config_buffer)
            .bind_buffer(rng_buffer, GpuBufferUsage::ReadWrite)
            .bind_buffer(output_buffer, GpuBufferUsage::ReadWrite)
            .bind_buffer(&world.per_vertex_buffer, GpuBufferUsage::ReadOnly)
            .bind_buffer(&world.index_buffer, GpuBufferUsage::ReadOnly)
            .bind_buffer(&world.bvh.nodes_buffer, GpuBufferUsage::ReadOnly)
            .bind_buffer(&world.bvh.indirect_indices_buffer, GpuBufferUsage::ReadOnly)
            .bind_buffer(&world.material_data_buffer, GpuBufferUsage::ReadOnly)
            .bind_sampler(&sampler)
            .bind_const_image(&world.albedo_atlas);
        let program = Program::new(&shader, "main_material").add_descriptor_set(bindings);
        let kernel = Kernel::new(&FW, program);

        Self(kernel)
    }
}

#[cfg(feature = "oidn")]
fn denoise_image(width: usize, height: usize, input: &mut [f32]) {
    let device = oidn::Device::new();
    oidn::RayTracing::new(&device)
        .srgb(true)
        .image_dimensions(width, height)
        .filter_in_place(input)
        .expect("Filter config error!");
}

pub fn trace(
    framebuffer: Arc<RwLock<Vec<f32>>>,
    running: Arc<AtomicBool>,
    samples: Arc<AtomicU32>,
    #[allow(unused_variables)] denoise: Arc<AtomicBool>,
    sync_rate: Arc<AtomicU32>,
    interacting: Arc<AtomicBool>,
    config: Arc<RwLock<TracingConfig>>,
) {
    let world = World::from_path("scene.glb");

    let screen_width = config.read().width;
    let screen_height = config.read().height;
    let mut rng = rand::thread_rng();
    let mut rng_data = Vec::new();
    for _ in 0..(screen_width * screen_height) {
        rng_data.push(UVec2::new(
            rand::Rng::gen(&mut rng),
            rand::Rng::gen(&mut rng),
        ));
    }

    let pixel_count = (screen_width * screen_height) as u64;
    let config_buffer = GpuUniformBuffer::from_slice(&FW, &[config.read().clone()]);
    let rng_buffer = GpuBuffer::from_slice(&FW, &rng_data);
    let output_buffer = GpuBuffer::with_capacity(&FW, pixel_count);

    let mut image_buffer_raw: Vec<Vec4> = vec![Vec4::ZERO; pixel_count as usize];
    let mut image_buffer: Vec<f32> = vec![0.0; pixel_count as usize * 3];

    let rt = PathTracingKernel::new(&config_buffer, &rng_buffer, &output_buffer, &world);

    while running.load(Ordering::Relaxed) {
        // Dispatch
        let interacting = interacting.load(Ordering::Relaxed);
        let sync_rate = if interacting { 1 } else { sync_rate.load(Ordering::Relaxed) };
        for _ in 0..sync_rate {
            rt.0.enqueue(screen_width.div_ceil(8), screen_height.div_ceil(8), 1);
        }
        samples.fetch_add(sync_rate, Ordering::Relaxed);

        // Readback from GPU
        output_buffer.read_blocking(&mut image_buffer_raw).unwrap();
        let sample_count = samples.load(Ordering::Relaxed) as f32;
        for (i, col) in image_buffer_raw.iter().enumerate() {
            image_buffer[i * 3 + 0] = col.x / sample_count;
            image_buffer[i * 3 + 1] = col.y / sample_count;
            image_buffer[i * 3 + 2] = col.z / sample_count;
        }

        // Denoise
        #[cfg(feature = "oidn")]
        if denoise.load(Ordering::Relaxed) {
            denoise_image(screen_width as usize, screen_height as usize, &mut image_buffer);
        }

        // Push to render thread
        framebuffer.write().copy_from_slice(image_buffer.as_slice());

        // Interaction
        if interacting {
            samples.store(0, Ordering::Relaxed);
            config_buffer.write(&[config.read().clone()]).unwrap();
            output_buffer.write(&vec![Vec4::ZERO; pixel_count as usize]).unwrap();
        }
    }
}
