const KERNEL: &[u8] = include_bytes!(env!("kernels.spv"));
const BLUE_BYTES: &'static [u8] = include_bytes!("resources/bluenoise.png");
lazy_static::lazy_static! {
    pub static ref FW: gpgpu2::GpuContext = gpgpu2::GpuContext::default();
    pub static ref BLUE_TEXTURE: RgbaImage = Reader::new(Cursor::new(BLUE_BYTES)).with_guessed_format().unwrap().decode().unwrap().into_rgba8();
}

use glam::{UVec2, Vec4};
use image::{RgbaImage, io::Reader};
pub use shared_structs::TracingConfig;
use std::{sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    Arc,
}, io::Cursor};
use parking_lot::RwLock;

use crate::{asset::World, gpgpu2::{self, GPU_BUFFER_USAGES}};

struct PathTracingKernel<'fw>(gpgpu2::GpuKernel<'fw>);

impl<'fw> PathTracingKernel<'fw> {
    fn new(
        config_buffer: &gpgpu2::GpuBuffer<'fw, TracingConfig>,
        rng_buffer: &gpgpu2::GpuBuffer<'fw, UVec2>,
        output_buffer: &gpgpu2::GpuBuffer<'fw, Vec4>,
        world: &World<'fw>,
    ) -> Self {
        let sampler = gpgpu2::GpuSampler::new(&FW, wgpu::AddressMode::ClampToEdge, wgpu::FilterMode::Linear);
        let kernel = gpgpu2::GpuKernelBuilder::new(&FW, KERNEL, "main_material")
            .bind_uniform_buffer(config_buffer)
            .bind_buffer(rng_buffer, true)
            .bind_buffer(output_buffer, true)
            .bind_buffer(&world.per_vertex_buffer, false)
            .bind_buffer(&world.index_buffer, false)
            .bind_buffer(&world.bvh.nodes_buffer, false)
            .bind_buffer(&world.material_data_buffer, false)
            .bind_buffer(&world.light_pick_buffer, false)
            .bind_sampler(&sampler)
            .bind_image(&world.atlas)
            .build();
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
    scene_path: &str,
    framebuffer: Arc<RwLock<Vec<f32>>>,
    running: Arc<AtomicBool>,
    samples: Arc<AtomicU32>,
    #[allow(unused_variables)] denoise: Arc<AtomicBool>,
    sync_rate: Arc<AtomicU32>,
    use_blue_noise: Arc<AtomicBool>,
    interacting: Arc<AtomicBool>,
    dirty: Arc<AtomicBool>,
    config: Arc<RwLock<TracingConfig>>,
) {
    let world = World::from_path(scene_path);

    let screen_width = config.read().width;
    let screen_height = config.read().height;
    let pixel_count = (screen_width * screen_height) as usize;
    let mut rng = rand::thread_rng();
    let mut rng_data_blue: Vec<UVec2> = vec![UVec2::ZERO; pixel_count];
    let mut rng_data_uniform: Vec<UVec2> = vec![UVec2::ZERO; pixel_count];
    for y in 0..screen_height {
        for x in 0..screen_width {
            let pixel_index = (y * screen_width + x) as usize;
            let pixel = BLUE_TEXTURE.get_pixel(x as u32 % BLUE_TEXTURE.width(), y as u32 % BLUE_TEXTURE.height())[0] as f32 / 255.0;
            rng_data_blue[pixel_index].x = 0;
            rng_data_blue[pixel_index].y = (pixel * 4294967295.0) as u32;
            rng_data_uniform[pixel_index].x = rand::Rng::gen(&mut rng);
        }
    }

    let pixel_count = (screen_width * screen_height) as u64;
    let config_buffer = gpgpu2::GpuBuffer::new(&FW, &[config.read().clone()], gpgpu2::GPU_UNIFORM_USAGES);
    let rng_buffer = gpgpu2::GpuBuffer::new(&FW, if use_blue_noise.load(Ordering::Relaxed) { &rng_data_blue } else { &rng_data_uniform }, gpgpu2::GPU_BUFFER_USAGES);
    let output_buffer = gpgpu2::GpuBuffer::new(&FW, &vec![Vec4::ZERO; pixel_count as usize], gpgpu2::GPU_BUFFER_USAGES);

    let mut image_buffer_raw: Vec<Vec4> = vec![Vec4::ZERO; pixel_count as usize];
    let mut image_buffer: Vec<f32> = vec![0.0; pixel_count as usize * 3];

    let rt = PathTracingKernel::new(&config_buffer, &rng_buffer, &output_buffer, &world);

    while running.load(Ordering::Relaxed) {
        // Dispatch
        let interacting = interacting.load(Ordering::Relaxed) || dirty.load(Ordering::Relaxed);
        let sync_rate = if interacting { 1 } else { sync_rate.load(Ordering::Relaxed) };
        for _ in 0..sync_rate {
            rt.0.enqueue(screen_width.div_ceil(8), screen_height.div_ceil(8), 1);
        }
        samples.fetch_add(sync_rate, Ordering::Relaxed);

        // Readback from GPU
        output_buffer.read(&mut image_buffer_raw);
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
            dirty.store(false, Ordering::Relaxed);
            samples.store(0, Ordering::Relaxed);
            config_buffer.write(&[config.read().clone()]);
            output_buffer.write(&vec![Vec4::ZERO; pixel_count as usize]);
            rng_buffer.write(if use_blue_noise.load(Ordering::Relaxed) { &rng_data_blue } else { &rng_data_uniform });
        }
    }
}
