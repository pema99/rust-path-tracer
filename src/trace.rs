const KERNEL: &[u8] = include_bytes!(env!("kernels.spv"));
const BLUE_BYTES: &[u8] = include_bytes!("resources/bluenoise.png");
lazy_static::lazy_static! {
    pub static ref FW: gpgpu::Framework = make_framework();
    pub static ref BLUE_TEXTURE: RgbaImage = Reader::new(Cursor::new(BLUE_BYTES)).with_guessed_format().unwrap().decode().unwrap().into_rgba8();
}

use glam::{UVec2, Vec4, UVec3};
use gpgpu::{
    BufOps, DescriptorSet, GpuBuffer, GpuBufferUsage, GpuUniformBuffer, Kernel, Program, Shader, Sampler, SamplerWrapMode, SamplerFilterMode, GpuConstImage, primitives::pixels::Rgba32Float
};
use image::{RgbaImage, io::Reader, GenericImageView};
use parking_lot::RwLock;
use pollster::FutureExt;
use shared_structs::CpuImage;
pub use shared_structs::TracingConfig;
use std::{sync::{
    atomic::{Ordering, AtomicBool, AtomicU32},
    Arc,
}, io::Cursor};
use rayon::prelude::*;

use crate::{asset::{World, GpuWorld, dynamic_image_to_cpu_buffer, load_dynamic_image, dynamic_image_to_gpu_image, fallback_gpu_image, fallback_cpu_buffer}};

fn make_framework() -> gpgpu::Framework {
    let backend = wgpu::util::backend_bits_from_env().unwrap_or(wgpu::Backends::PRIMARY);
    let power_preference = wgpu::util::power_preference_from_env()
        .unwrap_or(wgpu::PowerPreference::HighPerformance);
    let instance = wgpu::Instance::new(backend);
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference,
            ..Default::default()
        })
        .block_on()
        .expect("Failed at adapter creation.");
    gpgpu::Framework::new(adapter, std::time::Duration::from_millis(1)).block_on()
}

pub struct TracingState {
    pub framebuffer: RwLock<Vec<f32>>,
    pub running: AtomicBool,
    pub samples: AtomicU32,
    pub denoise: AtomicBool,
    pub sync_rate: AtomicU32,
    pub use_blue_noise: AtomicBool,
    pub interacting: AtomicBool,
    pub dirty: AtomicBool,
    pub config: RwLock<TracingConfig>,
}

impl TracingState {
    pub fn make_view_dependent_state(
        width: u32,
        height: u32,
        config: Option<TracingConfig>,
    ) -> (TracingConfig, Vec<f32>) {
        let config = TracingConfig {
            width,
            height,
            ..config.unwrap_or_default()
        };
        let data_size = width as usize * height as usize * 3;
        let framebuffer = vec![0.0; data_size];
        (config, framebuffer)
    }

    pub fn new (width: u32, height: u32) -> Self {
        let (config, framebuffer) = Self::make_view_dependent_state(width, height, None);
        let config = RwLock::new(config);
        let framebuffer = RwLock::new(framebuffer);
        let running = AtomicBool::new(false);
        let samples = AtomicU32::new(0);
        let denoise = AtomicBool::new(false);
        let sync_rate = AtomicU32::new(32);
        let use_blue_noise = AtomicBool::new(true);
        let interacting = AtomicBool::new(false);
        let dirty = AtomicBool::new(false);
        
        Self {
            framebuffer,
            running,
            samples,
            denoise,
            sync_rate,
            use_blue_noise,
            interacting,
            dirty,
            config,
        }
    }
}

struct PathTracingKernel<'fw>(Kernel<'fw>);

impl<'fw> PathTracingKernel<'fw> {
    fn new(
        config_buffer: &GpuUniformBuffer<'fw, TracingConfig>,
        rng_buffer: &GpuBuffer<'fw, UVec2>,
        output_buffer: &GpuBuffer<'fw, Vec4>,
        world: &GpuWorld<'fw>,
        skybox: &GpuConstImage<'fw, Rgba32Float>,
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
            .bind_buffer(&world.material_data_buffer, GpuBufferUsage::ReadOnly)
            .bind_buffer(&world.light_pick_buffer, GpuBufferUsage::ReadOnly)
            .bind_sampler(&sampler)
            .bind_const_image(&world.atlas)
            .bind_const_image(&skybox);
        let program = Program::new(&shader, "trace_kernel").add_descriptor_set(bindings);
        let kernel = Kernel::new(&FW, program);

        Self(kernel)
    }
}

#[cfg(feature = "oidn")]
fn denoise_image(width: usize, height: usize, input: &mut [f32]) {
    let device = oidn::Device::new();
    oidn::RayTracing::new(&device)
        .hdr(true)
        .srgb(false)
        .image_dimensions(width, height)
        .filter_in_place(input)
        .expect("Filter config error!");
}

pub fn trace_gpu(
    scene_path: &str,
    skybox_path: Option<&str>,
    state: Arc<TracingState>,
) {
    let Some(world) = World::from_path(scene_path).map(|w| w.into_gpu()) else {
        return;
    };
    let skybox = skybox_path.and_then(load_dynamic_image).map(dynamic_image_to_gpu_image).unwrap_or_else(|| fallback_gpu_image());

    let screen_width = state.config.read().width;
    let screen_height = state.config.read().height;
    let pixel_count = (screen_width * screen_height) as usize;
    let mut rng = rand::thread_rng();
    let mut rng_data_blue: Vec<UVec2> = vec![UVec2::ZERO; pixel_count];
    let mut rng_data_uniform: Vec<UVec2> = vec![UVec2::ZERO; pixel_count];
    for y in 0..screen_height {
        for x in 0..screen_width {
            let pixel_index = (y * screen_width + x) as usize;
            let pixel = BLUE_TEXTURE.get_pixel(x % BLUE_TEXTURE.width(), y % BLUE_TEXTURE.height())[0] as f32 / 255.0;
            rng_data_blue[pixel_index].x = 0;
            rng_data_blue[pixel_index].y = (pixel * 4294967295.0) as u32;
            rng_data_uniform[pixel_index].x = rand::Rng::gen(&mut rng);
        }
    }

    // Restore previous state, if there is any
    let samples_init = state.samples.load(Ordering::Relaxed) as f32;
    let output_buffer_init = state.framebuffer.read().chunks(3).map(|c| Vec4::new(c[0], c[1], c[2], 1.0) * samples_init).collect::<Vec<_>>();

    // Setup tracing state
    let pixel_count = (screen_width * screen_height) as u64;
    let config_buffer = GpuUniformBuffer::from_slice(&FW, &[*state.config.read()]);
    let rng_buffer = GpuBuffer::from_slice(&FW, if state.use_blue_noise.load(Ordering::Relaxed) { &rng_data_blue } else { &rng_data_uniform });
    let output_buffer = GpuBuffer::from_slice(&FW, &output_buffer_init);

    let mut image_buffer_raw: Vec<Vec4> = vec![Vec4::ZERO; pixel_count as usize];
    let mut image_buffer: Vec<f32> = vec![0.0; pixel_count as usize * 3];

    let rt = PathTracingKernel::new(&config_buffer, &rng_buffer, &output_buffer, &world, &skybox);

    while state.running.load(Ordering::Relaxed) {
        // Dispatch
        let sync_rate = state.sync_rate.load(Ordering::Relaxed);
        let mut flush = false;
        let mut finished_samples = 0;
        for _ in 0..sync_rate {
            rt.0.enqueue(screen_width.div_ceil(8), screen_height.div_ceil(8), 1);
            FW.poll_blocking();
            finished_samples += 1;
            
            flush |= state.interacting.load(Ordering::Relaxed) || state.dirty.load(Ordering::Relaxed);
            if flush {
                break;
            }
            if !state.running.load(Ordering::Relaxed) {
                return;
            }
        }
        state.samples.fetch_add(finished_samples, Ordering::Relaxed);

        // Readback from GPU
        let _ = output_buffer.read_blocking(&mut image_buffer_raw);
        let sample_count = state.samples.load(Ordering::Relaxed) as f32;
        for (i, col) in image_buffer_raw.iter().enumerate() {
            image_buffer[i * 3] = col.x / sample_count;
            image_buffer[i * 3 + 1] = col.y / sample_count;
            image_buffer[i * 3 + 2] = col.z / sample_count;
        }

        // Denoise
        #[cfg(feature = "oidn")]
        if state.denoise.load(Ordering::Relaxed) && !flush {
            denoise_image(screen_width as usize, screen_height as usize, &mut image_buffer);
        }

        // Push to render thread
        state.framebuffer.write().copy_from_slice(image_buffer.as_slice());

        // Interaction
        if flush {
            state.dirty.store(false, Ordering::Relaxed);
            state.samples.store(0, Ordering::Relaxed);
            let _ = config_buffer.write(&[*state.config.read()]);
            let _ = output_buffer.write(&vec![Vec4::ZERO; pixel_count as usize]);
            let _ = rng_buffer.write(if state.use_blue_noise.load(Ordering::Relaxed) { &rng_data_blue } else { &rng_data_uniform });
        }
    }
}

pub fn trace_cpu(
    scene_path: &str,
    skybox_path: Option<&str>,
    state: Arc<TracingState>,
) {
    let Some(world) = World::from_path(scene_path) else {
        return;
    };
    let mut skybox_image_buffer = fallback_cpu_buffer();
    let mut skybox_size = (2, 2);
    if let Some(skybox_source) = skybox_path.and_then(load_dynamic_image) {
        skybox_size = skybox_source.dimensions();
        skybox_image_buffer = dynamic_image_to_cpu_buffer(skybox_source);
    }
    let skybox_image = CpuImage::new(&skybox_image_buffer, skybox_size.0, skybox_size.1);

    let screen_width = state.config.read().width;
    let screen_height = state.config.read().height;
    let pixel_count = (screen_width * screen_height) as usize;
    let mut rng = rand::thread_rng();
    let mut rng_data_blue: Vec<UVec2> = vec![UVec2::ZERO; pixel_count];
    let mut rng_data_uniform: Vec<UVec2> = vec![UVec2::ZERO; pixel_count];
    for y in 0..screen_height {
        for x in 0..screen_width {
            let pixel_index = (y * screen_width + x) as usize;
            let pixel = BLUE_TEXTURE.get_pixel(x % BLUE_TEXTURE.width(), y % BLUE_TEXTURE.height())[0] as f32 / 255.0;
            rng_data_blue[pixel_index].x = 0;
            rng_data_blue[pixel_index].y = (pixel * 4294967295.0) as u32;
            rng_data_uniform[pixel_index].x = rand::Rng::gen(&mut rng);
        }
    }

    // Reset previous state, if there is any
    let samples_init = state.samples.load(Ordering::Relaxed) as f32;
    let mut output_buffer = state.framebuffer.read().chunks(3).map(|c| Vec4::new(c[0], c[1], c[2], 1.0) * samples_init).collect::<Vec<_>>();

    // Setup tracing state
    let pixel_count = (screen_width * screen_height) as u64;
    let mut rng_buffer = if state.use_blue_noise.load(Ordering::Relaxed) { &mut rng_data_blue } else { &mut rng_data_uniform };

    let mut image_buffer: Vec<f32> = vec![0.0; pixel_count as usize * 3];

    let atlas_width = world.atlas.width();
    let atlas_height = world.atlas.height();
    let atlas_buffer = dynamic_image_to_cpu_buffer(world.atlas);
    let atlas_image = CpuImage::new(&atlas_buffer, atlas_width, atlas_height);

    while state.running.load(Ordering::Relaxed) {
        // Dispatch
        let flush = state.interacting.load(Ordering::Relaxed) || state.dirty.load(Ordering::Relaxed);
        {
            let config = state.config.read();
            let outputs = output_buffer.par_chunks_mut(screen_width as usize).enumerate();
            let rngs = rng_buffer.par_chunks_mut(screen_width as usize);
            outputs.zip(rngs).for_each(|((y, output), rng)| {
                for x in 0..screen_width {
                    let (radiance, rng_state) = kernels::trace_pixel(
                        UVec3::new(x, y as u32, 1),
                        &config,
                        rng[x as usize],
                        &world.per_vertex_buffer,
                        &world.index_buffer,
                        &world.bvh.nodes,
                        &world.material_data_buffer,
                        &world.light_pick_buffer,
                        &shared_structs::Sampler,
                        &atlas_image,
                        &skybox_image,
                    );
                    output[x as usize] += radiance;
                    rng[x as usize] = rng_state;
                }
            });
        }
        state.samples.fetch_add(1, Ordering::Relaxed);

        // Readback from GPU
        let sample_count = state.samples.load(Ordering::Relaxed) as f32;
        for (i, col) in output_buffer.iter().enumerate() {
            image_buffer[i * 3] = col.x / sample_count;
            image_buffer[i * 3 + 1] = col.y / sample_count;
            image_buffer[i * 3 + 2] = col.z / sample_count;
        }

        // Denoise
        #[cfg(feature = "oidn")]
        if state.denoise.load(Ordering::Relaxed) && !flush {
            denoise_image(screen_width as usize, screen_height as usize, &mut image_buffer);
        }

        // Push to render thread
        state.framebuffer.write().copy_from_slice(image_buffer.as_slice());

        // Interaction
        if flush {
            state.dirty.store(false, Ordering::Relaxed);
            state.samples.store(0, Ordering::Relaxed);
            output_buffer = vec![Vec4::ZERO; pixel_count as usize];
            rng_buffer = if state.use_blue_noise.load(Ordering::Relaxed) { &mut rng_data_blue } else { &mut rng_data_uniform };
        }
    }
}

// Harness for running syncronous tracing
#[allow(dead_code)]
pub fn setup_trace(width: u32, height: u32, samples: u32) -> Arc<TracingState> {
    let state = Arc::new(TracingState::new(width, height));
    state.running.store(true, Ordering::Relaxed);
    {
        let state = state.clone();
        std::thread::spawn(move || {
            while state.samples.load(Ordering::Relaxed) < samples {
                std::thread::yield_now();
            }
            state.running.store(false, Ordering::Relaxed);
        });
    }
    state
}