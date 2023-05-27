const KERNEL: &[u8] = include_bytes!(env!("kernels.spv"));
const BLUE_BYTES: &[u8] = include_bytes!("resources/bluenoise.png");
lazy_static::lazy_static! {
    pub static ref FW: gpgpu::Framework = make_framework();
    pub static ref BLUE_TEXTURE: RgbaImage = Reader::new(Cursor::new(BLUE_BYTES)).with_guessed_format().unwrap().decode().unwrap().into_rgba8();
}

use glam::{UVec2, Vec4, UVec3};
use gpgpu::{
    BufOps, DescriptorSet, GpuBuffer, GpuBufferUsage, GpuUniformBuffer, Kernel, Program, Shader, Sampler, SamplerWrapMode, SamplerFilterMode
};
use image::{RgbaImage, io::Reader};
use pollster::FutureExt;
pub use shared_structs::TracingConfig;
use std::{sync::{
    atomic::{AtomicBool, AtomicU32, Ordering},
    Arc,
}, io::Cursor};
use parking_lot::RwLock;
use rayon::prelude::*;

use crate::asset::{World, GpuWorld};

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

struct PathTracingKernel<'fw>(Kernel<'fw>);

impl<'fw> PathTracingKernel<'fw> {
    fn new(
        config_buffer: &GpuUniformBuffer<'fw, TracingConfig>,
        rng_buffer: &GpuBuffer<'fw, UVec2>,
        output_buffer: &GpuBuffer<'fw, Vec4>,
        world: &GpuWorld<'fw>,
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
            .bind_const_image(&world.atlas);
        let program = Program::new(&shader, "trace_kernel").add_descriptor_set(bindings);
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

pub fn trace_gpu(
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
    let world = World::from_path(scene_path).into_gpu();

    let screen_width = config.read().width;
    let screen_height = config.read().height;
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

    let pixel_count = (screen_width * screen_height) as u64;
    let config_buffer = GpuUniformBuffer::from_slice(&FW, &[*config.read()]);
    let rng_buffer = GpuBuffer::from_slice(&FW, if use_blue_noise.load(Ordering::Relaxed) { &rng_data_blue } else { &rng_data_uniform });
    let output_buffer = GpuBuffer::with_capacity(&FW, pixel_count);

    let mut image_buffer_raw: Vec<Vec4> = vec![Vec4::ZERO; pixel_count as usize];
    let mut image_buffer: Vec<f32> = vec![0.0; pixel_count as usize * 3];

    let rt = PathTracingKernel::new(&config_buffer, &rng_buffer, &output_buffer, &world);

    while running.load(Ordering::Relaxed) {
        // Dispatch
        let sync_rate = sync_rate.load(Ordering::Relaxed);
        let mut flush = false;
        let mut finished_samples = 0;
        for _ in 0..sync_rate {
            rt.0.enqueue(screen_width.div_ceil(8), screen_height.div_ceil(8), 1);
            FW.poll_blocking();
            finished_samples += 1;
            
            flush |= interacting.load(Ordering::Relaxed) || dirty.load(Ordering::Relaxed);
            if flush {
                break;
            }
            if !running.load(Ordering::Relaxed) {
                return;
            }
        }
        samples.fetch_add(finished_samples, Ordering::Relaxed);

        // Readback from GPU
        output_buffer.read_blocking(&mut image_buffer_raw).unwrap();
        let sample_count = samples.load(Ordering::Relaxed) as f32;
        for (i, col) in image_buffer_raw.iter().enumerate() {
            image_buffer[i * 3] = col.x / sample_count;
            image_buffer[i * 3 + 1] = col.y / sample_count;
            image_buffer[i * 3 + 2] = col.z / sample_count;
        }

        // Denoise
        #[cfg(feature = "oidn")]
        if denoise.load(Ordering::Relaxed) && !flush {
            denoise_image(screen_width as usize, screen_height as usize, &mut image_buffer);
        }

        // Push to render thread
        framebuffer.write().copy_from_slice(image_buffer.as_slice());

        // Interaction
        if flush {
            dirty.store(false, Ordering::Relaxed);
            samples.store(0, Ordering::Relaxed);
            config_buffer.write(&[*config.read()]).unwrap();
            output_buffer.write(&vec![Vec4::ZERO; pixel_count as usize]).unwrap();
            rng_buffer.write(if use_blue_noise.load(Ordering::Relaxed) { &rng_data_blue } else { &rng_data_uniform }).unwrap();
        }
    }
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
            let pixel = BLUE_TEXTURE.get_pixel(x % BLUE_TEXTURE.width(), y % BLUE_TEXTURE.height())[0] as f32 / 255.0;
            rng_data_blue[pixel_index].x = 0;
            rng_data_blue[pixel_index].y = (pixel * 4294967295.0) as u32;
            rng_data_uniform[pixel_index].x = rand::Rng::gen(&mut rng);
        }
    }

    let pixel_count = (screen_width * screen_height) as u64;
    let mut rng_buffer = if use_blue_noise.load(Ordering::Relaxed) { &mut rng_data_blue } else { &mut rng_data_uniform };
    let mut output_buffer = vec![Vec4::ZERO; pixel_count as usize];

    let mut image_buffer: Vec<f32> = vec![0.0; pixel_count as usize * 3];

    let atlas_data = world.atlas.into_rgb8();
    let atlas_cpu: Vec<Vec4> = atlas_data.chunks(3).map(|f| Vec4::new(f[0] as f32, f[1] as f32, f[2] as f32, 1.0)).collect();
    let atlas_image = shared_structs::Image::new(&atlas_cpu, 0, 0);

    // This a bit of hack, but whatever
    while running.load(Ordering::Relaxed) {
        // Dispatch
        let flush = interacting.load(Ordering::Relaxed) || dirty.load(Ordering::Relaxed);
        {
            let config = config.read();
            let outputs = output_buffer.par_chunks_mut(screen_width as usize).enumerate();
            let rngs = rng_buffer.par_chunks_mut(screen_width as usize);
            outputs.zip(rngs).for_each(|((y, output), rng)| {
                for x in 0..screen_width {
                    kernels::trace_pixel(
                        UVec3::new(x, y as u32, 1),
                        &config,
                        &mut rng[x as usize],
                        &mut output[x as usize],
                        &world.per_vertex_buffer,
                        &world.index_buffer,
                        &world.bvh.nodes,
                        &world.material_data_buffer,
                        &world.light_pick_buffer,
                        &shared_structs::Sampler,
                        &atlas_image
                    );
                }
            });
        }
        samples.fetch_add(1, Ordering::Relaxed);

        // Readback from GPU
        let sample_count = samples.load(Ordering::Relaxed) as f32;
        for (i, col) in output_buffer.iter().enumerate() {
            image_buffer[i * 3] = col.x / sample_count;
            image_buffer[i * 3 + 1] = col.y / sample_count;
            image_buffer[i * 3 + 2] = col.z / sample_count;
        }

        // Denoise
        #[cfg(feature = "oidn")]
        if denoise.load(Ordering::Relaxed) && !flush {
            denoise_image(screen_width as usize, screen_height as usize, &mut image_buffer);
        }

        // Push to render thread
        framebuffer.write().copy_from_slice(image_buffer.as_slice());

        // Interaction
        if flush {
            dirty.store(false, Ordering::Relaxed);
            samples.store(0, Ordering::Relaxed);
            output_buffer = vec![Vec4::ZERO; pixel_count as usize];
            rng_buffer = if use_blue_noise.load(Ordering::Relaxed) { &mut rng_data_blue } else { &mut rng_data_uniform };
        }
    }
}
