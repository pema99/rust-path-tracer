
const KERNEL: &[u8] = include_bytes!(env!("compute.spv"));

lazy_static::lazy_static! {
    static ref FW: gpgpu::Framework = gpgpu::Framework::default();
}

use glam::{UVec2, Vec4};
use gpgpu::{BufOps, DescriptorSet, Framework, GpuBuffer, GpuBufferUsage, Kernel, Program, Shader};
use std::{rc::Rc, sync::{Arc, atomic::{AtomicBool, Ordering, AtomicU32}, Mutex}};

#[derive(Copy, Clone)]
pub struct Config {
    pub width: u32,
    pub height: u32,
    pub samples: u32,
}

#[allow(dead_code)]
struct RayGenKernel<'fw> {
    ray_origin_buffer: Rc<GpuBuffer<'fw, Vec4>>,
    ray_dir_buffer: Rc<GpuBuffer<'fw, Vec4>>,
    throughput_buffer: Rc<GpuBuffer<'fw, Vec4>>,
    rng_buffer: Rc<GpuBuffer<'fw, UVec2>>,
    kernel: Kernel<'fw>,
}

impl<'fw> RayGenKernel<'fw> {
    fn new(
        ray_origin_buffer: Rc<GpuBuffer<'fw, Vec4>>,
        ray_dir_buffer: Rc<GpuBuffer<'fw, Vec4>>,
        throughput_buffer: Rc<GpuBuffer<'fw, Vec4>>,
        rng_buffer: Rc<GpuBuffer<'fw, UVec2>>,
    ) -> Self {
        let shader = Shader::from_spirv_bytes(&FW, KERNEL, Some("compute"));
        let bindings = DescriptorSet::default()
            .bind_buffer(&ray_origin_buffer, GpuBufferUsage::ReadWrite)
            .bind_buffer(&ray_dir_buffer, GpuBufferUsage::ReadWrite)
            .bind_buffer(&throughput_buffer, GpuBufferUsage::ReadWrite)
            .bind_buffer(&rng_buffer, GpuBufferUsage::ReadOnly);
        let program = Program::new(&shader, "main_raygen").add_descriptor_set(bindings);
        let kernel = Kernel::new(&FW, program);

        Self {
            ray_origin_buffer,
            ray_dir_buffer,
            throughput_buffer,
            rng_buffer,
            kernel,
        }
    }
}

#[allow(dead_code)]
struct RayTraceKernel<'fw> {
    ray_origin_buffer: Rc<GpuBuffer<'fw, Vec4>>,
    ray_dir_buffer: Rc<GpuBuffer<'fw, Vec4>>,
    kernel: Kernel<'fw>,
}

impl<'fw> RayTraceKernel<'fw> {
    fn new(
        ray_origin_buffer: Rc<GpuBuffer<'fw, Vec4>>,
        ray_dir_buffer: Rc<GpuBuffer<'fw, Vec4>>,
    ) -> Self {
        let shader = Shader::from_spirv_bytes(&FW, KERNEL, Some("compute"));
        let bindings = DescriptorSet::default()
            .bind_buffer(&ray_origin_buffer, GpuBufferUsage::ReadWrite)
            .bind_buffer(&ray_dir_buffer, GpuBufferUsage::ReadWrite);
        let program = Program::new(&shader, "main_raytrace").add_descriptor_set(bindings);
        let kernel = Kernel::new(&FW, program);

        Self {
            ray_origin_buffer,
            ray_dir_buffer,
            kernel,
        }
    }
}

#[allow(dead_code)]
struct MaterialKernel<'fw> {
    ray_origin_buffer: Rc<GpuBuffer<'fw, Vec4>>,
    ray_dir_buffer: Rc<GpuBuffer<'fw, Vec4>>,
    throughput_buffer: Rc<GpuBuffer<'fw, Vec4>>,
    rng_buffer: Rc<GpuBuffer<'fw, UVec2>>,
    output_buffer: Arc<GpuBuffer<'fw, Vec4>>,
    kernel: Kernel<'fw>,
}

impl<'fw> MaterialKernel<'fw> {
    fn new(
        ray_origin_buffer: Rc<GpuBuffer<'fw, Vec4>>,
        ray_dir_buffer: Rc<GpuBuffer<'fw, Vec4>>,
        throughput_buffer: Rc<GpuBuffer<'fw, Vec4>>,
        rng_buffer: Rc<GpuBuffer<'fw, UVec2>>,
        output_buffer: Arc<GpuBuffer<'fw, Vec4>>,
    ) -> Self {
        let shader = Shader::from_spirv_bytes(&FW, KERNEL, Some("compute"));
        let bindings = DescriptorSet::default()
            .bind_buffer(&ray_origin_buffer, GpuBufferUsage::ReadWrite)
            .bind_buffer(&ray_dir_buffer, GpuBufferUsage::ReadWrite)
            .bind_buffer(&throughput_buffer, GpuBufferUsage::ReadWrite)
            .bind_buffer(&rng_buffer, GpuBufferUsage::ReadWrite)
            .bind_buffer(&output_buffer, GpuBufferUsage::ReadWrite);
        let program = Program::new(&shader, "main_material").add_descriptor_set(bindings);
        let kernel = Kernel::new(&FW, program);

        Self {
            ray_origin_buffer,
            ray_dir_buffer,
            throughput_buffer,
            rng_buffer,
            output_buffer,
            kernel,
        }
    }
}

#[cfg(feature = "oidn")]
fn denoise(config: &Config, input: &Vec<f32>) -> Vec<f32> {
    use oidn::filter;
    let mut filter_output = vec![0.0f32; input.len()];
    let device = oidn::Device::new();
    oidn::RayTracing::new(&device)
        .srgb(true)
        .image_dimensions(config.width as usize, config.height as usize)
        .filter(&input[..], &mut filter_output[..])
        .expect("Filter config error!");
    filter_output
}

pub fn trace(framebuffer: Arc<Mutex<Vec<f32>>>, running: Arc<AtomicBool>) {
    let config = Config {
        width: 1280,
        height: 720,
        samples: 1,
    };
    let fw = Framework::default();

    let mut rng = rand::thread_rng();
    let mut rng_data = Vec::new();
    for _ in 0..(config.width * config.height) {
        rng_data.push(UVec2::new(
            rand::Rng::gen(&mut rng),
            rand::Rng::gen(&mut rng),
        ));
    }

    let pixel_count = (config.width * config.height) as u64;
    let ray_origin_buffer = Rc::new(GpuBuffer::with_capacity(&FW, pixel_count));
    let ray_dir_buffer = Rc::new(GpuBuffer::with_capacity(&FW, pixel_count));
    let throughput_buffer = Rc::new(GpuBuffer::with_capacity(&FW, pixel_count));
    let rng_buffer = Rc::new(GpuBuffer::from_slice(&FW, &rng_data));
    let output_buffer = Arc::new(GpuBuffer::with_capacity(&FW, pixel_count));

    let rg = RayGenKernel::new(
        ray_origin_buffer.clone(),
        ray_dir_buffer.clone(),
        throughput_buffer.clone(),
        rng_buffer.clone(),
    );
    let rt = RayTraceKernel::new(ray_origin_buffer.clone(), ray_dir_buffer.clone());
    let mt = MaterialKernel::new(
        ray_origin_buffer.clone(),
        ray_dir_buffer.clone(),
        throughput_buffer.clone(),
        rng_buffer.clone(),
        output_buffer.clone(),
    );

    let bounces = 4;
    let samples = Arc::new(AtomicU32::new(0));
    {
        #[allow(dead_code)]
        let config = config.clone();
        let samples = samples.clone();
        let running = running.clone();
        std::thread::spawn(move || {
            while running.load(Ordering::Relaxed) {
                // Readback
                let sample_count = samples.load(Ordering::Relaxed) as f32;
                #[allow(unused_mut)]
                let mut image_buffer: Vec<f32> = mt
                    .output_buffer
                    .read_vec_blocking()
                    .unwrap()
                    .iter()
                    .map(|&x| x / sample_count)
                    .flat_map(|x| vec![x.x, x.y, x.z])
                    .collect();

                #[cfg(feature = "oidn")]
                {
                    image_buffer = denoise(&config, &image_buffer);
                }

                // Readback
                match framebuffer.lock() {
                    Ok(mut fb) => {
                        fb.copy_from_slice(&image_buffer);
                    }
                    Err(e) => {
                        println!("Error: {}", e);
                    }
                }
            }
        });
    }

    while running.load(Ordering::Relaxed) {
        for _ in 0..1 {
            rg.kernel.enqueue(config.width / 64, config.height, 1);
            for _ in 0..bounces {
                rt.kernel.enqueue(config.width / 64, config.height, 1);
                mt.kernel.enqueue(config.width / 64, config.height, 1);
            }
            samples.fetch_add(1, Ordering::Relaxed);
        }
    }
}
