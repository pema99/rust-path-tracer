const KERNEL: &[u8] = include_bytes!(env!("compute.spv"));

lazy_static::lazy_static! {
    static ref FW: gpgpu::Framework = gpgpu::Framework::default();
}

use glam::{UVec2, Vec4};
use gpgpu::{
    BufOps, DescriptorSet, GpuBuffer, GpuBufferUsage, GpuUniformBuffer, Kernel, Program, Shader,
};
pub use shared_structs::TracingConfig;
use std::{
    rc::Rc,
    sync::{
        atomic::{AtomicBool, AtomicU32, Ordering},
        Arc, Mutex,
    },
};

#[allow(dead_code)]
struct RayGenKernel<'fw> {
    config_buffer: Rc<GpuUniformBuffer<'fw, TracingConfig>>,
    ray_origin_buffer: Rc<GpuBuffer<'fw, Vec4>>,
    ray_dir_buffer: Rc<GpuBuffer<'fw, Vec4>>,
    throughput_buffer: Rc<GpuBuffer<'fw, Vec4>>,
    rng_buffer: Rc<GpuBuffer<'fw, UVec2>>,
    kernel: Kernel<'fw>,
}

impl<'fw> RayGenKernel<'fw> {
    fn new(
        config_buffer: Rc<GpuUniformBuffer<'fw, TracingConfig>>,
        ray_origin_buffer: Rc<GpuBuffer<'fw, Vec4>>,
        ray_dir_buffer: Rc<GpuBuffer<'fw, Vec4>>,
        throughput_buffer: Rc<GpuBuffer<'fw, Vec4>>,
        rng_buffer: Rc<GpuBuffer<'fw, UVec2>>,
    ) -> Self {
        let shader = Shader::from_spirv_bytes(&FW, KERNEL, Some("compute"));
        let bindings = DescriptorSet::default()
            .bind_uniform_buffer(&config_buffer)
            .bind_buffer(&ray_origin_buffer, GpuBufferUsage::ReadWrite)
            .bind_buffer(&ray_dir_buffer, GpuBufferUsage::ReadWrite)
            .bind_buffer(&throughput_buffer, GpuBufferUsage::ReadWrite)
            .bind_buffer(&rng_buffer, GpuBufferUsage::ReadOnly);
        let program = Program::new(&shader, "main_raygen").add_descriptor_set(bindings);
        let kernel = Kernel::new(&FW, program);

        Self {
            config_buffer,
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
    config_buffer: Rc<GpuUniformBuffer<'fw, TracingConfig>>,
    ray_origin_buffer: Rc<GpuBuffer<'fw, Vec4>>,
    ray_dir_buffer: Rc<GpuBuffer<'fw, Vec4>>,
    kernel: Kernel<'fw>,
}

impl<'fw> RayTraceKernel<'fw> {
    fn new(
        config_buffer: Rc<GpuUniformBuffer<'fw, TracingConfig>>,
        ray_origin_buffer: Rc<GpuBuffer<'fw, Vec4>>,
        ray_dir_buffer: Rc<GpuBuffer<'fw, Vec4>>,
    ) -> Self {
        let shader = Shader::from_spirv_bytes(&FW, KERNEL, Some("compute"));
        let bindings = DescriptorSet::default()
            .bind_uniform_buffer(&config_buffer)
            .bind_buffer(&ray_origin_buffer, GpuBufferUsage::ReadWrite)
            .bind_buffer(&ray_dir_buffer, GpuBufferUsage::ReadWrite);
        let program = Program::new(&shader, "main_raytrace").add_descriptor_set(bindings);
        let kernel = Kernel::new(&FW, program);

        Self {
            config_buffer,
            ray_origin_buffer,
            ray_dir_buffer,
            kernel,
        }
    }
}

#[allow(dead_code)]
struct MaterialKernel<'fw> {
    config_buffer: Rc<GpuUniformBuffer<'fw, TracingConfig>>,
    ray_origin_buffer: Rc<GpuBuffer<'fw, Vec4>>,
    ray_dir_buffer: Rc<GpuBuffer<'fw, Vec4>>,
    throughput_buffer: Rc<GpuBuffer<'fw, Vec4>>,
    rng_buffer: Rc<GpuBuffer<'fw, UVec2>>,
    output_buffer: Arc<GpuBuffer<'fw, Vec4>>,
    kernel: Kernel<'fw>,
}

impl<'fw> MaterialKernel<'fw> {
    fn new(
        config_buffer: Rc<GpuUniformBuffer<'fw, TracingConfig>>,
        ray_origin_buffer: Rc<GpuBuffer<'fw, Vec4>>,
        ray_dir_buffer: Rc<GpuBuffer<'fw, Vec4>>,
        throughput_buffer: Rc<GpuBuffer<'fw, Vec4>>,
        rng_buffer: Rc<GpuBuffer<'fw, UVec2>>,
        output_buffer: Arc<GpuBuffer<'fw, Vec4>>,
    ) -> Self {
        let shader = Shader::from_spirv_bytes(&FW, KERNEL, Some("compute"));
        let bindings = DescriptorSet::default()
            .bind_uniform_buffer(&config_buffer)
            .bind_buffer(&ray_origin_buffer, GpuBufferUsage::ReadWrite)
            .bind_buffer(&ray_dir_buffer, GpuBufferUsage::ReadWrite)
            .bind_buffer(&throughput_buffer, GpuBufferUsage::ReadWrite)
            .bind_buffer(&rng_buffer, GpuBufferUsage::ReadWrite)
            .bind_buffer(&output_buffer, GpuBufferUsage::ReadWrite);
        let program = Program::new(&shader, "main_material").add_descriptor_set(bindings);
        let kernel = Kernel::new(&FW, program);

        Self {
            config_buffer,
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
fn denoise_image(config: &TracingConfig, input: &mut [f32]) {
    let mut filter_output = vec![0.0f32; input.len()];
    let device = oidn::Device::new();
    oidn::RayTracing::new(&device)
        .srgb(true)
        .image_dimensions(config.width as usize, config.height as usize)
        .filter_in_place(input)
        .expect("Filter config error!");
}

pub fn trace(
    framebuffer: Arc<Mutex<Vec<f32>>>,
    running: Arc<AtomicBool>,
    samples_readback: Arc<AtomicU32>,
    #[allow(unused_variables)] denoise: Arc<AtomicBool>,
    config: TracingConfig,
) {
    let mut rng = rand::thread_rng();
    let mut rng_data = Vec::new();
    for _ in 0..(config.width * config.height) {
        rng_data.push(UVec2::new(
            rand::Rng::gen(&mut rng),
            rand::Rng::gen(&mut rng),
        ));
    }

    let pixel_count = (config.width * config.height) as u64;
    let config_buffer = Rc::new(GpuUniformBuffer::<TracingConfig>::from_slice(
        &FW,
        &[config],
    ));
    let ray_origin_buffer = Rc::new(GpuBuffer::with_capacity(&FW, pixel_count));
    let ray_dir_buffer = Rc::new(GpuBuffer::with_capacity(&FW, pixel_count));
    let throughput_buffer = Rc::new(GpuBuffer::with_capacity(&FW, pixel_count));
    let rng_buffer = Rc::new(GpuBuffer::from_slice(&FW, &rng_data));
    let output_buffer = Arc::new(GpuBuffer::with_capacity(&FW, pixel_count));

    let mut image_buffer_raw: Vec<Vec4> = vec![Vec4::ZERO; pixel_count as usize];
    let mut image_buffer: Vec<f32> = vec![0.0; pixel_count as usize * 3];

    let rg = RayGenKernel::new(
        config_buffer.clone(),
        ray_origin_buffer.clone(),
        ray_dir_buffer.clone(),
        throughput_buffer.clone(),
        rng_buffer.clone(),
    );
    let rt = RayTraceKernel::new(
        config_buffer.clone(),
        ray_origin_buffer.clone(),
        ray_dir_buffer.clone(),
    );
    let mt = MaterialKernel::new(
        config_buffer.clone(),
        ray_origin_buffer.clone(),
        ray_dir_buffer.clone(),
        throughput_buffer.clone(),
        rng_buffer.clone(),
        output_buffer.clone(),
    );

    let bounces = 4;
    let samples = Arc::new(AtomicU32::new(0));
    {
        #[allow(unused_variables)]
        let config = config.clone();
        let samples = samples.clone();
        let samples_readback = samples_readback.clone();
        let running = running.clone();
        std::thread::spawn(move || {
            while running.load(Ordering::Relaxed) {
                let sample_count = samples.load(Ordering::Relaxed) as f32;

                output_buffer.read_blocking(&mut image_buffer_raw).unwrap();
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
                    Ok(mut fb) => {
                        fb.copy_from_slice(image_buffer.as_slice());
                    }
                    Err(e) => {
                        println!("Error: {}", e);
                    }
                }
                samples_readback.store(sample_count as u32, Ordering::Relaxed);
            }
        });
    }

    while running.load(Ordering::Relaxed) {
        rg.kernel
            .enqueue(config.width.div_ceil(64), config.height, 1);
        for _ in 0..bounces {
            rt.kernel
                .enqueue(config.width.div_ceil(64), config.height, 1);
            mt.kernel
                .enqueue(config.width.div_ceil(64), config.height, 1);
        }
        samples.fetch_add(1, Ordering::Relaxed);

        // If we're scheduling too much work, and the readback thread can't keep up,
        // yield for a bit to let it catch up.
        while samples.load(Ordering::Relaxed) - samples_readback.load(Ordering::Relaxed) > 128 {
            std::thread::yield_now();
        }
    }
}
