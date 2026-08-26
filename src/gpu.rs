use std::borrow::Cow;
use std::marker::PhantomData;
use std::sync::OnceLock;

use bytemuck::Pod;
use wgpu::util::DeviceExt;

pub type BufferError = wgpu::BufferAsyncError;

pub struct Framework {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    passthrough: bool,
}

impl Framework {
    pub async fn new(adapter: wgpu::Adapter) -> Self {
        let wanted = wgpu::Features::FLOAT32_FILTERABLE | wgpu::Features::PASSTHROUGH_SHADERS;
        let required_features = adapter.features() & wanted;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features,
                required_limits: adapter.limits(),
                ..Default::default()
            })
            .await
            .expect("Failed at device creation.");
        Self {
            device,
            queue,
            passthrough: required_features.contains(wgpu::Features::PASSTHROUGH_SHADERS),
        }
    }

    pub fn poll_blocking(&self) {
        let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
    }
}

pub trait BufOps<'fw, T: Pod>: Sized {
    fn from_slice(fw: &'fw Framework, slice: &[T]) -> Self;
    fn write(&self, buf: &[T]) -> Result<(), BufferError>;
    fn read_blocking(&self, buf: &mut [T]) -> Result<(), BufferError>;
}

pub struct GpuBuffer<'fw, T: Pod> {
    fw: &'fw Framework,
    buf: wgpu::Buffer,
    staging: OnceLock<wgpu::Buffer>,
    _marker: PhantomData<T>,
}

impl<'fw, T: Pod> BufOps<'fw, T> for GpuBuffer<'fw, T> {
    fn from_slice(fw: &'fw Framework, slice: &[T]) -> Self {
        let buf = fw.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(slice),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
        });
        Self { fw, buf, staging: OnceLock::new(), _marker: PhantomData }
    }

    fn write(&self, buf: &[T]) -> Result<(), BufferError> {
        self.fw.queue.write_buffer(&self.buf, 0, bytemuck::cast_slice(buf));
        Ok(())
    }

    fn read_blocking(&self, buf: &mut [T]) -> Result<(), BufferError> {
        let size = self.buf.size();
        let staging = self.staging.get_or_init(|| {
            self.fw.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        });
        let mut encoder = self.fw.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        encoder.copy_buffer_to_buffer(&self.buf, 0, staging, 0, size);
        self.fw.queue.submit(Some(encoder.finish()));

        let (tx, rx) = std::sync::mpsc::channel();
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, move |res| { let _ = tx.send(res); });
        self.fw.poll_blocking();
        rx.recv().unwrap()?;
        {
            let mapped = slice.get_mapped_range().unwrap();
            let bytes = bytemuck::cast_slice_mut(buf);
            bytes.copy_from_slice(&mapped[..bytes.len()]);
        }
        staging.unmap();
        Ok(())
    }
}

pub struct GpuUniformBuffer<'fw, T: Pod> {
    fw: &'fw Framework,
    buf: wgpu::Buffer,
    _marker: PhantomData<T>,
}

impl<'fw, T: Pod> BufOps<'fw, T> for GpuUniformBuffer<'fw, T> {
    fn from_slice(fw: &'fw Framework, slice: &[T]) -> Self {
        let buf = fw.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(slice),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        Self { fw, buf, _marker: PhantomData }
    }

    fn write(&self, buf: &[T]) -> Result<(), BufferError> {
        self.fw.queue.write_buffer(&self.buf, 0, bytemuck::cast_slice(buf));
        Ok(())
    }

    fn read_blocking(&self, _buf: &mut [T]) -> Result<(), BufferError> {
        unimplemented!("Uniform buffers are write-only")
    }
}

pub mod primitives {
    pub trait PixelInfo {
        fn byte_size() -> usize;
        fn wgpu_format() -> wgpu::TextureFormat;
    }

    pub mod pixels {
        use super::PixelInfo;

        pub struct Rgba8UintNorm;
        impl PixelInfo for Rgba8UintNorm {
            fn byte_size() -> usize { 4 }
            fn wgpu_format() -> wgpu::TextureFormat { wgpu::TextureFormat::Rgba8Unorm }
        }

        pub struct Rgba32Float;
        impl PixelInfo for Rgba32Float {
            fn byte_size() -> usize { 16 }
            fn wgpu_format() -> wgpu::TextureFormat { wgpu::TextureFormat::Rgba32Float }
        }
    }
}

pub trait ImgOps<'fw>: Sized {
    fn from_bytes(fw: &'fw Framework, data: &[u8], width: u32, height: u32) -> Self;
}

pub struct GpuConstImage<'fw, P: primitives::PixelInfo> {
    #[allow(dead_code)]
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    _marker: PhantomData<(&'fw Framework, P)>,
}

impl<'fw, P: primitives::PixelInfo> ImgOps<'fw> for GpuConstImage<'fw, P> {
    fn from_bytes(fw: &'fw Framework, data: &[u8], width: u32, height: u32) -> Self {
        let texture = fw.device.create_texture_with_data(
            &fw.queue,
            &wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width, height, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: P::wgpu_format(),
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            data,
        );
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self { texture, view, _marker: PhantomData }
    }
}

pub enum SamplerWrapMode {
    ClampToEdge,
    Repeat,
}

pub enum SamplerFilterMode {
    Nearest,
    Linear,
}

pub struct Sampler {
    sampler: wgpu::Sampler,
}

impl Sampler {
    pub fn new(fw: &Framework, wrap_mode: SamplerWrapMode, filter_mode: SamplerFilterMode) -> Self {
        let address_mode = match wrap_mode {
            SamplerWrapMode::ClampToEdge => wgpu::AddressMode::ClampToEdge,
            SamplerWrapMode::Repeat => wgpu::AddressMode::Repeat,
        };
        let filter = match filter_mode {
            SamplerFilterMode::Nearest => wgpu::FilterMode::Nearest,
            SamplerFilterMode::Linear => wgpu::FilterMode::Linear,
        };
        let sampler = fw.device.create_sampler(&wgpu::SamplerDescriptor {
            label: None,
            address_mode_u: address_mode,
            address_mode_v: address_mode,
            address_mode_w: address_mode,
            mag_filter: filter,
            min_filter: filter,
            ..Default::default()
        });
        Self { sampler }
    }
}

pub enum GpuBufferUsage {
    ReadOnly,
    ReadWrite,
}

#[derive(Default)]
pub struct DescriptorSet<'res> {
    layout_entries: Vec<wgpu::BindGroupLayoutEntry>,
    resources: Vec<wgpu::BindingResource<'res>>,
}

impl<'res> DescriptorSet<'res> {
    fn bind(mut self, ty: wgpu::BindingType, resource: wgpu::BindingResource<'res>) -> Self {
        self.layout_entries.push(wgpu::BindGroupLayoutEntry {
            binding: self.layout_entries.len() as u32,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        });
        self.resources.push(resource);
        self
    }

    pub fn bind_uniform_buffer<T: Pod>(self, buffer: &'res GpuUniformBuffer<T>) -> Self {
        let ty = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        self.bind(ty, buffer.buf.as_entire_binding())
    }

    pub fn bind_buffer<T: Pod>(self, buffer: &'res GpuBuffer<T>, usage: GpuBufferUsage) -> Self {
        let ty = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: matches!(usage, GpuBufferUsage::ReadOnly) },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        self.bind(ty, buffer.buf.as_entire_binding())
    }

    pub fn bind_sampler(self, sampler: &'res Sampler) -> Self {
        let ty = wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering);
        self.bind(ty, wgpu::BindingResource::Sampler(&sampler.sampler))
    }

    pub fn bind_const_image<P: primitives::PixelInfo>(self, image: &'res GpuConstImage<P>) -> Self {
        let ty = wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        };
        self.bind(ty, wgpu::BindingResource::TextureView(&image.view))
    }
}

pub struct Shader {
    words: Vec<u32>,
}

impl Shader {
    pub fn from_spirv_bytes(_fw: &Framework, bytes: &[u8], _name: Option<&str>) -> Self {
        Self { words: bytemuck::pod_collect_to_vec(bytes) }
    }

    // Reads the LocalSize execution mode of the given entry point.
    fn workgroup_size(&self, entry_point: &str) -> (u32, u32, u32) {
        const OP_ENTRY_POINT: u32 = 15;
        const OP_EXECUTION_MODE: u32 = 16;
        const MODE_LOCAL_SIZE: u32 = 17;
        let mut entry_id = None;
        let mut i = 5;
        while i < self.words.len() {
            let word_count = (self.words[i] >> 16) as usize;
            let opcode = self.words[i] & 0xffff;
            let operands = &self.words[i + 1..i + word_count];
            if opcode == OP_ENTRY_POINT {
                let name_bytes: Vec<u8> = operands[2..].iter().flat_map(|w| w.to_le_bytes()).collect();
                let name = name_bytes.split(|&b| b == 0).next().unwrap_or(&[]);
                if name == entry_point.as_bytes() {
                    entry_id = Some(operands[1]);
                }
            } else if opcode == OP_EXECUTION_MODE && Some(operands[0]) == entry_id && operands[1] == MODE_LOCAL_SIZE {
                return (operands[2], operands[3], operands[4]);
            }
            i += word_count;
        }
        (1, 1, 1)
    }
}

pub struct Program<'sha, 'res> {
    shader: &'sha Shader,
    entry_point: &'sha str,
    descriptor_sets: Vec<DescriptorSet<'res>>,
}

impl<'sha, 'res> Program<'sha, 'res> {
    pub fn new(shader: &'sha Shader, entry_point: &'sha str) -> Self {
        Self { shader, entry_point, descriptor_sets: Vec::new() }
    }

    pub fn add_descriptor_set(mut self, desc: DescriptorSet<'res>) -> Self {
        self.descriptor_sets.push(desc);
        self
    }
}

pub struct Kernel<'fw> {
    fw: &'fw Framework,
    pipeline: wgpu::ComputePipeline,
    bind_groups: Vec<wgpu::BindGroup>,
}

impl<'fw> Kernel<'fw> {
    pub fn new(fw: &'fw Framework, program: Program) -> Self {
        let device = &fw.device;
        let module = if fw.passthrough {
            let entry_points = [wgpu::PassthroughShaderEntryPoint {
                name: program.entry_point.into(),
                workgroup_size: program.shader.workgroup_size(program.entry_point),
            }];
            unsafe {
                device.create_shader_module_passthrough(wgpu::ShaderModuleDescriptorPassthrough {
                    label: None,
                    entry_points: Cow::Borrowed(&entry_points),
                    spirv: Some(Cow::Borrowed(&program.shader.words)),
                    ..Default::default()
                })
            }
        } else {
            device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::SpirV(Cow::Borrowed(&program.shader.words)),
            })
        };

        let layouts: Vec<wgpu::BindGroupLayout> = program
            .descriptor_sets
            .iter()
            .map(|set| {
                device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: &set.layout_entries,
                })
            })
            .collect();
        let layout_refs: Vec<Option<&wgpu::BindGroupLayout>> = layouts.iter().map(Some).collect();
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &layout_refs,
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(program.entry_point),
            compilation_options: Default::default(),
            cache: None,
        });
        let bind_groups = program
            .descriptor_sets
            .iter()
            .zip(&layouts)
            .map(|(set, layout)| {
                let entries: Vec<wgpu::BindGroupEntry> = set
                    .resources
                    .iter()
                    .enumerate()
                    .map(|(i, resource)| wgpu::BindGroupEntry { binding: i as u32, resource: resource.clone() })
                    .collect();
                device.create_bind_group(&wgpu::BindGroupDescriptor { label: None, layout, entries: &entries })
            })
            .collect();

        Self { fw, pipeline, bind_groups }
    }

    pub fn enqueue(&self, x: u32, y: u32, z: u32) {
        let mut encoder = self.fw.device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(&self.pipeline);
            for (i, bind_group) in self.bind_groups.iter().enumerate() {
                pass.set_bind_group(i as u32, bind_group, &[]);
            }
            pass.dispatch_workgroups(x, y, z);
        }
        self.fw.queue.submit(Some(encoder.finish()));
    }
}
